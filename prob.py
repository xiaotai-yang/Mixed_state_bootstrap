from tensor_contraction import *
from coord_index import *
from util import *
from jax.random import PRNGKey, split, uniform, randint, bernoulli

def _bulk_step(p, str_col):
    """
    Returns a lax.scan step function that
    does one row of the 'bulk' contraction.
    Closes over `p` and `str_col` (both static).
    """
    def step(carry, row):
        tensor, log_s = carry
        # row[0] is the first slice, row[-1] the last
        t_ini = jnp.einsum("abcd, de->abce", row[0], inverse_boundary_T(p))[..., 0]
        t_end = jnp.einsum("abcd, d->abc", row[-1],   boundary_T_tensor(p))
        t_arr = (t_ini, *row[1:-1], t_end)
        new_tensor, step_scale = process_col(tensor, t_arr, str_col)
        return (new_tensor, log_s + step_scale), None
    return step

@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def log_Pr_bulk_pre_func(n, p, config_init, str_ini, str_col, use_BC):
    # 1) build full config_init of shape (5n × 3n)

    # 2) if BC‐mode, slice down to (3n × 2n) and reset loop dims
    if use_BC:
        config_init = config_init[n:4*n, :2*n]
        x, y = n-1, 2*n
    else:
        x, y = 2*n-1, 3*n

    # 3) build our two‐state template and index it
    T0 = full_tensor(0, p)     # → shape (2,2,2,2)
    T1 = full_tensor(1, p)     # → shape (2,2,2,2)
    templates = jnp.stack([T0, T1], axis=0)
    tensor_arr = templates[config_init]  # shape (x, 2,2,2,2)

    # 4) do the “ABC” contraction loop
    tensor, log_scale = process_initial(p, tensor_arr[0], str_ini)
    step_fn = _bulk_step(p, str_col)
    (tensor, log_scale), _ = lax.scan(step_fn, (tensor, log_scale), tensor_arr[1:x])

    tensor_end, log_scale_end = process_initial(p, tensor_arr[-1], str_ini)
    (tensor_end, log_scale_end), _ = lax.scan(step_fn, (tensor_end, log_scale_end), tensor_arr[-2:-x-1:-1])
    return tensor, log_scale, tensor_end, log_scale_end


@partial(jax.jit, static_argnums=(0, 3, 4))
def log_Pr_bulk_func(n, p, config_init, str_col, use_BC, tensor, log_scale, tensor_end, log_scale_end):
    # 1) if BC‐mode, slice down to (3n × 2n) and reset loop dims
    if use_BC:
        config_init = config_init[n:4*n, :2*n]
        init_x = n - 1
    else:
        init_x = 2*n - 1

    # 2) build our two‐state template and index it
    T0 = full_tensor(0, p)     # → shape (2,2,2,2)
    T1 = full_tensor(1, p)     # → shape (2,2,2,2)
    templates = jnp.stack([T0, T1], axis=0)
    tensor_arr = templates[config_init]  # shape (x,2,2,2,2)

    step_fn = _bulk_step(p, str_col)
    (tensor, log_scale), _ = lax.scan(step_fn, (tensor, log_scale), tensor_arr[init_x:-init_x])

    Pr  = jnp.sum(tensor * tensor_end)
    log_scale += log_scale_end
    return jnp.log(Pr) + log_scale


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
def log_Pr_AB_func(n, p, plaq_active, str_shrink, str_AB_col_ini, str_AB_col, str_AB_col_end, str_expand,  tensor, log_scale_total, tensor_end, log_scale_end):

    num_plaq_main = 15*n**2 - 5*n - n*(n+2) + 2 + (2*n - 1)*2
    num_middle = 3 * (n - 1) - 1
    num_corner = 4

    T0 = full_tensor(0, p); T1 = full_tensor(1, p)
    main_templates = jnp.stack([T0, T1], axis=0)  # shape (2, 2,2,2,2)
    IE0 = inner_edge_tensor(0, p); IE1 = inner_edge_tensor(1, p)
    mid_templates = jnp.stack([IE0, IE1], axis=0)
    C0 = inner_edge_corner_tensor(0, p); C1 = inner_edge_corner_tensor(1, p)
    corner_templates = jnp.stack([C0, C1], axis=0)

    # The loop part
    loop = (jnp.sum(plaq_active[2*n-1:3*n+1, :n+1]) - plaq_active[2*n-1, n] - plaq_active[3*n, n]) % 2
    plaq_active = plaq_active.at[2*n-1:3*n+1, :n].set(0)
    plaq_active = plaq_active.at[2*n:3*n, n].set(0)
    plaq_active = plaq_active.at[2*n-1, 0].set(loop)
    plaq_index = jnp.arange(num_plaq_main + num_middle + num_corner)
    plaq_coord = batch_index_to_coord_AB(n, plaq_index, num_plaq_main, num_middle)

    config_init = plaq_active[plaq_coord[0], plaq_coord[1]]
    tensor_arr = main_templates[config_init[:num_plaq_main]]
    t_middle = mid_templates[config_init[num_plaq_main:num_plaq_main+num_middle]]
    t_corner = corner_templates[config_init[num_plaq_main+num_middle:]]

    t_col1 = tensor_arr[(2*n-1)*(3*n):(2*n-1)*(3*n)+2*n]
    t_2 = tensor_arr[(2*n-1)*(3*n)+2*n:(2*n-1)*(3*n)+2*n+n*(2*n-1)].reshape((n, 2*n-1, 2, 2, 2, 2))
    t_col2 = jnp.flip(tensor_arr[(2*n-1)*(3*n)+2*n+n*(2*n-1):(2*n-1)*(3*n)+2*n+n*(2*n-1)+2*n], axis = 0)

    # shrinking column
    t_end = jnp.einsum("abcd, d -> abc", t_col1[-1], boundary_T_tensor(p))
    t_arr = tuple(t_middle[:n-1]) + (t_corner[0],) + tuple(t_col1[:-1]) + (t_end, )
    tensor, log_scale_step = process_col(tensor, t_arr, str_shrink)
    log_scale_total += log_scale_step

    #first column in AB
    t_end = jnp.einsum("abcd, d -> abc", t_2[0][-1], boundary_T_tensor(p))
    t_arr = (t_corner[1], ) + tuple(t_2[0][:-1]) + (t_end, )
    tensor, log_scale_step = process_col(tensor, t_arr, str_AB_col_ini)
    log_scale_total += log_scale_step

    # n - 2 column in AB
    for i in range(1, n-1):
        t_end = jnp.einsum("abcd, d -> abc", t_2[i][-1], boundary_T_tensor(p))
        t_arr = (t_middle[n-2+i], ) + tuple(t_2[i][:-1]) + (t_end, )
        tensor, log_scale_step = process_col(tensor, t_arr, str_AB_col)
        log_scale_total += log_scale_step

    # last column in AB
    t_end = jnp.einsum("abcd, d -> abc", t_2[-1][-1], boundary_T_tensor(p))
    t_arr = (t_corner[2], ) + tuple(t_2[-1][:-1]) + (t_end, )
    tensor, log_scale_step = process_col(tensor, t_arr, str_AB_col_end)
    log_scale_total += log_scale_step


    # expanding column (reversing order)
    t_ini = jnp.einsum("abcd, d -> abc", t_col2[0], boundary_T_tensor(p))
    t_arr = (t_ini, ) + tuple(t_col2[1:]) + (t_corner[3], ) + tuple(t_middle[2*n-3:])
    tensor, log_scale_step = process_col(tensor, t_arr, str_expand)
    log_scale_total += log_scale_step

    Pr = jnp.sum(tensor * tensor_end)
    log_Pr = jnp.log(Pr) + log_scale_total + log_scale_end
    return log_Pr


@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def log_Pr_B_func(n, p, plaq_active, str_B_col_ini, str_B_col,  str_B_col_end, tensor, log_scale_total, tensor_end, log_scale_end):
    num_plaq = 2 * (2 * n) * (n - 1) + (n + 2) * (n - 1) + 2
    T0 = full_tensor(0, p)
    T1 = full_tensor(1, p)
    templates = jnp.stack([T0, T1], axis=0)

    plaq_active = plaq_active[n:4*n, :2*n]
    plaq_index = jnp.arange(num_plaq)
    plaq_coord = batch_index_to_coord_B(plaq_index, n)
    config_init = plaq_active[plaq_coord[0], plaq_coord[1]]
    tensor_arr = templates[config_init]

    t_B_ini = tensor_arr[(n-2)*(2*n)+(2*n):(n-2)*(2*n)+(2*n)+n].reshape((n, 2, 2, 2, 2))
    t_2 = tensor_arr[(n-2)*(2*n)+(2*n)+n:(n-2)*(2*n)+(2*n)+n+n*(n-1)].reshape((n, n-1, 2, 2, 2, 2))
    t_B_end = jnp.flip(tensor_arr[(n-2)*(2*n)+(2*n)+n+n*(n-1):(n-2)*(2*n)+(2*n)+n+n*(n-1)+n].reshape((n, 2, 2, 2, 2)), axis = 0)

    revision_vec = n_tensor_product_shape_2(n, p)
    tensor = jnp.tensordot(tensor, revision_vec, axes=(list(range(n)), list(range(n))))
    tensor_end = jnp.tensordot(tensor_end, revision_vec, axes=(list(range(n)), list(range(n))))

    # t_B_ini
    t_ini = jnp.einsum("abcd, c, d -> ab", t_B_ini[0], boundary_T_tensor(p), boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, d -> abc", t_B_ini[-1], boundary_T_tensor(p))
    t_arr = (t_ini,) + tuple(t_B_ini[1:n-1]) + (t_end,)
    tensor, log_scale_step = process_col(tensor, t_arr, str_B_col_ini)
    log_scale_total += log_scale_step

    #t_2
    for i in range(n):
        t_ini = jnp.einsum("abcd, d -> abc", t_2[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_2[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_2[i][1:-1]) + (t_end, )
        tensor, log_scale_step = process_col(tensor, t_arr, str_B_col)
        log_scale_total += log_scale_step

    #t_2_end (reverse order)
    t_ini = jnp.einsum("abcd, d -> abc", t_B_end[0], boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, c, d -> ab", t_B_end[-1], boundary_T_tensor(p), boundary_T_tensor(p))
    t_arr = (t_ini, ) + tuple(t_B_end[1:n-1]) + (t_end,)
    tensor, log_scale_step = process_col(tensor, t_arr, str_B_col_end)
    log_scale_total += log_scale_step

    Pr = jnp.sum(tensor * tensor_end)
    log_Pr = jnp.log(Pr) + log_scale_total + log_scale_end

    return log_Pr

@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def log_Pr_AB_func_P0(n, p, plaq_active, str_B_col_ini, str_B_col,  str_B_col_end, tensor, log_scale_total, tensor_end, log_scale_end):
    y = 3*n ; x_left = 2*n - 2
    num_plaq = 2 * y * (x_left + 1) + (n + 2) * (2 * n - 1) + 2
    T0 = full_tensor(0, p)  # shape (2,2,2,2)
    T1 = full_tensor(1, p)  # shape (2,2,2,2)
    templates = jnp.stack([T0, T1], axis=0)

    plaq_index = jnp.arange(num_plaq)
    plaq_coord = batch_index_to_coord_AB_P0(plaq_index, n)
    config_init = plaq_active[plaq_coord[0], plaq_coord[1]]
    tensor_arr = templates[config_init]

    t_B_ini = tensor_arr[x_left*y + y : x_left*y + y + 2*n].reshape((2*n, 2, 2, 2, 2))
    t_2 = tensor_arr[x_left*y + y + 2*n : x_left * y + y + 2*n + n*(2*n-1)].reshape((n, 2*n-1, 2, 2, 2, 2))
    t_B_end = jnp.flip(tensor_arr[x_left*y+y+2*n+n*(2*n-1) : x_left*y+y+2*n+n*(2*n-1)+2*n].reshape((2*n, 2, 2, 2, 2)), axis = 0)

    revision_vec = n_tensor_product_shape_2(n, p)
    tensor = jnp.tensordot(tensor, revision_vec, axes=(list(range(n)), list(range(n))))
    tensor_end = jnp.tensordot(tensor_end, revision_vec, axes=(list(range(n)), list(range(n))))

    # t_B_ini
    t_ini = jnp.einsum("abcd, c, d -> ab", t_B_ini[0], boundary_T_tensor(p), boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, d -> abc", t_B_ini[-1], boundary_T_tensor(p))
    t_arr = (t_ini,) + tuple(t_B_ini[1:2*n-1]) + (t_end,)
    tensor, log_scale_step = process_col(tensor, t_arr, str_B_col_ini)
    log_scale_total += log_scale_step

    #t_2
    for i in range(n):
        t_ini = jnp.einsum("abcd, d -> abc", t_2[i][0], boundary_T_tensor(p))
        t_end = jnp.einsum("abcd, d -> abc", t_2[i][-1], boundary_T_tensor(p))
        t_arr = (t_ini, ) + tuple(t_2[i][1:-1]) + (t_end, )
        tensor, log_scale_step = process_col(tensor, t_arr, str_B_col)
        log_scale_total += log_scale_step

    #t_2_end (reverse order)
    t_ini = jnp.einsum("abcd, d -> abc", t_B_end[0], boundary_T_tensor(p))
    t_end = jnp.einsum("abcd, c, d -> ab", t_B_end[-1], boundary_T_tensor(p), boundary_T_tensor(p))
    t_arr = (t_ini, ) + tuple(t_B_end[1:2*n-1]) + (t_end,)
    tensor, log_scale_step = process_col(tensor, t_arr, str_B_col_end)
    log_scale_total += log_scale_step

    Pr = jnp.sum(tensor * tensor_end)
    log_Pr = jnp.log(Pr) + log_scale_total + log_scale_end
    return log_Pr

batch_log_Pr_bulk_pre = vmap(vmap(log_Pr_bulk_pre_func, (None, None, 0, None, None, None)), (None, 0, 0, None, None, None))
batch_log_Pr_bulk = vmap(vmap(log_Pr_bulk_func, (None, None, 0, None, None, 0, 0, 0, 0)), (None, 0, 0, None, None, 0, 0, 0, 0))
batch_log_Pr_AB = vmap(vmap(log_Pr_AB_func, (None, None, 0, None, None, None, None, None, 0, 0, 0, 0)), (None, 0, 0,  None, None, None, None, None, 0, 0, 0, 0))
batch_log_Pr_B = vmap(vmap(log_Pr_B_func, (None, None, 0, None,  None, None, 0, 0, 0, 0)), (None, 0, 0, None, None, None, 0, 0, 0, 0))
batch_log_Pr_AB_P0 = vmap(vmap(log_Pr_AB_func_P0, (None, None, 0, None,  None, None, 0, 0, 0, 0)), (None, 0, 0, None, None, None, 0, 0, 0, 0))
