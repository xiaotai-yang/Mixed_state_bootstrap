import jax
import jax.numpy as jnp
import string
from util import *
from functools import partial
# ——————————————————————————————————————————
# 1) initial‑row (“top”) contractions for BC
# ——————————————————————————————————————————

def einsum_strs_initial(n):
    """Tuple of n‑3 einsum subscripts to contract the top row (left→right)."""
    lbl = string.ascii_letters
    subs = []
    term = f"{lbl[0]}{lbl[n]}"
    for i in range(1, n - 1):
        term2 = f"{lbl[n]}{lbl[i]}{lbl[n+1]}"
        out   = f"{lbl[:i+1]}{lbl[n+1]}"
        subs.append(f"{term},{term2}->{out}")
        term = f"{lbl[:i+1]}{lbl[n]}"
    # final boundary merge
    term2 = f"{lbl[n]}{lbl[n-1]}"
    out   = f"{lbl[:n]}"
    subs.append(f"{term},{term2}->{out}")
    return tuple(subs)

def process_initial(p, t_arr, einsubs):
    # 1) build a single sequence of all the tensors we want to fold in:
    #    (first the n-3 middles, then the final boundary tensor)
    t = jnp.einsum("abcd, ce, d -> abe", t_arr[0], inverse_boundary_T(p), boundary_T_tensor(p))[:, :, 0]
    t_end = jnp.einsum("abcd, c, d -> ab", t_arr[-1], boundary_T_tensor(p), boundary_T_tensor(p))
    t_mid_list = jnp.einsum("abcde, e -> abcd", t_arr[1:-1], boundary_T_tensor(p))
    seq = tuple(t_mid_list) + (t_end,)

    # 2) run a single for‑loop, zipping subscripts ↔ tensors
    log_scale = 0
    for s, m in zip(einsubs, seq):
        t = jnp.einsum(s, t, m)
        t, log_scale_step = rescale_tensor(t)
        log_scale += log_scale_step
    return t, log_scale


# ——————————————————————————————————————————
# 2) single‑column (“bulk”) contractions for BC
# ——————————————————————————————————————————

def einsum_strs_column(n):
    """Tuple of n‑2 einsum subscripts to merge a column into a running tensor."""
    lbl = string.ascii_letters
    subs = []
    # first merge (left boundary)
    term1 = lbl[:n]
    term2 = f"{lbl[0]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[n]}{lbl[n+1]}{lbl[1:n]}"
    subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[0]}{lbl[n]}{lbl[1:n]}"
    # middle plaquettes
    for j in range(1, n-1):
        term2 = f"{lbl[j]}{lbl[n]}{lbl[n+1]}{lbl[n+2]}"
        out   = f"{lbl[:j]}{lbl[n+1]}{lbl[n+2]}{lbl[j+1:n]}"
        subs.append(f"{term1},{term2}->{out}")
        term1 = f"{lbl[:j+1]}{lbl[n]}{lbl[j+1:n]}"
    # final boundary merge
    term2 = f"{lbl[n-1]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[:n-1]}{lbl[n+1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)


def einsum_strs_shrink(n):
    lbl = string.ascii_letters
    subs = []
    term1 = lbl[:3*n]
    term2 = f"{lbl[3*n]}{lbl[0]}{lbl[-1]}"
    out   = f"{lbl[3*n]}{lbl[1:3*n]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(n-2):
        term1 = f"{lbl[:3*n-i]}{lbl[-1]}"
        term2 = f"{lbl[0]}{lbl[1]}{lbl[3*n]}"
        out   = f"{lbl[3*n]}{lbl[2:3*n-i]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[:3*n-1-(n-3)]}{lbl[-1]}"
    term2 = f"{lbl[0]}{lbl[1]}{lbl[3*n]}{lbl[-2]}"
    out   = f"{lbl[3*n]}{lbl[2:3*n-1-(n-3)]}{lbl[-2]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(1, 2*n):
        term1 = f"{lbl[:i-1]}{lbl[3*n]}{lbl[i-1:2*n]}{lbl[-2]}{lbl[-1]}"
        term2 = f"{lbl[3*n]}{lbl[i-1]}{lbl[3*n+1]}{lbl[3*n+2]}"
        out   = f"{lbl[:i-1]}{lbl[3*n+1]}{lbl[3*n+2]}{lbl[i:2*n]}{lbl[-2]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[:2*n-1]}{lbl[3*n]}{lbl[2*n-1]}{lbl[-2]}{lbl[-1]}"
    term2 = f"{lbl[3*n]}{lbl[2*n-1]}{lbl[3*n+1]}"
    out = f"{lbl[:2*n-1]}{lbl[3*n+1]}{lbl[-2]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)

def einsum_strs_AB_column_ini(n):
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[:n]}{lbl[-2]}{lbl[-1]}"
    term2 = f"{lbl[-2]}{lbl[0]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[n+1]}{lbl[n]}{lbl[1:n]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[0]}{lbl[n]}{lbl[1:n]}{lbl[-1]}"
    for j in range(1, n-1):
        term2 = f"{lbl[j]}{lbl[n]}{lbl[n+1]}{lbl[n+2]}"
        out   = f"{lbl[:j]}{lbl[n+1]}{lbl[n+2]}{lbl[j+1:n]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
        term1 = f"{lbl[:j+1]}{lbl[n]}{lbl[j+1:n]}{lbl[-1]}"
    # final boundary merge
    term2 = f"{lbl[n-1]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[:n-1]}{lbl[n+1]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)


def einsum_strs_AB_column(n):
    """Tuple of n‑2 einsum subscripts to merge a column into a running tensor."""
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[:n]}{lbl[-1]}"
    term2 = f"{lbl[0]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[n+1]}{lbl[n]}{lbl[1:n]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[0]}{lbl[n]}{lbl[1:n]}{lbl[-1]}"
    # middle plaquettes
    for j in range(1, n-1):
        term2 = f"{lbl[j]}{lbl[n]}{lbl[n+1]}{lbl[n+2]}"
        out   = f"{lbl[:j]}{lbl[n+1]}{lbl[n+2]}{lbl[j+1:n]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
        term1 = f"{lbl[:j+1]}{lbl[n]}{lbl[j+1:n]}{lbl[-1]}"
    # final boundary merge
    term2 = f"{lbl[n-1]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[:n-1]}{lbl[n+1]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)

def einsum_strs_AB_column_end(n):
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[:n]}{lbl[-1]}"
    term2 = f"{lbl[0]}{lbl[n]}{lbl[n+1]}{lbl[-2]}"
    out   = f"{lbl[n+1]}{lbl[n]}{lbl[1:n]}{lbl[-2]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[0]}{lbl[n]}{lbl[1:n]}{lbl[-2]}{lbl[-1]}"
    for j in range(1, n-1):
        term2 = f"{lbl[j]}{lbl[n]}{lbl[n+1]}{lbl[n+2]}"
        out   = f"{lbl[:j]}{lbl[n+1]}{lbl[n+2]}{lbl[j+1:n]}{lbl[-2]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
        term1 = f"{lbl[:j+1]}{lbl[n]}{lbl[j+1:n]}{lbl[-2]}{lbl[-1]}"
    # final boundary merge
    term2 = f"{lbl[n-1]}{lbl[n]}{lbl[n+1]}"
    out   = f"{lbl[:n-1]}{lbl[n+1]}{lbl[-2]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)


# Note that the contraction direction is reversed here
def einsum_strs_expand(n):
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[n:3*n]}{lbl[-2]}{lbl[-1]}"
    term2 = f"{lbl[3*n-1]}{lbl[3*n]}{lbl[3*n+1]}"
    out   = f"{lbl[n:3*n-1]}{lbl[3*n]}{lbl[3*n+1]}{lbl[-2]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(1, 2*n):
        term1 = f"{lbl[n:3*n-i]}{lbl[3*n]}{lbl[3*n-i:3*n]}{lbl[-2]}{lbl[-1]}"
        term2 = f"{lbl[3*n-i-1]}{lbl[3*n]}{lbl[3*n+1]}{lbl[3*n+2]}"
        out   = f"{lbl[n:3*n-i-1]}{lbl[3*n+1]}{lbl[3*n+2]}{lbl[3*n-i:3*n]}{lbl[-2]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[3*n]}{lbl[n:3*n]}{lbl[-2]}{lbl[-1]}"
    term2 = f"{lbl[-2]}{lbl[3*n]}{lbl[n-1]}{lbl[3*n+1]}"
    out = f"{lbl[3*n+1]}{lbl[n-1]}{lbl[n:3*n]}{lbl[-1]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(1, n-1):
        term1 = f"{lbl[3*n]}{lbl[n-i:n]}{lbl[n:3*n]}{lbl[-1]}"
        term2 = f"{lbl[3*n]}{lbl[n-i-1]}{lbl[3*n+1]}"
        out   = f"{lbl[3*n+1]}{lbl[n-i-1]}{lbl[n-i:n]}{lbl[n:3*n]}{lbl[-1]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[3*n]}{lbl[1:n]}{lbl[n:3*n]}{lbl[-1]}"
    term2 = f"{lbl[3*n]}{lbl[0]}{lbl[-1]}"
    out   = f"{lbl[0]}{lbl[1:n]}{lbl[n:3*n]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)


def einsum_B_col_ini(n):
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[:n]}"
    term2 = f"{lbl[0]}{lbl[n]}"
    out = f"{lbl[n]}{lbl[1:n]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(n-2):
        term1 = f"{lbl[:i]}{lbl[n]}{lbl[i+1:n]}"
        term2 = f"{lbl[n]}{lbl[i+1]}{lbl[n+1]}{lbl[n+2]}"
        out = f"{lbl[:i]}{lbl[n+1]}{lbl[n+2]}{lbl[i+2:n]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[:n-2]}{lbl[n]}{lbl[n-1]}"
    term2 = f"{lbl[n]}{lbl[n-1]}{lbl[n+1]}"
    out = f"{lbl[:n-2]}{lbl[n+1]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)



# reverse order
def einsum_B_col_end(n):
    lbl = string.ascii_letters
    subs = []
    term1 = f"{lbl[:n-1]}"
    term2 = f"{lbl[n-2]}{lbl[n-1]}{lbl[n]}"
    out = f"{lbl[:n-2]}{lbl[n-1]}{lbl[n]}"
    subs.append(f"{term1},{term2}->{out}")
    for i in range(n-2):
        term1 = f"{lbl[:n-2-i]}{lbl[n]}{lbl[n-1-i:n]}"
        term2 = f"{lbl[n-2-i-1]}{lbl[n]}{lbl[n+1]}{lbl[n+2]}"
        out = f"{lbl[:n-2-i-1]}{lbl[n+1]}{lbl[n+2]}{lbl[n-1-i:n]}"
        subs.append(f"{term1},{term2}->{out}")
    term1 = f"{lbl[n]}{lbl[1:n]}"
    term2 = f"{lbl[n]}{lbl[0]}"
    out = f"{lbl[:n]}"
    subs.append(f"{term1},{term2}->{out}")
    return tuple(subs)


def n_tensor_product_shape_2(n, p):
    vec = jnp.sqrt(jnp.array([1 - p, p]))
    result = vec
    for _ in range(n - 1):
        result = jnp.tensordot(result, vec, axes=0)
    return result

@partial(jax.jit, static_argnums=(2,))
def process_col(tensor, t_arr, einsum_strs):
    """
    Take an existing 'tensor' and fold in one column of BC‑tensors,
    using the precomputed subscripts.
    """
    log_scale = 0.0
    for sub, tpl in zip(einsum_strs, t_arr):
        tensor = jnp.einsum(sub, tensor, tpl)
        tensor, log_scale_step = rescale_tensor(tensor)
        log_scale += log_scale_step
    return tensor, log_scale