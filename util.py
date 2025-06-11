import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import lax
from functools import partial
from jax.random import split, bernoulli
from jax import vmap

def boundary_T_tensor(p):
    return jnp.sqrt(jnp.array([1 - p, p]))

def inverse_boundary_T(p):
    return jnp.where(p == 0., jnp.array([[1., 0.], [0. ,1.]]), jnp.sqrt(jnp.array([[1/(1 - p), 0], [0., 1/p]])))

def T_tensor(p):
    """
    Constructs the two-leg tensor that represents the weight on each edge.

    T_{s1 s2} = delta(s1,s2) * (p**s1)*((1-p)**(1-s1))
    It is a 2x2 diagonal matrix.
    """
    return jnp.sqrt(jnp.array([[1 - p, 0], [0, p]]))

def Q_tensor(m):
    """
    Constructs the four-leg tensor for a plaquette that enforces the parity.

    Q^m_{s1 s2 s3 s4} = 1 if (s1 + s2 + s3 + s4) mod 2 equals m, else 0.
    The tensor has shape (2,2,2,2).

    m should be 0 or 1.
    """
    Q = jnp.zeros((2, 2, 2, 2))
    # Loop over all 16 combinations of {s1, s2, s3, s4}.
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                for s4 in (0, 1):
                    #print(s1, s2, s3, s4)
                    parity = (s1 + s2 + s3 + s4) % 2
                    # If parity matches the anyon measurement, set entry to 1.
                    value = lax.select(parity == m, 1., 0.)
                    Q = Q.at[s1, s2, s3, s4].set(value)
    return Q


def incomplete_Q_tensor(m):
    Q = jnp.zeros((2, 2, 2))
    # Loop over all 16 combinations of {s1, s2, s3, s4}.
    for s1 in (0, 1):
        for s2 in (0, 1):
            for s3 in (0, 1):
                parity = (s1 + s2 + s3) % 2
                # If parity matches the anyon measurement, set entry to 1.
                value = lax.select(parity == m, 1., 0.)
                Q = Q.at[s1, s2, s3].set(value)
    return Q

def full_tensor(m, p):
    tensor_list = []
    Q = Q_tensor(m)
    T = T_tensor(p)
    tensor_list.append(Q)
    for i in range(4):
        tensor_list.append(T)
    einsum_str = 'ijkl, ia, jb, kc, ld->abcd'

    return jnp.einsum(einsum_str, *tensor_list)

def inner_edge_tensor(m, p):
    Q = incomplete_Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'ijk, ja->iak'

    return jnp.einsum(einsum_str, Q, T)

def inner_edge_corner_tensor(m, p):
    Q = Q_tensor(m)
    T = T_tensor(p)
    einsum_str = 'hijk, ia, jb->habk'
    return jnp.einsum(einsum_str, Q, T, T)


@jax.jit
def rescale_tensor(t):
    """Rescales tensor by max absolute value and returns log scaling factor."""
    max_abs_val = jnp.max(jnp.abs(t))
    # Prevent division by zero and log(0)
    safe_max_abs_val = jnp.maximum(max_abs_val, 1e-30)
    t_rescaled = t / safe_max_abs_val
    log_scale = jnp.log(safe_max_abs_val)
    return t_rescaled, log_scale

@partial(jax.jit, static_argnums=(2,))
def make_config(key, p, n):
    full_x, full_y = 5*n, 3*n
    key, key_h, key_v, _ = split(key, 4)
    h_active = bernoulli(key_h, p, (full_x+1, full_y)).astype(jnp.int64)
    v_active = bernoulli(key_v, p, (full_x,   full_y)).astype(jnp.int64)
    v_active = jnp.concatenate([jnp.zeros((full_x, 1), jnp.int64), v_active], axis=1)
    config_init = (h_active[:-1] + h_active[1:] + v_active[:,:-1] + v_active[:,1:]) % 2
    return config_init

batch_make_config = vmap(vmap(make_config, (0, None, None)), (0, 0, None))
