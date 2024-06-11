from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def colnorm(M: Array) -> Array:
    return M / jnp.linalg.norm(M, axis=0, keepdims=True)


def diag_prod(A: Array, B: Array) -> float:
    return jnp.sum(A * B, axis=0)


def xdiag(
    A_apply: Callable, n: int, m: int, key=jax.random.PRNGKey(10)
) -> tuple[Array, list[Array]]:
    """x diag estimator from
    "XTrace: Making the Most of Every Sample in Stochastic Trace Estimation"
    by Epperly, Tropp, Webber

    Parameters
    ----------
    A_apply
        Linear operator, assumed symmetric
    m
        Number of test vectors
    n
        Number of columns of A
    key
        Random key, by default jax.random.PRNGKey(10)

    Returns
    -------
    Array, List[Array]
        xdiag estimate, list of intermediate values

    """
    m = int(jnp.floor(m / 2))
    rad = jax.random.rademacher(key, (n, m), "float64")
    Y = A_apply(rad)
    Q, R = jnp.linalg.qr(Y)
    Z = A_apply(Q)
    T = Z.T @ rad
    S = colnorm(jnp.linalg.inv(R).T)

    d_QZ = diag_prod(Q.T, Z.T)
    d_QSSZ = diag_prod((Q @ S).T, (Z @ S).T)
    d_radQT = diag_prod(rad.T, (Q @ T).T)
    d_radY = diag_prod(rad.T, Y.T)
    d_rad_QSST = diag_prod(rad.T, (Q @ S @ jnp.diag(diag_prod(S, T))).T)
    d = d_QZ + (-d_QSSZ + d_radY - d_radQT + d_rad_QSST) / m
    return d, [d_QZ, d_QSSZ, d_radY, d_radQT, d_rad_QSST]
