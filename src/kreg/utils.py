from typing import Callable

import jax
import jax.numpy as jnp


def cartesian_prod(x: jax.Array, y: jax.Array):
    """
    Computes Cartesian product of two arrays x,y
    """
    a, b = jnp.meshgrid(y, x)
    full_X = jnp.vstack([b.flatten(), a.flatten()]).T
    return full_X


def randomized_nystroem(
    vmapped_A: Callable, input_shape: int, rank: int, key: int
) -> tuple[jax.Array, jax.Array]:
    """Create a randomized Nystrom approximation of a matrix.

    Parameters
    ----------
    vmapped_A
        Function that computes the matrix-vector product of the matrix to be approximated.
    input_shape
        Number of rows of the matrix.
    rank
        Rank of the approximation.
    key
        Random key.

    Returns
    -------
    tuple[Array, Array]
        Approximation matrices U and E.

    """
    X = jax.random.normal(key, (input_shape, rank))
    Q = jnp.linalg.qr(X).Q
    AQ = vmapped_A(Q)
    eps = 1e-8 * jnp.linalg.norm(AQ, ord="fro")
    C = jnp.linalg.cholesky(Q.T @ AQ + eps * jnp.identity(rank))
    B = jax.scipy.linalg.solve_triangular(C, AQ.T, lower=True).T
    U, S = jax.scipy.linalg.svd(B, full_matrices=False)[:2]
    E = jnp.maximum(0, S**2 - eps * jnp.ones(rank))
    return U, E


def build_ny_precon(U, E, lam):
    def precon(x):
        inner_diag = 1 / (E + lam) - (1 / lam)
        return (1 / lam) * x + U @ (inner_diag * (U.T @ x))

    return precon
