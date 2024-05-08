import jax
import jax.numpy as jnp

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.typing import Callable, JAXArray


def randomized_nystroem(
    vmapped_A: Callable, input_shape: int, rank: int, key: int
) -> tuple[JAXArray, JAXArray]:
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


def build_nystroem_precon(
    hess_diag: JAXArray,
    kernel: KroneckerKernel,
    lam: float,
    key: int,
    rank: int = 50,
) -> Callable:
    """Builds a Nystroem preconditioner for Hessian operator form the
    likelihood, `hess_diag + lam * kernel.inv`.

    Parameters
    ----------
    hess_diag
        Diagonal of the Hessian of the likelihood.
    kernel
        Kronecker kernel for the prior from the likelihood.
    lam
        Regularization parameter that multiplies the kernel.
    key
        Random key.
    rank
        Rank of the Nystroem approximation.

    Returns
    -------
    Callable
        Function that applies the Nystroem preconditioner.

    """
    op_sqrt_k = kernel.op_root_k

    op_sqrt_kdk = jax.vmap(
        lambda x: op_sqrt_k @ (hess_diag * (op_sqrt_k @ x)),
        in_axes=1,
        out_axes=1,
    )
    U, E = randomized_nystroem(op_sqrt_kdk, len(hess_diag), rank, key)

    def precon(x):
        inner_diag = 1 / (E + lam) - (1 / lam)
        return (1 / lam) * x + U @ (inner_diag * (U.T @ x))

    def full_precon(x: JAXArray) -> JAXArray:
        return op_sqrt_k @ (precon(op_sqrt_k @ x))

    return full_precon
