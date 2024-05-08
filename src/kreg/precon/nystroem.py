import jax
import jax.numpy as jnp

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon.base import PreconBuilder
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


class NystroemPreconBuilder(PreconBuilder):
    """Nystroem preconditioner builder for the Hessian operator.

    Parameters
    ----------
    likelihood
        Likelihood object.
    kernel
        Kronecker kernel for the prior from the likelihood.
    lam
        Regularization parameter that multiplies the kernel.
    max_iter
        Maximum number of iterations. This is used for generating a series of
        keys for randomized_nystroem function.
    rank
        Rank of the approximation. Default is 25.
    key
        Random key. Default is 101.

    """

    def __init__(
        self,
        likelihood: Likelihood,
        kernel: KroneckerKernel,
        lam: float,
        rank: int = 25,
        key: int = 101,
    ) -> None:
        self.likelihood = likelihood
        self.kernel = kernel
        self.lam = lam
        self.rank = rank
        self.key = jax.random.PRNGKey(key)
        self.keys: list[JAXArray] = []

    def __call__(self, x: JAXArray) -> Callable:
        hess_diag = self.likelihood.hessian_diag(x)
        op_sqrt_k = self.kernel.op_root_k

        op_sqrt_kdk = jax.vmap(
            lambda x: op_sqrt_k @ (hess_diag * (op_sqrt_k @ x)),
            in_axes=1,
            out_axes=1,
        )
        self.key += 1
        U, E = randomized_nystroem(
            op_sqrt_kdk, len(hess_diag), self.rank, self.key
        )
        self.keys.append(self.key)

        def precon(x):
            inner_diag = 1 / (E + self.lam) - (1 / self.lam)
            return (1 / self.lam) * x + U @ (inner_diag * (U.T @ x))

        def full_precon(x: JAXArray) -> JAXArray:
            return op_sqrt_k @ (precon(op_sqrt_k @ x))

        return full_precon
