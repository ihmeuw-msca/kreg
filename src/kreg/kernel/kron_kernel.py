import jax.numpy as jnp
from pykronecker import KroneckerProduct

from kreg.typing import Callable, JAXArray


class KroneckerKernel:
    """Kronecker product of all kernel functions to form a complete kernel
    linear mapping.

    Parameters
    ----------
    kernels
        List of kernel functions.
    grids
        List of value grids, unique values for each dimension.
    nugget
        Regularization for the kernel matrix.

    """

    def __init__(
        self,
        kernels: list[Callable],
        grids: list[JAXArray],
        nugget: float = 5e-8,
    ) -> None:
        self.shape = tuple(map(len, grids))
        self.kmats = [
            k(x, x) + nugget * jnp.identity(len(x))
            for k, x in zip(kernels, grids)
        ]
        self.op_k = KroneckerProduct(self.kmats)

        self.eigdecomps = list(map(jnp.linalg.eigh, self.kmats))

        self.op_p = KroneckerProduct(
            [(mat / vec).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        self.op_root_k = KroneckerProduct(
            [(mat * jnp.sqrt(vec)).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        self.op_root_p = KroneckerProduct(
            [(mat / jnp.sqrt(vec)).dot(mat.T) for vec, mat in self.eigdecomps]
        )

    def dot(self, x: JAXArray) -> JAXArray:
        return self.op_k @ x

    def __matmul__(self, x: JAXArray) -> JAXArray:
        return self.dot(x)

    def __len__(self) -> int:
        return len(self.op_k)
