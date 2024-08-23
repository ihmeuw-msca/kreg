import math

import jax.numpy as jnp
from pykronecker import KroneckerProduct

from kreg.kernel.component import KernelComponent
from kreg.typing import DataFrame, JAXArray


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
        kernel_components: list[KernelComponent],
        nugget: float = 5e-8,
    ) -> None:
        self.kernel_components = kernel_components
        self.nugget = nugget

        self.names: list[str] = []
        for component in self.kernel_components:
            name = component.name
            if isinstance(name, str):
                self.names.append(name)
            else:
                self.names.extend(list(name))

        self.kmats: list[JAXArray]
        self.op_k: KroneckerProduct
        self.eigdecomps: list[tuple[JAXArray, JAXArray]]
        self.op_p: KroneckerProduct
        self.op_root_k: KroneckerProduct
        self.op_root_p: KroneckerProduct
        self.matrices_computed: bool = False
        self.data_attached: bool = False

    def build_matrices(self):
        self.kmats = [
            component.build_kmat(self.nugget)
            for component in self.kernel_components
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
        self.matrices_computed = True


    def attach(self, data: DataFrame) -> None:
        for component in self.kernel_components:
            component.attach(data)
        self.build_matrices()
        self.data_attached = True

    def clear_matrices(self) -> None:
        del self.kmats
        del self.op_k
        del self.eigdecomps
        del self.op_p
        del self.op_root_k
        del self.op_root_p
        self.matrices_computed = False

    def dot(self, x: JAXArray) -> JAXArray:
        return self.op_k @ x

    def __matmul__(self, x: JAXArray) -> JAXArray:
        return self.dot(x)

    def __len__(self) -> int:
        return math.prod(map(len, self.kernel_components))
