import itertools

import jax.numpy as jnp
import numpy as np
from pykronecker import KroneckerProduct

from kreg.kernel.component import KernelComponent
from kreg.kernel.dimension import Dimension
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

        self.dimensions: list[Dimension] = []
        self.columns: list[str] = []
        for component in self.kernel_components:
            dimensions = component.dimensions
            if isinstance(dimensions, Dimension):
                self.dimensions.append(dimensions)
            else:
                self.dimensions.extend(dimensions)
        for dimension in self.dimensions:
            columns = dimension.columns
            if isinstance(columns, str):
                self.columns.append(columns)
            else:
                self.columns.extend(columns)

        self.kmats: list[JAXArray]
        self.op_k: KroneckerProduct
        self.eigdecomps: list[tuple[JAXArray, JAXArray]]
        self.op_p: KroneckerProduct
        self.op_root_k: KroneckerProduct
        self.op_root_p: KroneckerProduct
        self.status = "detached"

    @property
    def span(self) -> DataFrame:
        span = DataFrame(
            data=np.asarray(
                list(itertools.product(*[dim.span for dim in self.dimensions])),
            ),
            columns=[dim.name for dim in self.dimensions],
        )
        return span

    def _build_matrices(self):
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

    def attach(self, data: DataFrame) -> None:
        if self.status == "detached":
            for component in self.kernel_components:
                component.set_span(data)
            self._build_matrices()
            self.status = "attached"

    def clear_matrices(self) -> None:
        if self.status == "attached":
            del self.kmats
            del self.op_k
            del self.eigdecomps
            del self.op_p
            del self.op_root_k
            del self.op_root_p
            self.status = "detached"

    def dot(self, x: JAXArray) -> JAXArray:
        return self.op_k @ x

    def __matmul__(self, x: JAXArray) -> JAXArray:
        return self.dot(x)

    def __len__(self) -> int:
        return len(self.span)
