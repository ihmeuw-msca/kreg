import itertools
import math

import jax.numpy as jnp
import numpy as np

from kreg.kernel.component import KernelComponent
from kreg.kernel.dimension import Dimension
from kreg.kernel.kron_op import KroneckerOperator
from kreg.typing import DataFrame, JAXArray


class KroneckerKernel:
    """Kronecker product of the kernel matrices of several components.

    The kernel matrix is ``K = K_1 ⊗ K_2 ⊗ ... ⊗ K_m`` where ``K_i`` is the
    kernel matrix of ``kernel_components[i]``. The product is never formed:
    after ``attach`` the operators ``op_k`` (``K``), ``op_p`` (``K^{-1}``),
    ``op_root_k`` (``K^{1/2}``) and ``op_root_p`` (``K^{-1/2}``) are
    ``KroneckerOperator`` instances built from per-component
    eigendecompositions. Rows of ``K`` are ordered row-major over the
    component spans, first component slowest, which is also the column order
    of the encoding matrix produced by ``Term.encode``.

    Parameters
    ----------
    kernel_components
        Components, outermost first. Each supplies the dimensions it covers
        and a kernel function on them.
    nugget
        Value added to the diagonal of every component kernel matrix so that
        it is positive definite.
    normalize
        If True, divide every component kernel matrix by the square roots of
        its column maxima. For stationary kernels this puts ones on the
        diagonal.

    """

    def __init__(
        self,
        kernel_components: list[KernelComponent],
        nugget: float = 5e-8,
        normalize: bool = False,
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
        self.normalize = normalize

        self.kmats: list[JAXArray]
        self.op_k: KroneckerOperator
        self.eigdecomps: list[tuple[JAXArray, JAXArray]]
        self.op_p: KroneckerOperator
        self.op_root_k: KroneckerOperator
        self.op_root_p: KroneckerOperator
        self.status = "detached"

    @property
    def label(self) -> str:
        return "*".join([dim.label for dim in self.dimensions])

    @property
    def span(self) -> DataFrame:
        columns = list(
            itertools.chain.from_iterable(
                [dim.columns for dim in self.dimensions]
            )
        )
        # interval dimensions are labelled by their grid cells, not midpoints
        rows = [
            dim.span
            if dim.interval is None
            else np.column_stack([dim.grid[:-1], dim.grid[1:]])
            for dim in self.dimensions
        ]
        span = DataFrame(
            data=np.array(
                [np.hstack(list(row)) for row in itertools.product(*rows)]
            ),
            columns=columns,
        )
        return span

    def _build_matrices(self):
        self.kmats = [
            component.build_kmat(self.nugget, self.normalize)
            for component in self.kernel_components
        ]
        self.op_k = KroneckerOperator(self.kmats)
        self.eigdecomps = list(map(jnp.linalg.eigh, self.kmats))
        self.op_p = KroneckerOperator(
            [(mat / vec).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        self.op_root_k = KroneckerOperator(
            [(mat * jnp.sqrt(vec)).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        self.op_root_p = KroneckerOperator(
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
        return math.prod(len(dim.span) for dim in self.dimensions)
