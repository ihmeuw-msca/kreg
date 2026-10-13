import functools
import math
from typing import Self

import jax.numpy as jnp

from kreg.typing import JAXArray


class KroneckerOperator:
    """Linear operator for a Kronecker product of dense square matrices.

    ``KroneckerOperator([A, B, C])`` represents ``A ⊗ B ⊗ C`` without ever
    forming the product. Matrix-vector products use the identity
    ``(A ⊗ B) vec(X) = vec(A X Bᵀ)`` generalised to any number of factors:
    the vector is reshaped to a tensor with one axis per factor (first factor
    slowest, matching row-major / ``jnp.kron`` ordering) and each factor is
    contracted along its own axis.

    Parameters
    ----------
    mats
        Square factor matrices, outermost first.

    """

    def __init__(self, mats: list[JAXArray]) -> None:
        if not mats:
            raise ValueError("KroneckerOperator needs at least one factor.")
        for mat in mats:
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError("Kronecker factors must be square matrices.")
        self.mats = list(mats)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(mat.shape[0] for mat in self.mats)

    @property
    def shape(self) -> tuple[int, int]:
        size = math.prod(self.sizes)
        return size, size

    @property
    def T(self) -> Self:
        return type(self)([mat.T for mat in self.mats])

    def dot(self, x: JAXArray) -> JAXArray:
        """Apply the operator to a vector or to each column of a matrix."""
        x = jnp.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError("Expected a vector or a matrix.")
        if x.shape[0] != self.shape[1]:
            raise ValueError(
                f"Dimension mismatch: operator shape {self.shape}, "
                f"input shape {x.shape}."
            )
        batch = x.shape[1:]
        out = x.reshape(*self.sizes, *batch)
        for axis, mat in enumerate(self.mats):
            out = jnp.moveaxis(
                jnp.tensordot(mat, out, axes=([1], [axis])), 0, axis
            )
        return out.reshape(self.shape[0], *batch)

    def to_array(self) -> JAXArray:
        """Materialise the full Kronecker product (use only for small sizes)."""
        return functools.reduce(jnp.kron, self.mats)

    def __matmul__(self, x: JAXArray) -> JAXArray:
        return self.dot(x)

    def __rmatmul__(self, x: JAXArray) -> JAXArray:
        # x @ K == (Kᵀ @ xᵀ)ᵀ
        return self.T.dot(jnp.asarray(x).T).T

    def __mul__(self, scalar: float) -> Self:
        """Scale the operator by folding the scalar into the first factor."""
        return type(self)([scalar * self.mats[0], *self.mats[1:]])

    __rmul__ = __mul__

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"KroneckerOperator(sizes={self.sizes})"
