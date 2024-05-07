import functools
from typing import Callable

import jax
import jax.numpy as jnp
from pykronecker import KroneckerDiag, KroneckerIdentity, KroneckerProduct

from kreg.utils import outer_fold


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
        grids: list[jax.Array],
        nugget: float = 5e-8,
    ) -> None:
        self.shape = tuple(map(len, grids))
        self.kmats = [k(x, x) for k, x in zip(kernels, grids)]
        self.op_k = KroneckerProduct(self.kmats) + nugget * KroneckerIdentity(
            tensor_shape=self.shape
        )

        evals, evecs = zip(*map(jnp.linalg.eigh, self.kmats))
        self.evecs = KroneckerProduct(evecs)
        self.evals = functools.reduce(outer_fold, evals) + nugget

        self.op_p = self.evecs @ (KroneckerDiag(1 / self.evals)) @ self.evecs.T
        self.op_root_k = (
            self.evecs @ (KroneckerDiag(jnp.sqrt(self.evals))) @ self.evecs.T
        )
        self.op_root_p = (
            self.evecs
            @ (KroneckerDiag(1 / jnp.sqrt(self.evals)))
            @ self.evecs.T
        )

    def dot(self, x: jax.Array) -> jax.Array:
        return self.op_k @ x

    def __matmul__(self, x: jax.Array) -> jax.Array:
        return self.dot(x)

    def __len__(self) -> int:
        return len(self.op_k)
