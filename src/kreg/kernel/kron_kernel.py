from functools import reduce
from typing import Callable

import jax
import jax.numpy as jnp
from pykronecker import KroneckerDiag, KroneckerProduct
from pykronecker.base import KroneckerOperator

from kreg.utils import outer_fold


class KroneckerKernel:
    """Kronecker product of all kernel functions to form a complete kernel
    linear mapping.

    Parameters
    ----------
    kernels
        List of kernel functions.
    value_grids
        List of value grids, unique values for each dimension.
    nugget
        Regularization for the kernel matrix.

    """

    def __init__(
        self,
        kernels: list[Callable],
        value_grids: list[jax.Array],
        nugget: float = 5e-8,
    ) -> None:
        """
        TODO: Abstract this to lists of kernels and grids, kronecker out sex, age and time
        """
        self.kmats = [k(x, x) for k, x in zip(kernels, value_grids)]
        if len(self.kmats) == 1:
            nugget_vals = jnp.ones([len(m) for m in self.kmats]).reshape(-1, 1)
        else:
            nugget_vals = jnp.ones([len(m) for m in self.kmats])

        self.K = KroneckerProduct(self.kmats) + nugget * KroneckerDiag(
            nugget_vals
        )
        eigvals, eigvecs = zip(*[jnp.linalg.eigh(Ki) for Ki in self.kmats])
        self.left = KroneckerProduct(eigvecs)
        self.right = self.left.T
        self.kronvals = reduce(outer_fold, eigvals) + nugget
        self.P = self.left @ (KroneckerDiag(1 / self.kronvals)) @ self.right

        self.left_etimes_left = KroneckerProduct([e * e for e in eigvecs])
        self.shapes = [len(grid) for grid in value_grids]

        self.rootK = (
            self.left @ (KroneckerDiag(jnp.sqrt(self.kronvals))) @ self.right
        )
        self.rootP = (
            self.left
            @ (KroneckerDiag(1 / jnp.sqrt(self.kronvals)))
            @ self.right
        )

    def get_preconditioners(
        self, lam: float, beta: float
    ) -> tuple[KroneckerOperator, KroneckerOperator]:
        PC = (
            self.left
            @ KroneckerDiag(
                jnp.sqrt(self.kronvals / (lam + beta * self.kronvals))
            )
            @ self.right
        )
        PC_inv = (
            self.left
            @ KroneckerDiag(
                1 / jnp.sqrt(self.kronvals / (lam + beta * self.kronvals))
            )
            @ self.right
        )
        return PC, PC_inv

    def get_M(self, lam: float, beta: float) -> KroneckerOperator:
        middle = self.kronvals / (lam + beta * self.kronvals)
        return self.left @ KroneckerDiag(middle) @ self.right
