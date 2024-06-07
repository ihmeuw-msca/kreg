from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from kreg.kernel import KroneckerKernel
from kreg.typing import Callable, DataFrame, JAXArray


class Likelihood(ABC):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        self.obs = obs
        self.weights = weights
        self.offset = offset
        self.data: dict[str, JAXArray] = {}

    @property
    def size(self) -> int | None:
        if not self.data:
            raise ValueError("No data attached.")
        return len(self.data["obs"])

    def attach(self, data: DataFrame, kernel: KroneckerKernel) -> None:
        if not (data[self.weights] >= 0).all():
            raise ValueError("Weights must be non-negative.")
        self.data["obs"] = jnp.asarray(data[self.obs])
        self.data["weights"] = jnp.asarray(data[self.weights])
        self.data["offset"] = jnp.asarray(data[self.offset])
        self.data["mat"] = self.encode(data, kernel)

    def detach(self) -> None:
        self.data.clear()

    @staticmethod
    def encode(data: DataFrame, kernel: KroneckerKernel) -> JAXArray:
        nrow, ncol = len(data), len(kernel.span)
        val = jnp.ones(nrow)
        row = jnp.arange(nrow)
        col = jnp.asarray(
            data[kernel.names]
            .merge(kernel.span.reset_index(), how="left", on=kernel.names)
            .eval("index")
        )

        indices = jnp.vstack([row, col]).T
        return BCOO((val, indices), shape=(nrow, ncol))

    @abstractmethod
    def objective(self, x: JAXArray) -> JAXArray:
        """Negative log likelihood."""

    @abstractmethod
    def gradient(self, x: JAXArray) -> JAXArray:
        """Gradient of the negative log likelihood."""

    @abstractmethod
    def hessian(self, x: JAXArray) -> JAXArray:
        """Hessian of the negative log likelihood."""


class BinomialLikelihood(Likelihood):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame, kernel: KroneckerKernel) -> None:
        if not ((data[self.obs] >= 0).all() and (data[self.obs] <= 1).all()):
            raise ValueError("Observations must be in [0, 1].")
        return super().attach(data, kernel)

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"].dot(
            jnp.log(1 + jnp.exp(-y)) + (1 - self.data["obs"]) * y
        )

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        z = jnp.exp(x + self.data["offset"])
        return self.data["weights"] * (z / (1 + z) - self.data["obs"])

    def hessian(self, x: JAXArray) -> Callable:
        z = jnp.exp(x + self.data["offset"])
        scale = self.data["weights"] * (z / ((1 + z) ** 2))

        def op_hess(x: JAXArray) -> JAXArray:
            return self.data["mat"].T @ (scale * (self.data["mat"] @ x))

        return op_hess


class GaussianLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return 0.5 * self.data["weights"].dot((self.data["obs"] - y) ** 2)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (y - self.data["obs"])

    def hessian(self, x: JAXArray) -> JAXArray:
        scale = self.data["weights"]

        def op_hess(x: JAXArray) -> JAXArray:
            return self.data["mat"].T @ (scale * (self.data["mat"] @ x))

        return op_hess


class PoissonLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame, kernel: KroneckerKernel) -> None:
        if not (data[self.obs] >= 0).all():
            raise ValueError("Observations must be non-negative.")
        return super().attach(data, kernel)

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"].dot(jnp.exp(y) - self.data["obs"] * y)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (jnp.exp(y) - self.data["obs"])

    def hessian(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        scale = self.data["weights"] * jnp.exp(y)

        def op_hess(x: JAXArray) -> JAXArray:
            return self.data["mat"].T @ (scale * (self.data["mat"] @ x))

        return op_hess
