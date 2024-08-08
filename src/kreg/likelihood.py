from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp

from kreg.typing import DataFrame, JAXArray


class Likelihood(ABC):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        self.obs = obs
        self.weights = weights
        self.offset = offset
        self.data: dict[str, JAXArray] = {}
        self.trim_weights: JAXArray

    @property
    def size(self) -> int | None:
        if not self.data:
            return None
        return len(self.data["obs"])

    def attach(self, data: DataFrame) -> None:
        if not (data[self.weights] >= 0).all():
            raise ValueError("Weights must be non-negative.")
        self.data["obs"] = jnp.asarray(data[self.obs])
        self.data["weights"] = jnp.asarray(data[self.weights])
        self.data["orig_weights"] = jnp.asarray(data[self.weights])
        self.data["offset"] = jnp.asarray(data[self.offset])
        self.data["trim_weights"] = jnp.ones(len(data))

    def update_trim_weights(self, w: JAXArray) -> None:
        self.data["trim_weights"] = jnp.asarray(w)
        self.data["weights"] = (
            self.data["orig_weights"] * self.data["trim_weights"]
        )

    def detach(self) -> None:
        self.data.clear()

    @abstractmethod
    def nll_terms(self, x: JAXArray) -> JAXArray:
        """Terms of the negative log likelihood."""

    @abstractmethod
    def objective(self, x: JAXArray) -> JAXArray:
        """Negative log likelihood."""

    @abstractmethod
    def gradient(self, x: JAXArray) -> JAXArray:
        """Gradient of the negative log likelihood."""

    @abstractmethod
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        """Diagonal of the Hessian of the negative log likelihood."""


class BinomialLikelihood(Likelihood):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame) -> None:
        if not ((data[self.obs] >= 0).all() and (data[self.obs] <= 1).all()):
            raise ValueError("Observations must be in [0, 1].")
        return super().attach(data)

    @partial(jax.jit, static_argnums=0)
    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (
            jnp.log(1 + jnp.exp(-y)) + (1 - self.data["obs"]) * y
        )

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

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        z = jnp.exp(x + self.data["offset"])
        return self.data["weights"] * (z / ((1 + z) ** 2))


class GaussianLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    @partial(jax.jit, static_argnums=0)
    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return 0.5 * self.data["weights"] * (self.data["obs"] - y) ** 2

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return 0.5 * self.data["weights"].dot((self.data["obs"] - y) ** 2)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (y - self.data["obs"])

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        return self.data["weights"]


class PoissonLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame) -> None:
        if not (data[self.obs] >= 0).all():
            raise ValueError("Observations must be non-negative.")
        return super().attach(data)

    @partial(jax.jit, static_argnums=0)
    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (jnp.exp(y) - self.data["obs"] * y)

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"].dot(jnp.exp(y) - self.data["obs"] * y)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * (jnp.exp(y) - self.data["obs"])

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        y = x + self.data["offset"]
        return self.data["weights"] * jnp.exp(y)
