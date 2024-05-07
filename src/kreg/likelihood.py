from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp


class Likelihood(ABC):
    def __init__(
        self, obs: jax.Array, weights: jax.Array, offset: jax.Array
    ) -> None:
        if not (weights >= 0).all():
            raise ValueError("Weights must be non-negative.")
        self.obs = obs
        self.weights = weights
        self.offset = offset
        self.size = len(self.obs)

    @abstractmethod
    def objective(self, x: jax.Array) -> jax.Array:
        """Negative log likelihood."""

    @abstractmethod
    def gradient(self, x: jax.Array) -> jax.Array:
        """Gradient of the negative log likelihood."""

    @abstractmethod
    def hessian_diag(self, x: jax.Array) -> jax.Array:
        """Diagonal of the Hessian of the negative log likelihood."""


class BinomialLikelihood(Likelihood):
    def __init__(
        self, obs: jax.Array, weights: jax.Array, offset: jax.Array
    ) -> None:
        if not ((obs >= 0).all() and (obs <= 1).all()):
            raise ValueError("Observations must be in [0, 1].")
        super().__init__(obs, weights, offset)

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: jax.Array) -> jax.Array:
        y = x + self.offset
        return self.weights.dot(jnp.log(1 + jnp.exp(-y)) + (1 - self.obs) * y)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: jax.Array) -> jax.Array:
        z = jnp.exp(-(x + self.offset))
        return self.weights * (1 / (1 + z) - self.obs)

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: jax.Array) -> jax.Array:
        z = jnp.exp(-(x + self.offset))
        return self.weights * (z / ((1 + z) ** 2))
