from functools import partial

import jax
import jax.numpy as jnp


class LogisticLikelihood:
    def __init__(
        self,
        obs_counts: jax.Array,
        sample_sizes: jax.Array,
    ) -> None:
        assert (obs_counts <= sample_sizes).all()
        self.sample_sizes = sample_sizes
        self.obs_counts = obs_counts
        self.beta_smoothness = jnp.max(sample_sizes) / 4
        self.N = len(obs_counts)

    def loss_single(y: jax.Array, k: jax.Array, n: jax.Array) -> jax.Array:
        return n * jnp.log(1 + jnp.exp(-y)) + (n - k) * y

    @partial(jax.jit, static_argnums=0)
    def f(self, y: jax.Array) -> jax.Array:
        return jnp.sum(
            LogisticLikelihood.loss_single(
                y, self.obs_counts, self.sample_sizes
            )
        )

    grad_f = jax.jit(jax.grad(f, argnums=1), static_argnums=0)
    val_grad_f = jax.jit(jax.value_and_grad(f, argnums=1), static_argnums=0)

    @partial(jax.jit, static_argnums=0)
    def H_diag(self, y):
        z = jnp.exp(y)
        return self.sample_sizes * (z / ((z + 1) ** 2))
