import jax
import jax.numpy as jnp


def vectorize_kfunc(k):
    return jax.vmap(jax.vmap(k, in_axes=(None, 0)), in_axes=(0, None))


def get_exp_similarity_kernel(exp_a):
    log_exp = jnp.log(exp_a)

    def k(x, y):
        return jnp.exp(jnp.sum(log_exp * (x != y)))

    return k


def get_matern_three_half(rho):
    def k(x, y):
        d = jnp.sqrt(jnp.sum((x - y) ** 2))
        return (1 + jnp.sqrt(3) * d / rho) * jnp.exp(-jnp.sqrt(5) * d / rho)

    return k


def get_matern_five_half(rho):
    def k(x, y):
        d = jnp.sqrt(jnp.sum((x - y) ** 2))
        return (1 + jnp.sqrt(5) * d / rho + 5 * d**2 / (3 * rho**2)) * jnp.exp(
            -jnp.sqrt(5) * d / rho
        )

    return k


def get_gaussianRBF(gamma):
    def k(x, y):
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))

    return k


def shifted_scaled_linear_kernel(a, b):
    def k(x, y):
        return jnp.dot(x - a, y - a) / (b**2)

    return k


def get_RQ_kernel(alpha, gamma):
    def k(x, y):
        d2 = jnp.sum((x - y) ** 2)
        return (1 + d2 / (2 * alpha * gamma**2)) ** (-alpha)

    return k
