import jax
import jax.numpy as jnp

from kreg.typing import Callable, KernelFunction

# TODO: Add more math description on each kernel function generator


def vectorize_kfunc(k: Callable) -> KernelFunction:
    """Vectorize kernel function.

    Parameters
    ----------
    k
        Kernel function.

    Returns
    -------
    Callable
        Vectorized kernel function.

    """
    return jax.vmap(jax.vmap(k, in_axes=(None, 0)), in_axes=(0, None))


@vectorize_kfunc
def build_exp_similarity_kfunc(exp_a: float) -> KernelFunction:
    """Create exponential similarity kernel function.

    Parameters
    ----------
    exp_a
        Exponential parameter.

    Returns
    -------
    Callable
        Exponential similarity kernel function.

    """
    log_exp = jnp.log(exp_a)

    def k(x, y):
        return jnp.exp(jnp.sum(log_exp * (x != y)))

    return k


@vectorize_kfunc
def build_matern_three_half_kfunc(rho: float) -> KernelFunction:
    """Create Matern 3/2 kernel function.

    Parameters
    ----------
    rho
        Length scale.

    Returns
    -------
    Callable
        Matern 3/2 kernel function.

    """

    def k(x, y):
        d = jnp.sqrt(jnp.sum((x - y) ** 2))
        return (1 + jnp.sqrt(3) * d / rho) * jnp.exp(-jnp.sqrt(5) * d / rho)

    return k


@vectorize_kfunc
def build_matern_five_half_kfunc(rho: float) -> KernelFunction:
    """Create Matern 5/2 kernel function.

    Parameters
    ----------
    rho
        Length scale.

    Returns
    -------
    Callable
        Matern 5/2 kernel function.

    """

    def k(x, y):
        d = jnp.sqrt(jnp.sum((x - y) ** 2))
        return (1 + jnp.sqrt(5) * d / rho + 5 * d**2 / (3 * rho**2)) * jnp.exp(
            -jnp.sqrt(5) * d / rho
        )

    return k


@vectorize_kfunc
def build_gaussianRBF_kfunc(gamma: float) -> KernelFunction:
    """Create Gaussian RBF kernel function.

    Parameters
    ----------
    gamma
        Scaling parameter.

    Returns
    -------
    Callable
        Gaussian RBF kernel function.

    """

    def k(x, y):
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))

    return k


@vectorize_kfunc
def build_shifted_scaled_linear_kfunc(a: float, b: float) -> KernelFunction:
    """Create shifted and scaled linear kernel function.

    Parameters
    ----------
    a
        Shift parameter.
    b
        Scaling parameter.

    Returns
    -------
    Callable
        Shifted and scaled linear kernel function.

    """

    def k(x, y):
        return jnp.dot(x - a, y - a) / (b**2)

    return k


@vectorize_kfunc
def build_RQ_kfunc(alpha: float, gamma: float) -> KernelFunction:
    """Create Rational Quadratic kernel function.

    Parameters
    ----------
    alpha
        Relative weighting of large-scale and small-scale variations.
    gamma
        Length scale.

    Returns
    -------
    Callable
        Rational Quadratic kernel function.

    """

    def k(x, y):
        d2 = jnp.sum((x - y) ** 2)
        return (1 + d2 / (2 * alpha * gamma**2)) ** (-alpha)

    return k
