import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc
from kreg.likelihood import (
    BinomialLikelihood,
    GaussianLikelihood,
    PoissonLikelihood,
)
from kreg.term import Term


@pytest.fixture
def term() -> Term:
    kernel = KroneckerKernel(
        [
            KernelComponent(
                ["age_mid"],
                vectorize_kfunc(build_matern_three_half_kfunc(rho=8.0)),
            )
        ]
    )
    return Term("intercept", kernel=kernel)


@pytest.fixture
def bad_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs": [-0.5, 0.5, 0.5, 0.5],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "offset": [0.0, 0.0, 0.0, 0.0],
            "age_mid": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.mark.parametrize(
    "likelihood_class", [BinomialLikelihood, PoissonLikelihood]
)
def test_likelihood_validate_data(likelihood_class, bad_data, term):
    term.attach(bad_data)
    likelihood = likelihood_class(obs="obs", weights="weights", offset="offset")

    with pytest.raises(ValueError):
        likelihood.attach(data=bad_data, terms=[term], train=True)

    # This should raise error, _validate_data call is skipped when train=False
    likelihood.attach(data=bad_data, terms=[term], train=False)


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs": [0.2, 0.4, 0.6, 0.8],
            "weights": [1.0, 2.0, 3.0, 4.0],
            "age_mid": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.mark.parametrize(
    "likelihood_class",
    [BinomialLikelihood, GaussianLikelihood, PoissonLikelihood],
)
def test_likelihood_attach_without_weights(likelihood_class, data, term):
    """Regression test: attach used to raise KeyError when weights=None."""
    term.attach(data)
    likelihood = likelihood_class(obs="obs")
    likelihood.attach(data=data, terms=[term], train=True)

    assert jnp.allclose(likelihood.data["weights"], 1.0)
    assert jnp.allclose(likelihood.data["orig_weights"], 1.0)
    assert jnp.allclose(likelihood.data["trim_weights"], 1.0)


def test_likelihood_attach_with_weights(data, term):
    term.attach(data)
    likelihood = GaussianLikelihood(obs="obs", weights="weights")
    likelihood.attach(data=data, terms=[term], train=True)

    assert jnp.allclose(likelihood.data["weights"], data["weights"].to_numpy())
    assert jnp.allclose(
        likelihood.data["orig_weights"], data["weights"].to_numpy()
    )
    # trimming scales the working weights but keeps the originals
    likelihood.update_trim_weights(jnp.asarray([1.0, 0.0, 1.0, 0.0]))
    assert np.allclose(likelihood.data["weights"], [1.0, 0.0, 3.0, 0.0])
    assert jnp.allclose(
        likelihood.data["orig_weights"], data["weights"].to_numpy()
    )


def test_likelihood_size_before_attach() -> None:
    with pytest.raises(ValueError, match="attach"):
        _ = GaussianLikelihood("obs").size


LINKS = {
    BinomialLikelihood: lambda p: jnp.log(p / (1 - p)),
    GaussianLikelihood: lambda y: y,
    PoissonLikelihood: jnp.log,
}


@pytest.mark.parametrize("likelihood_class", list(LINKS))
def test_likelihood_derivatives(likelihood_class, data, term) -> None:
    term.attach(data)
    likelihood = likelihood_class(obs="obs", weights="weights")
    likelihood.attach(data, terms=[term])
    x = jnp.asarray(np.random.default_rng(0).normal(size=term.size))
    v = jnp.asarray(np.random.default_rng(1).normal(size=term.size))

    assert jnp.allclose(likelihood.objective(x), likelihood.nll_terms(x).sum())
    assert jnp.allclose(
        likelihood.gradient(x), jax.grad(likelihood.objective)(x)
    )
    hess = jax.hessian(likelihood.objective)(x)
    assert jnp.allclose(likelihood.hessian(x)(v), hess @ v)
    assert jnp.allclose(likelihood.hessian_matrix(x), hess)

    # the saturated fit has zero deviance
    x_sat = LINKS[likelihood_class](likelihood.data["obs"])
    assert jnp.allclose(likelihood.nll_terms(x_sat), 0.0, atol=1e-12)


def test_binomial_inv_link_stable_in_tails() -> None:
    inv_link = BinomialLikelihood(obs="obs").inv_link
    z = jnp.array([-800.0, 800.0])
    assert jnp.allclose(inv_link(z), jnp.array([0.0, 1.0]))
    assert jnp.isfinite(jax.vmap(jax.grad(inv_link))(z)).all()
