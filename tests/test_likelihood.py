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
        dict(
            obs=[-0.5, 0.5, 0.5, 0.5],
            weights=[1.0, 1.0, 1.0, 1.0],
            offset=[0.0, 0.0, 0.0, 0.0],
            age_mid=[1.0, 2.0, 3.0, 4.0],
        )
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
