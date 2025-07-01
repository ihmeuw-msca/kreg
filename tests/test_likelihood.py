import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc
from kreg.likelihood import BinomialLikelihood, PoissonLikelihood
from kreg.variable import Variable


@pytest.fixture
def variable() -> Variable:
    kernel = KroneckerKernel(
        [
            KernelComponent(
                ["age_mid"],
                vectorize_kfunc(build_matern_three_half_kfunc(rho=8.0)),
            )
        ]
    )
    variable = Variable("intercept", kernel=kernel)
    return variable


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
def test_likelihood_validate_data(likelihood_class, bad_data, variable):
    variable.attach(bad_data)
    likelihood = likelihood_class(obs="obs", weights="weights", offset="offset")

    with pytest.raises(ValueError):
        likelihood.attach(data=bad_data, variables=[variable], train=True)

    # This should raise error, _validate_data call is skipped when train=False
    likelihood.attach(data=bad_data, variables=[variable], train=False)
