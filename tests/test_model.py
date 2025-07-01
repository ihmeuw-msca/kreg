import numpy as np
import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc
from kreg.likelihood import GaussianLikelihood
from kreg.model import KernelRegModel
from kreg.variable import Variable


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            obs=[1.0, 1.0, 1.0],
            weights=[1.0, 1.0, 1.0],
            offset=[1.0, 1.0, 1.0],
            age_mid=[1.0, 2.0, 3.0],
        )
    )


@pytest.fixture
def data_with_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            obs=[1.0, 1.0, 1.0, 2.0],
            weights=[1.0, 1.0, 1.0, 1.0],
            offset=[1.0, 1.0, 1.0, 1.0],
            age_mid=[1.0, 2.0, 3.0, 4.0],
        )
    )


@pytest.fixture
def model() -> KernelRegModel:
    kernel = KroneckerKernel(
        [
            KernelComponent(
                ["age_mid"], vectorize_kfunc(build_matern_three_half_kfunc(8.0))
            )
        ]
    )
    likelihood = GaussianLikelihood(
        obs="obs", weights="weights", offset="offset"
    )

    variable = Variable("intercept", kernel=kernel)
    model = KernelRegModel([variable], likelihood, lam=1.0)
    return model


def test_model_predict(model: KernelRegModel, data: pd.DataFrame) -> None:
    model.fit(data, use_direct=True)

    y = model.predict(data)
    assert np.allclose(y, 1.0)


def test_model_fit_trimming(
    model: KernelRegModel, data_with_outlier: pd.DataFrame
) -> None:
    x, trim_weights = model.fit_trimming(
        data_with_outlier,
        inlier_pct=0.75,
        solver_options=dict(use_direct=True),
    )
    assert np.allclose(x, 0.0)
    assert np.allclose(trim_weights, [1.0, 1.0, 1.0, 0.0])


@pytest.mark.skip(reason="need further development")
def test_model_predict_from_kernel(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    model.fit(data, use_direct=True)

    data_pred = pd.DataFrame(dict(age_mid=np.linspace(0.0, 3.0, 7), offset=1.0))
    y = model.predict(data_pred, from_kernel=True)
    assert np.allclose(y, 1.0)
