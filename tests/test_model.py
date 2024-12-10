import numpy as np
import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc
from kreg.likelihood import GaussianLikelihood
from kreg.model import KernelRegModel


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
def model() -> KernelRegModel:
    kernel = KroneckerKernel(
        [
            KernelComponent(
                "age_mid", vectorize_kfunc(build_matern_three_half_kfunc(8.0))
            )
        ]
    )
    likelihood = GaussianLikelihood(
        obs="obs", weights="weights", offset="offset"
    )

    model = KernelRegModel(kernel, likelihood, lam=1.0)
    return model


def test_model_predict(model: KernelRegModel, data: pd.DataFrame):
    model.fit(data, use_direct=True)

    y = model.predict(data)
    assert np.allclose(y, 1.0)