import numpy as np
import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc
from kreg.likelihood import GaussianLikelihood
from kreg.model import KernelRegModel
from kreg.term import Term


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs": [1.0, 1.0, 1.0],
            "weights": [1.0, 1.0, 1.0],
            "offset": [1.0, 1.0, 1.0],
            "age_mid": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def data_with_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs": [1.0, 1.0, 1.0, 2.0],
            "weights": [1.0, 1.0, 1.0, 1.0],
            "offset": [1.0, 1.0, 1.0, 1.0],
            "age_mid": [1.0, 2.0, 3.0, 4.0],
        }
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

    term = Term("intercept", kernel=kernel, lam=1.0)
    model = KernelRegModel([term], likelihood)
    return model


@pytest.mark.parametrize("use_direct", [True, False])
def test_model_predict(
    model: KernelRegModel, data: pd.DataFrame, use_direct: bool
) -> None:
    model.fit(data, use_direct=use_direct, verbose=False)

    y = model.predict(data)
    assert np.allclose(y, 1.0)


def test_model_fit_trimming(
    model: KernelRegModel, data_with_outlier: pd.DataFrame
) -> None:
    x, trim_weights = model.fit_trimming(
        data_with_outlier,
        inlier_pct=0.75,
        solver_options={"use_direct": True},
    )
    assert np.allclose(x, 0.0)
    assert np.allclose(trim_weights, [1.0, 1.0, 1.0, 0.0])


@pytest.mark.skip(reason="need further development")
def test_model_predict_from_kernel(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    model.fit(data, use_direct=True)

    data_pred = pd.DataFrame(
        {"age_mid": np.linspace(0.0, 3.0, 7), "offset": 1.0}
    )
    y = model.predict(data_pred, from_kernel=True)
    assert np.allclose(y, 1.0)


def test_model_refit_with_new_lam(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    model.fit(data, lam=1.0, use_direct=True, detach=False)
    hess_old = model.hessian_prior_matrix()
    model.fit(data, lam=100.0, use_direct=True, detach=False)
    assert np.allclose(model.hessian_prior_matrix(), 100.0 * hess_old)


@pytest.mark.parametrize("use_direct", [True, False])
def test_model_fit_quiet(
    model: KernelRegModel, data: pd.DataFrame, use_direct: bool, capsys
) -> None:
    model.fit(data, use_direct=use_direct, verbose=False)
    assert capsys.readouterr().out == ""


def test_model_fit_lam_unknown_label(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Unknown term labels"):
        model.fit(data, lam={"intercept": 1.0}, use_direct=True, verbose=False)
    model.fit(
        data, lam={"intercept/age_mid": 2.0}, use_direct=True, verbose=False
    )
    assert model.terms[0].lam == 2.0


def test_model_fit_trimming_keeps_solver_options(
    model: KernelRegModel, data_with_outlier: pd.DataFrame
) -> None:
    solver_options = {"use_direct": True, "verbose": False}
    model.fit_trimming(
        data_with_outlier, inlier_pct=0.75, solver_options=solver_options
    )
    assert solver_options == {"use_direct": True, "verbose": False}


def test_model_predict_before_fit(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="fit"):
        model.predict(data)


def test_model_fit_cg_matches_direct(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    data = data.assign(obs=[0.0, 1.0, 2.0])
    x0 = np.zeros(3)
    x_cg, _ = model.fit(data, x0=x0, gtol=1e-8, verbose=False)
    x_direct, _ = model.fit(
        data, x0=x0, gtol=1e-8, use_direct=True, verbose=False
    )
    assert np.allclose(x_cg, x_direct, atol=1e-6)


def test_model_fit_without_weights(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    model.likelihood = GaussianLikelihood(obs="obs", offset="offset")
    x, _ = model.fit(data.drop(columns="weights"), verbose=False)
    assert x.shape == (3,)
    assert np.allclose(model.predict(data), 1.0)


def test_model_multi_term(model: KernelRegModel, data: pd.DataFrame) -> None:
    data = data.assign(cov=[1.0, 2.0, 3.0])
    data["obs"] = data["offset"] + 2.0 * data["cov"]
    model.terms.append(Term("cov"))
    x, _ = model.fit(data, use_direct=True, verbose=False)
    # unpenalized covariate explains the data; kernel term shrinks to zero
    assert np.allclose(x, [0.0, 0.0, 0.0, 2.0])
    assert np.allclose(model.predict(data), data["obs"])


def test_model_fit_lam_scalar(
    model: KernelRegModel, data: pd.DataFrame
) -> None:
    model.fit(data, lam=3.0, use_direct=True, verbose=False)
    assert model.terms[0].lam == 3.0


def test_model_fit_trimming_no_trim(
    model: KernelRegModel, data_with_outlier: pd.DataFrame
) -> None:
    x, trim_weights = model.fit_trimming(
        data_with_outlier,
        inlier_pct=1.0,
        solver_options={"use_direct": True, "verbose": False},
    )
    assert np.allclose(trim_weights, 1.0)
    assert not np.allclose(x, 0.0)
