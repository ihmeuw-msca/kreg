from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_gaussianRBF_kfunc, vectorize_kfunc
from kreg.term import Term


def test_label():
    dummy_kfunc = vectorize_kfunc(build_gaussianRBF_kfunc(1.0))

    kernel = KroneckerKernel([KernelComponent("x", dummy_kfunc)])
    term = Term("intercept", kernel=kernel)
    assert term.label == "intercept/x"

    kernel = KroneckerKernel([KernelComponent(["x", "y"], dummy_kfunc)])
    term = Term("intercept", kernel=kernel)
    assert term.label == "intercept/x*y"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "coords": ("x", "y", "z")}], dummy_kfunc
            )
        ]
    )
    term = Term("intercept", kernel=kernel)
    assert term.label == "intercept/(x,y,z)"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "interval": ("a", "b")}], dummy_kfunc
            )
        ]
    )
    term = Term("intercept", kernel=kernel)
    assert term.label == "intercept/(a-b)"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "coords": ("x", "y", "z")}], dummy_kfunc
            ),
            KernelComponent(
                [{"name": "dummy", "interval": ("a", "b")}], dummy_kfunc
            ),
        ]
    )
    term = Term("intercept", kernel=kernel)
    assert term.label == "intercept/(x,y,z)*(a-b)"


import numpy as np
import pandas as pd
import pytest

from kreg.likelihood import Likelihood


@pytest.fixture
def kernel() -> KroneckerKernel:
    kfunc = vectorize_kfunc(build_gaussianRBF_kfunc(1.0))
    return KroneckerKernel(
        [
            KernelComponent("loc", kfunc),
            KernelComponent({"name": "age", "interval": ("a0", "a1")}, kfunc),
        ]
    )


@pytest.fixture
def span_data() -> pd.DataFrame:
    # loc span: [1, 2]; age grid: [0, 1, 2] -> 4 columns, row-major (loc, age)
    return pd.DataFrame(
        {
            "loc": [1, 1, 2, 2],
            "a0": [0.0, 1.0, 0.0, 1.0],
            "a1": [1.0, 2.0, 1.0, 2.0],
        }
    )


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loc": [2, 1, 1, 2],
            "a0": [0.0, 1.0, 0.0, 1.0],
            "a1": [1.0, 2.0, 2.0, 2.0],
            "cov": [1.0, 0.0, 2.0, 3.0],
        }
    )


def test_encode_column_order(kernel, span_data, data) -> None:
    term = Term("intercept", kernel=kernel)
    term.attach(span_data)
    df = term.encode(data)
    span = kernel.span
    # row i of `data` must land in the span row(s) with the same (loc, age)
    for row, col, val in zip(df["row_index"], df["col_index"], df["val"]):
        assert span.loc[col, "loc"] == data.loc[row, "loc"]
        assert data.loc[row, "a0"] <= span.loc[col, "a0"]
        assert span.loc[col, "a1"] <= data.loc[row, "a1"]
    assert df.groupby("row_index")["val"].sum().tolist() == [1.0] * 4
    assert df["col_index"].tolist() == [2, 1, 0, 1, 3]


def test_encode_density(kernel, span_data, data) -> None:
    term = Term("intercept", kernel=kernel)
    term.attach(span_data)
    span = kernel.span
    density = pd.Series(
        [1.0, 3.0, 1.0, 1.0], index=pd.MultiIndex.from_frame(span)
    )
    df = term.encode(data, density=density)
    # row 2 covers both age cells of loc 1; density 1:3 -> weights 0.25:0.75
    assert df.loc[df["row_index"] == 2, "val"].tolist() == [0.25, 0.75]

    with pytest.raises(TypeError):
        term.encode(data, density=density.to_numpy())
    with pytest.raises(ValueError, match="density index"):
        term.encode(data, density=density.droplevel("a1"))
    with pytest.raises(ValueError, match="Missing density"):
        term.encode(data, density=density.iloc[:-1])


def test_encode_covariate(kernel, span_data, data) -> None:
    term = Term("cov", kernel=kernel)
    term.attach(span_data)
    df = term.encode(data)
    assert 1 not in df["row_index"].tolist()  # cov == 0 rows are dropped
    assert df.loc[df["row_index"] == 2, "val"].tolist() == [1.0, 1.0]


def test_encode_no_kernel(data) -> None:
    df = Term("cov").encode(data)
    assert df["col_index"].tolist() == [0, 0, 0]
    assert df["val"].tolist() == [1.0, 2.0, 3.0]


def test_likelihood_encode_two_terms(kernel, span_data, data) -> None:
    terms = [Term("intercept", kernel=kernel), Term("cov")]
    for term in terms:
        term.attach(span_data)
    mat = Likelihood.encode(data, terms).todense()
    assert mat.shape == (4, 5)
    assert np.allclose(mat[:, 4], data["cov"])
    assert np.allclose(mat[:, :4].sum(axis=1), 1.0)
    assert mat[2, 0] == 0.5 and mat[2, 1] == 0.5
