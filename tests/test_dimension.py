import pandas as pd
import pytest

from kreg.kernel.dimension import Dimension


@pytest.fixture
def plain() -> Dimension:
    dim = Dimension("age")
    dim.set_span(pd.DataFrame({"age": [1.0, 2.0, 3.0]}))
    return dim


@pytest.fixture
def interval() -> Dimension:
    dim = Dimension("age", interval=("age_start", "age_end"))
    dim.set_span(
        pd.DataFrame({"age_start": [0.0, 1.0, 2.0], "age_end": [1.0, 2.0, 3.0]})
    )
    return dim


def test_build_mat_plain_unseen_value(plain: Dimension) -> None:
    with pytest.raises(ValueError, match="'age'.*not in its span.*4.0"):
        plain.build_mat(pd.DataFrame({"age": [2.0, 4.0]}))


@pytest.mark.parametrize("bounds", [(2.5, 4.0), (-1.0, 0.5)])
def test_build_mat_interval_outside_grid(interval: Dimension, bounds) -> None:
    lower, upper = bounds
    with pytest.raises(ValueError, match="'age'.*outside its grid"):
        interval.build_mat(
            pd.DataFrame({"age_start": [lower], "age_end": [upper]})
        )


def test_build_mat_interval_inside_grid(interval: Dimension) -> None:
    df = interval.build_mat(
        pd.DataFrame({"age_start": [0.5], "age_end": [1.5]})
    )
    assert df["age_col_index"].tolist() == [0, 1]
    assert df["age_val"].tolist() == [0.5, 0.5]
