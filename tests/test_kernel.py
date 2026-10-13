import functools

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import (
    build_exp_similarity_kfunc,
    build_matern_three_half_kfunc,
    vectorize_kfunc,
)
from kreg.term import Term


def test_hierarchical_kernel() -> None:
    kfunc = vectorize_kfunc(
        build_exp_similarity_kfunc(jnp.asarray([10.0, 10.0, 10.0]))
    )
    kernel_component = KernelComponent(
        # we decided to always provide a list of dim_configs here, it can be
        # a str for name or a dict for more options
        [
            {
                "name": "age_mid",
                "coords": ("super_region_id", "region_id", "location_id"),
            }
        ],
        kfunc,
    )

    data = pd.DataFrame(
        {
            "super_region_id": [0, 0, 0, 1, 1, 1],
            "region_id": [0, 0, 1, 2, 2, 3],
            "location_id": [0, 1, 2, 3, 4, 5],
        }
    )

    kernel_component.set_span(data)
    # six unique rows and three location id columns
    assert kernel_component.span.shape == (6, 3)
    kmat = kernel_component.build_kmat()
    assert kmat.shape == (6, 6)


def test_kernel_component_normalization() -> None:
    kfunc_1 = build_matern_three_half_kfunc(rho=5.0)
    kfunc_2 = build_matern_three_half_kfunc(rho=10.0)

    def kfunc(x, y):
        return kfunc_1(x, y) + kfunc_2(x, y)

    kernel_component = KernelComponent(["x"], vectorize_kfunc(kfunc))

    data = pd.DataFrame({"x": jnp.linspace(0, 10, 11)})

    kernel_component.set_span(data)

    kmat = kernel_component.build_kmat(nugget=0.0, normalize=True)
    assert jnp.allclose(jnp.diag(kmat), 1.0)

    kmat = kernel_component.build_kmat(nugget=0.0, normalize=False)
    assert jnp.allclose(jnp.diag(kmat), 2.0)


@pytest.fixture
def kernel() -> KroneckerKernel:
    return KroneckerKernel(
        [
            KernelComponent(
                ["loc"], vectorize_kfunc(build_exp_similarity_kfunc(0.5))
            ),
            KernelComponent(
                ["age"], vectorize_kfunc(build_matern_three_half_kfunc(5.0))
            ),
        ],
        nugget=1e-6,
    )


@pytest.fixture
def grid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loc": np.repeat([1.0, 2.0, 3.0], 4),
            "age": np.tile(np.arange(4.0), 3),
        }
    )


def test_kronecker_kernel_attach_detach(kernel, grid):
    assert kernel.status == "detached"
    kernel.attach(grid)
    assert kernel.status == "attached"
    assert len(kernel) == 12
    assert kernel.op_k.shape == (12, 12)
    assert [mat.shape for mat in kernel.kmats] == [(3, 3), (4, 4)]

    kernel.clear_matrices()
    assert kernel.status == "detached"
    assert not hasattr(kernel, "op_k")


def test_kronecker_kernel_operators(kernel, grid):
    kernel.attach(grid)
    dense_k = functools.reduce(jnp.kron, kernel.kmats)
    x = jnp.asarray(np.random.default_rng(0).normal(size=12))

    # op_k is the Kronecker product of the component matrices
    assert jnp.allclose(kernel.op_k @ x, dense_k @ x)
    assert jnp.allclose(kernel @ x, dense_k @ x)
    assert jnp.allclose(kernel.op_k.to_array(), dense_k)
    # op_p is its inverse
    assert jnp.allclose(kernel.op_p @ (kernel.op_k @ x), x)
    assert jnp.allclose(kernel.op_p.to_array(), jnp.linalg.inv(dense_k))
    # the roots square to k and p
    assert jnp.allclose(kernel.op_root_k @ (kernel.op_root_k @ x), dense_k @ x)
    assert jnp.allclose(
        kernel.op_root_p @ (kernel.op_root_p @ x), kernel.op_p @ x
    )


def test_kronecker_kernel_span_order(kernel, grid):
    """Row i of ``span`` is grid cell i of the Kronecker ordering."""
    kernel.attach(grid)
    span = kernel.span
    assert list(span.columns) == ["loc", "age"]
    # first dimension slowest, matching jnp.kron / row-major ordering
    assert np.allclose(span["loc"], np.repeat([1.0, 2.0, 3.0], 4))
    assert np.allclose(span["age"], np.tile(np.arange(4.0), 3))

    term = Term("intercept", kernel=kernel)
    encoded = term.encode(span)
    assert np.array_equal(encoded["row_index"], np.arange(12))
    assert np.array_equal(encoded["col_index"], np.arange(12))
    assert np.allclose(encoded["val"], 1.0)


@pytest.mark.parametrize("config", ["age_mid", {"name": "age_mid"}])
def test_kernel_component_single_config(config) -> None:
    component = KernelComponent(config, vectorize_kfunc(lambda x, y: x * y))
    assert component.dim_names == ["age_mid"]
