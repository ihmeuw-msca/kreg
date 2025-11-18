import jax.numpy as jnp
import pandas as pd

from kreg.kernel.component import KernelComponent
from kreg.kernel.factory import (
    build_exp_similarity_kfunc,
    build_matern_three_half_kfunc,
    vectorize_kfunc,
)


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
