import jax.numpy as jnp
import pandas as pd

from kreg.kernel.component import KernelComponent
from kreg.kernel.factory import build_exp_similarity_kfunc, vectorize_kfunc


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
