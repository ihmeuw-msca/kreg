import itertools

import jax.numpy as jnp
import numpy as np

from kreg.kernel.dimension import Dimension
from kreg.typing import (
    DataFrame,
    DimensionColumns,
    JAXArray,
    KernelFunction,
    NDArray,
)

DimensionConfig = str | tuple[str, DimensionColumns | None]


class KernelComponent:
    """Kernel component used to build Kronecker Kernel.

    Parameters
    ----------
    name
        Column name(s) in the dataframe corresponding to the coordinates of the
        kernel.
    kfunc
        Kernel function for computing the kernel matrix for the component.

    """

    def __init__(
        self,
        dim_config: DimensionConfig | list[DimensionConfig],
        kfunc: KernelFunction,
    ) -> None:
        self.dimensions: Dimension | list[Dimension]
        if isinstance(dim_config, list):
            self.dimensions = [
                Dimension.from_config(config) for config in dim_config
            ]
        else:
            self.dimensions = Dimension.from_config(dim_config)
        self.kfunc = kfunc

    @property
    def span(self) -> NDArray:
        if isinstance(self.dimensions, list):
            span = np.asarray(
                list(itertools.product(*[dim.span for dim in self.dimensions])),
            )
        else:
            span = self.dimensions.span
        return span

    @property
    def size(self) -> int:
        return len(self.span)

    def set_span(self, data: DataFrame) -> None:
        if not hasattr(self, "_span"):
            if isinstance(self.dimensions, list):
                for dimension in self.dimensions:
                    dimension.set_span(data)

            else:
                self.dimensions.set_span(data)

    def build_kmat(self, nugget: float = 0.0) -> JAXArray:
        span = jnp.asarray(self.span)
        return self.kfunc(span, span) + nugget * jnp.identity(len(self))

    def __len__(self) -> int:
        return len(self.span)
