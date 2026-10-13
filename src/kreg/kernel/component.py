import itertools

import jax.numpy as jnp

from kreg.kernel.dimension import Dimension
from kreg.typing import DataFrame, JAXArray, KernelFunction

DimensionConfig = str | dict


class KernelComponent:
    """Kernel component used to build Kronecker Kernel.

    Parameters
    ----------
    dim_configs
        Dimension configurations, one per coordinate of the kernel. Each is a
        column name or a dict of ``Dimension`` keyword arguments. A single
        configuration may be passed without wrapping it in a list.
    kfunc
        Kernel function for computing the kernel matrix for the component.

    """

    def __init__(
        self,
        dim_configs: DimensionConfig | list[DimensionConfig],
        kfunc: KernelFunction,
    ) -> None:
        if isinstance(dim_configs, str | dict):
            dim_configs = [dim_configs]
        self.dimensions: list[Dimension]
        dimensions = []
        for dim_config in dim_configs:
            if isinstance(dim_config, str):
                dimensions.append(Dimension(dim_config))
            else:
                dimensions.append(Dimension(**dim_config))
        self.dimensions = dimensions
        self.kfunc = kfunc

    @property
    def span(self) -> JAXArray:
        span: JAXArray = jnp.asarray(
            list(
                map(
                    jnp.hstack,
                    itertools.product(*[dim.span for dim in self.dimensions]),
                )
            )
        )
        if len(self.dimensions) == 1 and span.shape[1] == 1:
            span = span.ravel()
        return span

    @property
    def size(self) -> int:
        return len(self.span)

    @property
    def dim_names(self) -> list[str]:
        return [dim.name for dim in self.dimensions]

    @property
    def columns(self) -> list[str]:
        columns = []
        for dim in self.dimensions:
            columns.extend(dim.columns)
        return columns

    def set_span(self, data: DataFrame) -> None:
        """This function will only get triggered if the span has not been set."""
        for dimension in self.dimensions:
            dimension.set_span(data)

    def build_kmat(
        self, nugget: float = 0.0, normalize: bool = False
    ) -> JAXArray:
        mat = self.kfunc(self.span, self.span)
        if normalize:
            vec = jnp.sqrt(mat.max(axis=0))
            mat = mat / jnp.outer(vec, vec)
        return mat + nugget * jnp.identity(len(self))

    def __len__(self) -> int:
        return len(self.span)
