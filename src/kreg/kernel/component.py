import jax.numpy as jnp

from kreg.typing import DataFrame, JAXArray, KernelFunction


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

    def __init__(self, name: str | list[str], kfunc: KernelFunction) -> None:
        self.name = name
        self.kfunc = kfunc
        self._grid: JAXArray
        self._span: DataFrame

    @property
    def grid(self) -> JAXArray:
        if not hasattr(self, "_grid"):
            raise ValueError(
                "Kernel component does not have data, please attach dataframe"
                "first."
            )
        return self._grid

    @property
    def span(self) -> DataFrame:
        if not hasattr(self, "_span"):
            raise ValueError(
                "Kernel component does not have data, please attach dataframe"
                "first."
            )
        return self._span

    def attach(self, data: DataFrame) -> None:
        if not hasattr(self, "_grid"):
            self._grid = jnp.asarray(
                data.sort_values(self.name)[self.name].drop_duplicates()
            )
            self._span = DataFrame(
                data=self.grid,
                columns=[self.name]
                if isinstance(self.name, str)
                else self.name,
            )

    def build_kmat(self, nugget: float = 0.0) -> JAXArray:
        return self.kfunc(self.grid, self.grid) + nugget * jnp.identity(
            len(self.grid)
        )

    def __len__(self) -> int:
        return len(self.grid)
