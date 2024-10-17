from typing import Literal

import numpy as np
from msca.integrate.integration_weights import build_integration_weights

from kreg.typing import DataFrame, DimensionColumns, NDArray


class Dimension:
    def __init__(
        self, name: str, columns: DimensionColumns | None = None
    ) -> None:
        self.name = name
        self.columns = name if columns is None else columns
        self._grid: NDArray
        self._span: NDArray

    @property
    def span(self) -> NDArray:
        if not hasattr(self, "_span"):
            raise AttributeError("Dimension span is not set")
        return self._span

    @property
    def grid(self) -> NDArray:
        if not hasattr(self, "_grid"):
            raise AttributeError("Dimension grid is not set")
        return self._grid

    @property
    def size(self) -> int:
        return len(self.span)

    def set_span(
        self, data: DataFrame, rule: Literal["midpoint"] = "midpoint"
    ) -> None:
        if not hasattr(self, "_span"):
            if isinstance(self.columns, str):
                self._span = np.unique(data[self.columns])
            else:
                lb, ub = self.columns
                data = (
                    data[[lb, ub]]
                    .drop_duplicates()
                    .sort_values(by=[lb, ub], ignore_index=True)
                )
                grid = np.hstack([data.loc[0, lb], data[ub]])
                if not np.allclose(data[lb], grid[:-1]):
                    raise ValueError("Range intervals contain gap(s)")
                self._grid = grid
                if rule == "midpoint":
                    self._span = np.asarray(data.mean(axis=1))
                else:
                    raise ValueError(f"Unknown rule='{rule}'")

    def build_mat(
        self, data: DataFrame, rule: Literal["midpoint"] = "midpoint"
    ) -> DataFrame:
        if isinstance(self.columns, str):
            row_index = np.arange(len(data), dtype=int)
            col_index = (
                DataFrame({self.columns: self.span})
                .reset_index()
                .merge(data[[self.columns]], on=self.columns, how="left")[
                    "index"
                ]
                .to_numpy()
            )
            val = np.ones(len(data))
        else:
            val, (row_index, col_index) = build_integration_weights(
                data[self.columns[0]].to_numpy(),
                data[self.columns[1]].to_numpy(),
                self.grid,
                rule=rule,
            )
        return DataFrame(
            {
                "row_index": row_index,
                f"{self.name}_col_index": col_index,
                f"{self.name}_val": val,
            }
        )

    def __len__(self) -> int:
        return len(self.span)

    @classmethod
    def from_config(
        cls, config: str | tuple[str, DimensionColumns | None]
    ) -> "Dimension":
        if isinstance(config, str):
            return cls(config)
        return cls(*config)
