from typing import Literal, Self

import numpy as np
from msca.integrate.integration_weights import build_integration_weights

from kreg.typing import DataFrame, NDArray


class Dimension:
    def __init__(
        self, name: str, interval: tuple[str, str] | None = None
    ) -> None:
        self.name = name
        self.interval = interval

        self._grid: NDArray
        self._span: NDArray

    @property
    def span(self) -> NDArray:
        if not hasattr(self, "_span"):
            raise AttributeError("Please set dimension span first")
        return self._span

    @property
    def grid(self) -> NDArray:
        if not hasattr(self, "_grid"):
            raise AttributeError("Please set dimension span first")
        return self._grid

    @property
    def columns(self) -> list[str]:
        return [self.name] if self.interval is None else list(self.interval)

    def set_span(
        self, data: DataFrame, rule: Literal["midpoint"] = "midpoint"
    ) -> None:
        if hasattr(self, "_span") and hasattr(self, "_grid"):
            return None

        data = (
            data[self.columns]
            .drop_duplicates()
            .sort_values(by=self.columns, ignore_index=True)
        )

        if self.interval is None:
            self._span = np.unique(data[self.name])
            self._grid = self._span.copy()
        else:
            lb, ub = self.interval
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
        if self.interval is None:
            row_index = np.arange(len(data), dtype=int)
            col_index = (
                data[[self.name]]
                .merge(
                    DataFrame({self.name: self.span}).reset_index(),
                    on=self.name,
                    how="left",
                )["index"]
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
        cls, config: str | tuple[str, tuple[str, str] | None]
    ) -> Self:
        if isinstance(config, str):
            return cls(config)
        return cls(*config)
