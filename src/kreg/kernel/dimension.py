from typing import Literal

import numpy as np
from msca.integrate.integration_weights import build_integration_weights

from kreg.typing import DataFrame, NDArray


class Dimension:
    def __init__(
        self,
        name: str,
        coords: tuple[str, ...] | None = None,
        interval: tuple[str, str] | None = None,
    ) -> None:
        if interval is not None and coords is not None:
            raise ValueError("cannot use 'interval' and 'coords' together")

        self.name = name
        self.interval = interval
        self.coords = coords

        columns = coords or (interval or [name])
        self.columns = list(columns)

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
    def label(self) -> str:
        if self.coords is not None:
            return "(" + ",".join(self.coords) + ")"
        if self.interval is not None:
            return "(" + "-".join(self.interval) + ")"
        return self.name

    def set_span(
        self, data: DataFrame, rule: Literal["midpoint"] = "midpoint"
    ) -> None:
        if hasattr(self, "_span") and hasattr(self, "_grid"):
            return None

        data = (
            data[self.columns]
            .drop_duplicates()
            .sort_values(by=self.columns, ignore_index=True)
            .to_numpy()
        )

        if self.interval is None:
            self._span = data.ravel() if self.coords is None else data
            self._grid = self._span.copy()
        else:
            grid = np.hstack([data[0, 0], data[:, 1]])
            if not np.allclose(data[:, 0], grid[:-1]):
                raise ValueError("Range intervals contain gap(s)")
            self._grid = grid
            if rule == "midpoint":
                self._span = data.mean(axis=1)
            else:
                raise ValueError(f"Unknown rule='{rule}'")

    def build_mat(
        self, data: DataFrame, rule: Literal["midpoint"] = "midpoint"
    ) -> DataFrame:
        if self.interval is None:
            row_index = np.arange(len(data), dtype=int)
            col_index = (
                data[self.columns]
                .merge(
                    DataFrame(self.span, columns=self.columns).reset_index(),
                    on=self.columns,
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
