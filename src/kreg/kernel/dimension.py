from typing import Literal

import numpy as np
from msca.integrate.integration_weights import build_integration_weights

from kreg.typing import DataFrame, NDArray


class Dimension:
    """One axis of a kernel grid.

    A dimension is one of three kinds, decided by the constructor arguments.

    - **plain**: a single column ``name``; data must match a grid value
      exactly.
    - **coords**: a tuple of columns that together identify a grid point,
      for example a location hierarchy. Data must match a tuple exactly.
    - **interval**: a lower and an upper column. The grid is the set of
      cell edges, ``span`` holds the cell midpoints (where the kernel is
      evaluated), and data intervals are spread over the cells they cover
      with integration weights, so they may be wider than a cell.

    ``span`` is the array of grid points, ``grid`` the array of cell edges
    (equal to ``span`` except for interval dimensions). Both are read from
    the first data frame passed to ``set_span`` and are then frozen.

    Parameters
    ----------
    name
        Dimension name; the data column for a plain dimension.
    coords
        Columns of a coordinate tuple.
    interval
        Lower and upper bound columns of an interval.

    """

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
        """Read the grid from the unique values of ``data``; no-op once set.

        For an interval dimension the rows of ``data`` must be the grid
        cells themselves, contiguous and without gaps.
        """
        if hasattr(self, "_span") and hasattr(self, "_grid"):
            return

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
        """Encode ``data`` along this dimension.

        Returns a long-format frame with ``row_index``, ``<name>_col_index``
        and ``<name>_val``; one entry per row for plain and coords
        dimensions, one per covered cell for interval dimensions. Values
        outside the grid raise ``ValueError``.
        """
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
            missing = np.isnan(col_index)
            if missing.any():
                unseen = data.loc[missing, self.columns].drop_duplicates()
                raise ValueError(
                    f"Dimension '{self.name}' has values that are not in its "
                    f"span: {unseen.to_numpy().tolist()[:5]}."
                )
            col_index = col_index.astype(int)
            val = np.ones(len(data))
        else:
            lower = data[self.columns[0]].to_numpy()
            upper = data[self.columns[1]].to_numpy()
            outside = (lower < self.grid[0]) | (upper > self.grid[-1])
            if outside.any():
                raise ValueError(
                    f"Dimension '{self.name}' has intervals outside its grid "
                    f"[{self.grid[0]}, {self.grid[-1]}]: "
                    f"{data.loc[outside, self.columns].to_numpy().tolist()[:5]}."
                )
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
