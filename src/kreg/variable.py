import functools

import numpy as np
import pandas as pd

from kreg.kernel import KroneckerKernel


class Variable:
    def __init__(
        self, name: str, kernel: KroneckerKernel | None = None
    ) -> None:
        self.name = name
        self.kernel = kernel

    @property
    def size(self) -> int:
        return 1 if self.kernel is None else len(self.kernel)

    @property
    def identifier(self) -> str:
        if self.kernel is None:
            return self.name
        return self.name + "/" + self.kernel.identifier

    def attach(self, data: pd.DataFrame) -> None:
        if self.kernel is not None:
            self.kernel.attach(data)

    def clear_matrices(self) -> None:
        if self.kernel is not None:
            self.kernel.clear_matrices()

    def encode(
        self, data: pd.DataFrame, density: pd.Series | None = None
    ) -> pd.DataFrame:
        df = _encode_integral(data, self.kernel)

        # normalization
        if density is not None and self.kernel is not None:
            if not isinstance(density, pd.Series):
                raise TypeError(
                    "density must be a pandas Series with index coincide with "
                    "the kernel dimensions."
                )
            density = density.rename("density").reset_index()
            kernel_span = self.kernel.span
            missing_cols = set(kernel_span.columns) - set(density.columns)
            if missing_cols:
                raise ValueError(
                    f"Please provide {missing_cols} as the density index."
                )
            matched_density = kernel_span.merge(density, how="left")
            if matched_density["density"].isna().any():
                raise ValueError(
                    "Missing density value for certain kernel dimension."
                )
            density = matched_density["density"].to_numpy()
            df["val"] *= density[df["col_index"].to_numpy()]
        df["val"] /= df.groupby("row_index")["val"].transform("sum")

        if self.name != "intercept":
            cov_mask = (
                data[self.name]
                .rename("mask")
                .to_frame()
                .reset_index(names="row_index")
            )
            df = df.merge(cov_mask, on="row_index")
            df["val"] *= df["mask"]
            df.drop(columns=["mask"], inplace=True)
        df = df.query("val != 0.0").reset_index(drop=True)
        return df


def _encode_integral(
    data: pd.DataFrame, kernel: KroneckerKernel | None
) -> pd.DataFrame:
    if kernel is None:
        df = pd.DataFrame(
            {"row_index": range(len(data)), "col_index": 0, "val": 1.0}
        )
        return df
    df = functools.reduce(
        lambda x, y: x.merge(y, on="row_index", how="outer"),
        (dimension.build_mat(data) for dimension in kernel.dimensions),
    )
    dim_sizes = [len(dimension) for dimension in kernel.dimensions]
    dim_names = [dimension.name for dimension in kernel.dimensions]
    res_sizes = np.hstack([1, np.cumprod(dim_sizes[::-1][:-1], dtype=int)])[
        ::-1
    ]

    df["col_index"] = 0
    df["val"] = 1.0
    for dim_name, res_size in zip(dim_names, res_sizes):
        df["col_index"] += df[f"{dim_name}_col_index"] * res_size
        df["val"] *= df[f"{dim_name}_val"]
    return df
