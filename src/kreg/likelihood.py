from abc import ABC, abstractmethod
from functools import reduce
from typing import NotRequired, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from kreg.kernel import KroneckerKernel
from kreg.typing import Callable, DataFrame, JAXArray, Series


class LikelihoodData(TypedDict):
    obs: NotRequired[JAXArray]
    weights: NotRequired[JAXArray]
    orig_weights: NotRequired[JAXArray]
    trim_weights: NotRequired[JAXArray]
    offset: JAXArray
    mat: BCOO


class Likelihood(ABC):
    def __init__(
        self, obs: str, weights: str | None = None, offset: str | None = None
    ) -> None:
        self.obs = obs
        self.weights = weights
        self.offset = offset

        self.data: LikelihoodData

    @property
    def size(self) -> int | None:
        if not self.data:
            raise ValueError("No data attached.")
        return len(self.data["obs"])

    @property
    @abstractmethod
    def inv_link(self) -> Callable:
        """Inverse link function to translate parameter from linear space to
        prediction space.
        """

    def _validate_data(self, data: DataFrame, train: bool = True) -> DataFrame:
        """Add necessary validation for the data for each likelihood."""
        if train:
            if data[self.obs].isna().any():
                raise ValueError(
                    "Observations must not contain missing values."
                )
            if self.weights is not None:
                if data[self.weights].isna().any():
                    raise ValueError("Weights must not contain missing values.")
                if not (data[self.weights] >= 0).all():
                    raise ValueError("Weights must be non-negative.")
        if self.offset is not None:
            if data[self.offset].isna().any():
                raise ValueError("Offset must not contain missing values.")
        return data

    def attach(
        self,
        data: DataFrame,
        kernel: KroneckerKernel,
        train: bool = True,
        density: Series | None = None,
    ) -> None:
        data = self._validate_data(data, train=train)
        size = len(data)
        self.data = {
            "offset": jnp.zeros(size)
            if self.offset is None
            else jnp.asarray(data[self.offset]),
            "mat": self.encode(data, kernel, density),
        }
        if train:
            self.data["obs"] = jnp.asarray(data[self.obs])
            self.data["weights"] = (
                jnp.ones(size)
                if self.weights is None
                else jnp.asarray(data[self.weights])
            )
            self.data["orig_weights"] = jnp.asarray(data[self.weights])
            self.data["trim_weights"] = jnp.ones(size)

        @jax.jit
        def mat_apply(x):
            return self.data['mat']@x

        @jax.jit
        def mat_adj_apply(x):
            return self.data['mat'].T@x

        self.mat_apply = mat_apply
        self.mat_adj_apply = mat_adj_apply


    def update_trim_weights(self, w: JAXArray) -> None:
        self.data["trim_weights"] = jnp.asarray(w)
        self.data["weights"] = (
            self.data["orig_weights"] * self.data["trim_weights"]
        )

    def detach(self) -> None:
        del self.data

    @staticmethod
    def encode_integral(data: DataFrame, kernel: KroneckerKernel) -> DataFrame:
        df = reduce(
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

    @staticmethod
    def integral_to_design_mat(
        integral: DataFrame, shape: tuple[int, int]
    ) -> BCOO:
        row, col, val = (
            jnp.asarray(integral["row_index"]),
            jnp.asarray(integral["col_index"]),
            jnp.asarray(integral["val"]),
        )
        indices = jnp.vstack([row, col]).T
        return BCOO((val, indices), shape=shape)

    @staticmethod
    def encode(
        data: DataFrame,
        kernel: KroneckerKernel,
        density: Series | None = None,
    ) -> BCOO:
        shape = len(data), len(kernel)
        df = Likelihood.encode_integral(data, kernel)

        # normalization
        if density is not None:
            if not isinstance(density, Series):
                raise TypeError(
                    "density must be a pandas Series with index coincide with "
                    "the kernel dimensions."
                )
            density = density.rename("density").reset_index()
            kernel_span = kernel.span
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

        return Likelihood.integral_to_design_mat(df, shape)

    def get_lin_param(self, x: JAXArray) -> JAXArray:
        return self.mat_apply(x) + self.data["offset"]

    def get_param(self, x: JAXArray) -> JAXArray:
        return self.inv_link(self.get_lin_param(x))

    @abstractmethod
    def nll_terms(self, x: JAXArray) -> JAXArray:
        """Terms of the negative log likelihood."""

    @abstractmethod
    def objective(self, x: JAXArray) -> JAXArray:
        """Negative log likelihood."""

    @abstractmethod
    def gradient(self, x: JAXArray) -> JAXArray:
        """Gradient of the negative log likelihood."""

    @abstractmethod
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        """Diagonal component of Hessian of the negative log likelihood."""

    def hessian(self, x: JAXArray) -> Callable:
        diag = self.hessian_diag(x)

        def op_hess(x: JAXArray) -> JAXArray:
            return self.mat_adj_apply(diag * (self.mat_apply(x)))

        return op_hess

    def hessian_matrix(self, x: JAXArray) -> JAXArray:
        diag = jnp.diag(self.hessian_diag(x))
        return self.data["mat"].T @ (diag @ self.data["mat"])


class BinomialLikelihood(Likelihood):
    def _validate_data(self, data: DataFrame, train: bool = True) -> DataFrame:
        data = super()._validate_data(data, train=train)
        if train:
            if not (
                (data[self.obs] >= 0).all() and (data[self.obs] <= 1).all()
            ):
                raise ValueError("Observations must be in [0, 1].")
        return data

    @property
    def inv_link(self) -> Callable:
        return expit

    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["weights"] * (
            jnp.log(1 + jnp.exp(-y)) + (1 - self.data["obs"]) * y
        )

    def objective(self, x: JAXArray) -> JAXArray:
        z = self.get_lin_param(x)
        return self.data["weights"].dot(
            jnp.logaddexp(0, z) - self.data["obs"] * z
            )

    def gradient(self, x: JAXArray) -> JAXArray:
        z = self.get_lin_param(x)
        sig_z = jax.scipy.special.expit(z)
        fz = (
            self.data["weights"] * (sig_z - self.data["obs"])
        )
        return self.mat_adj_apply(fz)

    def hessian_diag(self, x: JAXArray) -> JAXArray:
        z = self.get_lin_param(x)
        sig_z = expit(z)
        diag = self.data["weights"] * sig_z * (1.0 - sig_z)
        return diag


class GaussianLikelihood(Likelihood):
    @property
    def inv_link(self) -> Callable:
        return identity

    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return 0.5 * self.data["weights"] * (self.data["obs"] - y) ** 2

    def objective(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return 0.5 * self.data["weights"].dot((self.data["obs"] - y) ** 2)

    def gradient(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.mat_adj_apply(
            self.data["weights"] * (y - self.data["obs"])
        )

    def hessian_diag(self, x: JAXArray) -> JAXArray:
        diag = self.data["weights"]
        return diag


class PoissonLikelihood(Likelihood):
    def _validate_data(self, data: DataFrame, train: bool = True) -> DataFrame:
        data = super()._validate_data(data, train=train)
        if train:
            if not (data[self.obs] >= 0).all():
                raise ValueError("Observations must be non-negative.")
        return data

    @property
    def inv_link(self) -> Callable:
        return jnp.exp

    def nll_terms(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["weights"] * (jnp.exp(y) - self.data["obs"] * y)

    def objective(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["weights"].dot(jnp.exp(y) - self.data["obs"] * y)

    def gradient(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.mat_adj_apply(
            self.data["weights"] * (jnp.exp(y) - self.data["obs"])
        )

    def hessian_diag(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        diag = self.data["weights"] * jnp.exp(y)
        return diag


@jax.jit
def expit(x: JAXArray) -> JAXArray:
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def identity(x: JAXArray) -> JAXArray:
    return x
