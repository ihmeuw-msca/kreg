from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict

import jax
import jax.numpy as jnp
import pandas as pd
from jax.experimental.sparse import BCOO
from jax.scipy.special import xlogy

from kreg.term import Term
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
        terms: list[Term],
        train: bool = True,
        density: dict[str, Series] | None = None,
    ) -> None:
        data = self._validate_data(data, train=train)
        size = len(data)
        self.data = {
            "offset": jnp.zeros(size)
            if self.offset is None
            else jnp.asarray(data[self.offset]),
            "mat": self.encode(data, terms, density),
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
            return self.data["mat"] @ x

        @jax.jit
        def mat_adj_apply(x):
            return self.data["mat"].T @ x

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
    def encode(
        data: DataFrame,
        terms: list[Term],
        density: dict[str, Series] | None = None,
    ) -> BCOO:
        variable_sizes = pd.DataFrame(
            {
                "variable_index": range(len(terms)),
                "size": [v.size for v in terms],
            }
        )
        variable_sizes["shift"] = (
            variable_sizes["size"].cumsum().shift(1, fill_value=0)
        )
        shape = len(data), variable_sizes["size"].sum()

        density = density or {}
        df: pd.DataFrame = pd.concat(
            [
                variable.encode(
                    data, density=density.get(variable.label)
                ).assign(variable_index=variable_index)
                for variable_index, variable in enumerate(terms)
            ],
            axis=0,
            ignore_index=True,
        )
        df = df.merge(
            variable_sizes[["variable_index", "shift"]],
            on="variable_index",
            how="left",
        )
        df["col_index"] += df["shift"]
        df.drop(columns=["shift"], inplace=True)
        row, col, val = (
            jnp.asarray(df["row_index"]),
            jnp.asarray(df["col_index"]),
            jnp.asarray(df["val"]),
        )
        indices = jnp.vstack([row, col]).T
        return BCOO((val, indices), shape=shape)

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
        z = self.get_lin_param(x)
        t = jnp.logaddexp(0, z) - self.data["obs"] * z
        t_min = -(
            xlogy(self.data["obs"], self.data["obs"])
            + xlogy(1 - self.data["obs"], 1 - self.data["obs"])
        )
        return self.data["weights"] * (t - t_min)

    def objective(self, x: JAXArray) -> JAXArray:
        return self.nll_terms(x).sum()

    def gradient(self, x: JAXArray) -> JAXArray:
        z = self.get_lin_param(x)
        sig_z = jax.scipy.special.expit(z)
        fz = self.data["weights"] * (sig_z - self.data["obs"])
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
        return self.nll_terms(x).sum()

    def gradient(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.mat_adj_apply(self.data["weights"] * (y - self.data["obs"]))

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
        t = jnp.exp(y) - self.data["obs"] * y
        t_min = self.data["obs"] - xlogy(self.data["obs"], self.data["obs"])
        return self.data["weights"] * (t - t_min)

    def objective(self, x: JAXArray) -> JAXArray:
        return self.nll_terms(x).sum()

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
