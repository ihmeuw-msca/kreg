from abc import ABC, abstractmethod
from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from kreg.kernel import KroneckerKernel
from kreg.typing import Callable, DataFrame, JAXArray, NDArray


class Likelihood(ABC):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        self.obs = obs
        self.weights = weights
        self.offset = offset
        self.data: dict[str, JAXArray] = {}

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

    def attach(
        self,
        data: DataFrame,
        kernel: KroneckerKernel,
        train: bool = True,
        density: NDArray | None = None,
    ) -> None:
        if train:
            if not (data[self.weights] >= 0).all():
                raise ValueError("Weights must be non-negative.")
            self.data["obs"] = jnp.asarray(data[self.obs])
            self.data["weights"] = jnp.asarray(data[self.weights])
        self.data["offset"] = jnp.asarray(data[self.offset])
        self.data["mat"] = self.encode(data, kernel, density)

    def detach(self) -> None:
        self.data.clear()

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
        density: NDArray | None = None,
    ) -> JAXArray:
        shape = len(data), len(kernel)
        df = Likelihood.encode_integral(data, kernel)

        # normalization
        if density is not None:
            if len(density) != shape[1]:
                raise ValueError(
                    "Must provide density as the same size with the variables in the kernel."
                )
            df["val"] *= density[df["col_index"].to_numpy()]
        df["val"] /= df.groupby("row_index")["val"].transform("sum")

        return Likelihood.integral_to_design_mat(df, shape)

    @partial(jax.jit, static_argnums=0)
    def get_lin_param(self, x: JAXArray) -> JAXArray:
        return self.data["mat"] @ x + self.data["offset"]

    @partial(jax.jit, static_argnums=0)
    def get_param(self, x: JAXArray) -> JAXArray:
        return self.inv_link(self.get_lin_param(x))

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
            return self.data["mat"].T @ (diag * (self.data["mat"] @ x))

        return op_hess

    def hessian_matrix(self, x: JAXArray) -> JAXArray:
        diag = jnp.diag(self.hessian_diag(x))
        return self.data["mat"].T @ (diag @ self.data["mat"])


class BinomialLikelihood(Likelihood):
    def __init__(self, obs: str, weights: str, offset: str) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame, *args, **kwargs) -> None:
        if not ((data[self.obs] >= 0).all() and (data[self.obs] <= 1).all()):
            raise ValueError("Observations must be in [0, 1].")
        return super().attach(data, *args, **kwargs)

    @property
    def inv_link(self) -> Callable:
        return expit

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["weights"].dot(
            jnp.log(1 + jnp.exp(-y)) + (1 - self.data["obs"]) * y
        )

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        z = jnp.exp(self.get_lin_param(x))
        return self.data["mat"].T @ (
            self.data["weights"] * (z / (1 + z) - self.data["obs"])
        )

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: JAXArray) -> Callable:
        z = jnp.exp(self.get_lin_param(x))
        diag = self.data["weights"] * (z / ((1 + z) ** 2))
        return diag


class GaussianLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    @property
    def inv_link(self) -> Callable:
        return identity

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return 0.5 * self.data["weights"].dot((self.data["obs"] - y) ** 2)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["mat"].T @ (
            self.data["weights"] * (y - self.data["obs"])
        )

    @partial(jax.jit, static_argnums=0)
    def hessian_diag(self, x: JAXArray) -> JAXArray:
        diag = self.data["weights"]
        return diag


class PoissonLikelihood(Likelihood):
    def __init__(
        self, obs: JAXArray, weights: JAXArray, offset: JAXArray
    ) -> None:
        super().__init__(obs, weights, offset)

    def attach(self, data: DataFrame, *args, **kwargs) -> None:
        if not (data[self.obs] >= 0).all():
            raise ValueError("Observations must be non-negative.")
        return super().attach(data, *args, **kwargs)

    @property
    def inv_link(self) -> Callable:
        return jnp.exp

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["weights"].dot(jnp.exp(y) - self.data["obs"] * y)

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        y = self.get_lin_param(x)
        return self.data["mat"].T @ (
            self.data["weights"] * (jnp.exp(y) - self.data["obs"])
        )

    @partial(jax.jit, static_argnums=0)
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
