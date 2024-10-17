from typing import Callable

from jax import Array as JAXArray
from numpy.typing import NDArray
from pandas import DataFrame

KernelFunction = Callable[[JAXArray, JAXArray], JAXArray]
DimensionColumns = str | tuple[str, str]

__all__ = [
    Callable,
    NDArray,
    DataFrame,
    JAXArray,
    KernelFunction,
    DimensionColumns,
]
