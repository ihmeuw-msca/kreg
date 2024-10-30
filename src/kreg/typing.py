from typing import Callable

from jax import Array as JAXArray
from numpy.typing import NDArray
from pandas import DataFrame

KernelFunction = Callable[[JAXArray, JAXArray], JAXArray]

__all__ = [
    "Callable",
    "NDArray",
    "DataFrame",
    "JAXArray",
    "KernelFunction",
]
