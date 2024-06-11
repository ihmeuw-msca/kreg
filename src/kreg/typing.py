from typing import Callable

from jax import Array as JAXArray
from pandas import DataFrame

KernelFunction = Callable[[JAXArray, JAXArray], JAXArray]

__all__ = [
    Callable,
    DataFrame,
    JAXArray,
    KernelFunction,
]
