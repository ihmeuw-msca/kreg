from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

import pandas as pd
from pandas import DataFrame, Series

from jax import Array as JAXArray
from numpy.typing import NDArray

KernelFunction = Callable[[JAXArray, JAXArray], JAXArray]

# Additional type variables for generic functions
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

__all__ = [
    "Callable",
    "NDArray",
    "DataFrame",
    "Series",
    "JAXArray",
    "KernelFunction",
]
