from typing import Callable

from jax import Array as JAXArray

Kernel = Callable[[JAXArray, JAXArray], JAXArray]

__all__ = [
    Callable,
    JAXArray,
]
