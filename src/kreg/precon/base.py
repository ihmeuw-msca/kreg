from collections.abc import Callable
from typing import Protocol

from kreg.typing import JAXArray


class PreconBuilder(Protocol):
    def __call__(self, x: JAXArray) -> Callable:
        """Build a preconditioner."""
