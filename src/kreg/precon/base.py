from typing import Callable, Protocol

from kreg.typing import JAXArray


class PreconBuilder(Protocol):
    def __call__(self, x: JAXArray) -> Callable:
        """Build a preconditioner."""
