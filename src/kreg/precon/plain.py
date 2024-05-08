from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.precon.base import PreconBuilder
from kreg.typing import Callable, JAXArray


class PlainPreconBuilder(PreconBuilder):
    def __init__(self, kernel: KroneckerKernel) -> None:
        self.kernel = kernel

    def __call__(self, x: JAXArray) -> Callable:
        return self.kernel.dot
