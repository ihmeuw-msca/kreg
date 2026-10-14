from importlib.metadata import version

from kreg.kernel.component import KernelComponent
from kreg.kernel.dimension import Dimension
from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import (
    BinomialLikelihood,
    GaussianLikelihood,
    Likelihood,
    PoissonLikelihood,
)
from kreg.model import KernelRegModel
from kreg.term import Term

__version__ = version("kreg")

__all__ = [
    "BinomialLikelihood",
    "Dimension",
    "GaussianLikelihood",
    "KernelComponent",
    "KernelRegModel",
    "KroneckerKernel",
    "Likelihood",
    "PoissonLikelihood",
    "Term",
    "__version__",
]
