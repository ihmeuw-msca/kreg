import jax.numpy as jnp

from kreg.precon.base import PreconBuilder
from kreg.term import Term
from kreg.typing import Callable, JAXArray


class PlainPreconBuilder(PreconBuilder):
    def __init__(self, terms: list[Term]) -> None:
        self.terms = terms

    def __call__(self, x: JAXArray) -> Callable:
        def precon_op(x: JAXArray) -> JAXArray:
            start, val = 0, []
            for v in self.terms:
                val.append(v.precon_op(x[start : start + v.size]))
                start += v.size
            return jnp.hstack(val)

        return precon_op
