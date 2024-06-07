from functools import partial

import jax
import jax.numpy as jnp

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon import NystroemPreconBuilder, PlainPreconBuilder, PreconBuilder
from kreg.solver.newton_cg import NewtonCG
from kreg.typing import Callable, DataFrame, JAXArray

# TODO: Inexact solve, when to quit
jax.config.update("jax_enable_x64", True)


class KernelRegModel:
    def __init__(
        self,
        kernel: KroneckerKernel,
        likelihood: Likelihood,
        lam: float,
    ) -> None:
        self.kernel = kernel
        self.likelihood = likelihood
        self.lam = lam

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.objective(x)
            + 0.5 * self.lam * x.T @ self.kernel.op_p @ x
        )

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        return self.likelihood.gradient(x) + self.lam * self.kernel.op_p @ x

    def hessian(self, x: JAXArray) -> Callable:
        likli_hess = self.likelihood.hessian(x)

        def op_hess(z: JAXArray) -> JAXArray:
            return likli_hess(z) + self.lam * self.kernel.op_p @ z

        return op_hess

    def fit(
        self,
        data: DataFrame,
        data_span: DataFrame | None = None,
        x0: JAXArray | None = None,
        gtol: float = 1e-3,
        max_iter: int = 25,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        nystroem_rank: int = 25,
    ) -> tuple[JAXArray, dict]:
        # attach dataframe
        data = data.sort_values(self.kernel.names, ignore_index=True)
        self.kernel.attach(data_span or data)
        self.likelihood.attach(data, self.kernel)

        if x0 is None:
            x0 = jnp.zeros(len(self.kernel))

        precon_builder: PreconBuilder
        if nystroem_rank > 0:
            precon_builder = NystroemPreconBuilder(
                self.likelihood, self.kernel, self.lam, nystroem_rank
            )
        else:
            precon_builder = PlainPreconBuilder(self.kernel)

        solver = NewtonCG(
            self.objective,
            self.gradient,
            self.hessian,
            precon_builder,
        )

        result = solver.solve(
            x0,
            max_iter=max_iter,
            gtol=gtol,
            cg_maxiter=cg_maxiter,
            cg_maxiter_increment=cg_maxiter_increment,
            precon_build_freq=10,
        )

        self.likelihood.detach()
        return result
