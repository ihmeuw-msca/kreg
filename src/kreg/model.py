from functools import reduce

import jax
import jax.numpy as jnp

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon import NystroemPreconBuilder, PlainPreconBuilder, PreconBuilder
from kreg.solver.newton import NewtonCG, NewtonDirect
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
        self.fitted_result = None
        self.x: JAXArray
        self.solver_info: dict

    def objective(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.objective(x)
            + 0.5 * self.lam * x.T @ self.kernel.op_p @ x
        )

    def gradient(self, x: JAXArray) -> JAXArray:
        return self.likelihood.gradient(x) + self.lam * self.kernel.op_p @ x

    def hessian(self, x: JAXArray) -> Callable:
        likli_hess = self.likelihood.hessian(x)

        def op_hess(z: JAXArray) -> JAXArray:
            return likli_hess(z) + self.lam * self.kernel.op_p @ z

        return op_hess

    def hessian_matrix(self, x: JAXArray) -> JAXArray:
        return (
            jnp.diag(self.likelihood.hessian_diag(x))
            + self.lam * self.kernel.op_p.to_array()
        )

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
        disable_tqdm=False,
        lam=None,
        use_direct=False,
        grad_decrease=0.5,
    ) -> tuple[JAXArray, dict]:
        if lam is not None:
            self.lam = lam
        # attach dataframe
        self.kernel.attach(data_span or data)
        self.likelihood.attach(data, self.kernel)

        if x0 is None:
            if self.fitted_result is not None:
                x0 = self.fitted_result
            else:
                x0 = jnp.zeros(len(self.kernel))

        solver: NewtonDirect | NewtonCG
        if use_direct:
            solver = NewtonDirect(
                jax.jit(self.objective),
                jax.jit(self.gradient),
                self.hessian_matrix,
            )
            self.x, self.solver_info = solver.solve(
                x0,
                max_iter=max_iter,
                gtol=gtol,
                disable_tqdm=disable_tqdm,
                grad_decrease=grad_decrease,
            )
        else:
            precon_builder: PreconBuilder
            if nystroem_rank > 0:
                precon_builder = NystroemPreconBuilder(
                    self.likelihood, self.kernel, self.lam, nystroem_rank
                )
            else:
                precon_builder = PlainPreconBuilder(self.kernel)
            solver = NewtonCG(
                jax.jit(self.objective),
                jax.jit(self.gradient),
                self.hessian,
                precon_builder,
            )
            self.x, self.solver_info = solver.solve(
                x0,
                max_iter=max_iter,
                gtol=gtol,
                cg_maxiter=cg_maxiter,
                cg_maxiter_increment=cg_maxiter_increment,
                precon_build_freq=10,
                disable_tqdm=disable_tqdm,
                grad_decrease=grad_decrease,
            )

        self.likelihood.detach()
        self.kernel.clear_matrices()
        return self.x, self.solver_info

    def predict(self, data, y):
        if self.kernel.matrices_computed is False:
            self.kernel.build_matrices()
        self.likelihood.attach(data, self.kernel, train=False)
        components = self.kernel.kernel_components
        prediction_inputs = [
            jnp.array(data[kc.name].values) for kc in components
        ]

        def _predict_single(*single_input):
            return jnp.dot(
                reduce(
                    jnp.kron,
                    [
                        kc.kfunc(jnp.array([x]), kc.grid)
                        for kc, x in zip(components, *single_input)
                    ],
                ),
                self.kernel.op_p @ y,
            )

        predict_vec = jax.vmap(jax.jit(_predict_single))
        return predict_vec(prediction_inputs)
