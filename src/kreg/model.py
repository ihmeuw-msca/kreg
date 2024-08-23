import jax
import jax.numpy as jnp

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon import NystroemPreconBuilder, PlainPreconBuilder, PreconBuilder
from kreg.solver.newton_cg import NewtonCG
from kreg.typing import Callable, DataFrame, JAXArray
from functools import reduce

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

    def objective(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.objective(x)
            + 0.5 * self.lam * x.T @ self.kernel.op_p @ x
        )

    def gradient(self, x: JAXArray) -> JAXArray:
        return self.likelihood.gradient(x) + self.lam * self.kernel.op_p @ x

    def hessian(self, x: JAXArray) -> Callable:
        hess_diag = self.likelihood.hessian_diag(x)

        def op_hess(z: JAXArray) -> JAXArray:
            return hess_diag * z + self.lam * self.kernel.op_p @ z

        return op_hess

    def fit(
        self,
        data: DataFrame,
        x0: JAXArray | None = None,
        gtol: float = 1e-3,
        max_iter: int = 25,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        nystroem_rank: int = 25,
        disable_tqdm=False,
        lam=None,
    ) -> tuple[JAXArray, dict]:
        if lam is not None:
            self.lam = lam
        # attach dataframe
        self.kernel.attach(data)
        data = data.sort_values(self.kernel.names, ignore_index=True)
        self.likelihood.attach(data)

        if x0 is None:
            if self.fitted_result is not None:
                x0 = self.fitted_result
            else:
                x0 = jnp.zeros(len(self.kernel))

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

        result = solver.solve(
            x0,
            max_iter=max_iter,
            gtol=gtol,
            cg_maxiter=cg_maxiter,
            cg_maxiter_increment=cg_maxiter_increment,
            precon_build_freq=10,
            disable_tqdm=disable_tqdm,
        )

        self.likelihood.detach()
        self.kernel.clear_matrices()
        self.fitted_result = result[0]
        self.prev_convergence_data = result[1]
        return result

    def predict(self,new_data,y):
        if self.kernel.matrices_computed is False:
            self.kernel.build_matrices()
        components = self.kernel.kernel_components
        prediction_inputs = [jnp.array(new_data[kc.name].values) for kc in components]
        
        def _predict_single(*single_input):
            return jnp.dot(
                reduce(
                    jnp.kron,
                    [
                        kc.kfunc(jnp.array([x]),kc.grid) 
                        for kc,x in zip(components,*single_input)
                        ]
                        ),self.kernel.op_p@y
                        )
        predict_vec = jax.vmap(jax.jit(_predict_single))
        return predict_vec(prediction_inputs)
