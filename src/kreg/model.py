import functools

import jax
import jax.numpy as jnp
import numpy as np
from msca.optim.prox import proj_capped_simplex

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon import NystroemPreconBuilder, PlainPreconBuilder, PreconBuilder
from kreg.solver.newton import NewtonCG, NewtonDirect
from kreg.typing import Callable, DataFrame, JAXArray, NDArray, Series

# TODO: Inexact solve, when to quit
jax.config.update("jax_enable_x64", True)


class KernelRegModel:
    def __init__(
        self,
        kernel: KroneckerKernel,
        likelihood: Likelihood,
        lam: float,
        lam_ridge: float = 0.0,
    ) -> None:
        self.kernel = kernel
        self.likelihood = likelihood
        self.lam = lam
        self.lam_ridge = lam_ridge

        self.x: JAXArray
        self.solver_info: dict

    def objective(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.objective(x)
            + 0.5 * self.lam * x.T @ self.kernel.op_p @ x
            + 0.5 * self.lam_ridge * x.T @ x
        )

    def gradient(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.gradient(x)
            + self.lam * self.kernel.op_p @ x
            + self.lam_ridge * x
        )

    def hessian(self, x: JAXArray) -> Callable:
        likli_hess = self.likelihood.hessian(x)

        def op_hess(z: JAXArray) -> JAXArray:
            return (
                likli_hess(z)
                + self.lam * self.kernel.op_p @ z
                + self.lam_ridge * z
            )

        return op_hess

    def hessian_matrix(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.hessian_matrix(x)
            + self.lam * self.kernel.op_p.to_array()
            + self.lam_ridge * jnp.eye(len(x))
        )

    def attach(
        self,
        data: DataFrame,
        data_span: DataFrame | None = None,
        density: Series | None = None,
        train: bool = True,
    ) -> None:
        self.kernel.attach(data if data_span is None else data_span)
        self.likelihood.attach(data, self.kernel, train=train, density=density)

    def detach(self) -> None:
        self.likelihood.detach()
        self.kernel.clear_matrices()

    def fit(
        self,
        data: DataFrame | None = None,
        data_span: DataFrame | None = None,
        density: Series | None = None,
        x0: JAXArray | None = None,
        gtol: float = 1e-3,
        max_iter: int = 25,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        nystroem_rank: int = 25,
        disable_tqdm: bool = False,
        lam: float | None = None,
        detach: bool = True,
        use_direct=False,
        grad_decrease=0.5,
    ) -> tuple[JAXArray, dict]:
        if lam is not None:
            self.lam = lam
        # attach dataframe
        if data is not None:
            self.attach(data, data_span=data_span, density=density, train=True)

        if x0 is None:
            if hasattr(self, "x"):
                x0 = self.x
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

        if detach:
            self.detach()
        return self.x, self.solver_info

    def fit_trimming(
        self,
        data: DataFrame,
        data_span: DataFrame | None = None,
        density: Series | None = None,
        trim_steps: int = 10,
        step_size: float = 10.0,
        inlier_pct: float = 0.95,
        solver_options: dict | None = None,
    ) -> tuple[JAXArray, JAXArray]:
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        if solver_options is None:
            solver_options = {}
        solver_options["detach"] = False

        y = self.fit(
            data, data_span=data_span, density=density, **solver_options
        )[0]

        if inlier_pct < 1.0:
            num_inliers = int(inlier_pct * len(data))
            counter = 0
            success = False
            while (counter < trim_steps) and (not success):
                counter += 1
                nll_terms = self.likelihood.nll_terms(y)
                trim_weights = proj_capped_simplex(
                    self.likelihood.data["trim_weights"]
                    - step_size * nll_terms,
                    num_inliers,
                )
                self.likelihood.update_trim_weights(trim_weights)
                solver_options["x0"] = y
                y = self.fit(**solver_options)[0]
                success = all(
                    jnp.isclose(self.likelihood.data["trim_weights"], 0.0)
                    | jnp.isclose(self.likelihood.data["trim_weights"], 1.0)
                )
            if not success:
                trim_weights = self.likelihood.data["trim_weights"]
                sort_indices = jnp.argsort(trim_weights)
                trim_weights = trim_weights.at[sort_indices[-num_inliers:]].set(
                    1.0
                )
                trim_weights = trim_weights.at[sort_indices[:-num_inliers]].set(
                    0.0
                )
                self.likelihood.data["trim_weights"] = trim_weights

        trim_weights = self.likelihood.data["trim_weights"]
        self.detach()

        return y, trim_weights

    def predict(
        self, data, x: NDArray | None = None, from_kernel: bool = False
    ) -> NDArray:
        x = self.x if x is None else x
        if from_kernel:
            self.kernel.attach(data)
            kernel_components = self.kernel.kernel_components
            rows = [
                jnp.asarray(data[kc.columns].to_numpy())
                for kc in kernel_components
            ]
            inv_k_x = self.kernel.op_p @ x
            def predict_row(*row):
                k_new_x = functools.reduce(
                    jnp.kron,
                    [
                        kc.kfunc(jnp.asarray([coords]), kc.span)
                        for kc, coords in zip(kernel_components, row)
                    ],
                )
                return jnp.dot(k_new_x, inv_k_x)

            predict_rows = jax.vmap(jax.jit(predict_row))
            y = predict_rows(*rows)
            offset = data[self.likelihood.offset].to_numpy()
            print(offset.shape)
            print(y.shape)
            pred = self.likelihood.inv_link(offset + y.ravel())
            print(pred.shape)
            self.kernel.clear_matrices()
        else:
            self.attach(data, train=False)
            pred = self.likelihood.get_param(x)
            self.detach()
        import pdb; pdb.set_trace()
        return np.asarray(pred)
