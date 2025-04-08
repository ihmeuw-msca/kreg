import functools

import jax
import jax.numpy as jnp
import numpy as np
from msca.optim.prox import proj_capped_simplex

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon import NystroemPreconBuilder, PlainPreconBuilder, PreconBuilder
from kreg.solver.newton import NewtonCG, NewtonDirect
from kreg.typing import Callable, DataFrame, JAXArray, NDArray, Series, Optional, Union
from kreg.utils import memory_profiled, logger

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

    @memory_profiled
    def attach(
        self,
        data: DataFrame,
        data_span: Optional[DataFrame] = None,
        density: Optional[Series] = None,
        train: bool = True,
    ) -> None:
        logger.debug(f"Attaching data with shape {data.shape}, train={train}")
        self.kernel.attach(data if data_span is None else data_span)
        self.likelihood.attach(data, self.kernel, train=train, density=density)

    @memory_profiled
    def detach(self) -> None:
        logger.debug("Detaching model data")
        self.likelihood.detach()
        self.kernel.clear_matrices()

    @memory_profiled
    def fit(
        self,
        data: Optional[DataFrame] = None,
        data_span: Optional[DataFrame] = None,
        density: Optional[Series] = None,
        x0: Optional[JAXArray] = None,
        gtol: float = 1e-3,
        max_iter: int = 25,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        nystroem_rank: int = 25,
        disable_tqdm: bool = False,
        lam: Optional[float] = None,
        detach: bool = True,
        use_direct=False,
        grad_decrease=0.5,
    ) -> "tuple[JAXArray, dict]":
        if lam is not None:
            self.lam = lam
        # attach dataframe
        if data is not None:
            logger.info(f"Fitting model with data shape: {data.shape}")
            if data_span is not None:
                logger.info(f"Using data_span with shape: {data_span.shape}")
            self.attach(data, data_span=data_span, density=density, train=True)
        else:
            logger.info("Fitting model with pre-attached data")

        if x0 is None:
            if hasattr(self, "x"):
                x0 = self.x
            else:
                x0 = jnp.zeros(len(self.kernel))
                logger.debug(f"Initialized x0 with zeros, shape: {x0.shape}")

        solver: Union[NewtonDirect, NewtonCG]
        if use_direct:
            logger.info("Using direct Newton solver")
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
            logger.info(f"Using Newton-CG solver with nystroem_rank={nystroem_rank}")
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
        
        logger.debug(f"Fit complete, converged: {self.solver_info.get('converged', False)}")

        if detach:
            self.detach()
        return self.x, self.solver_info

    @memory_profiled
    def fit_trimming(
        self,
        data: DataFrame,
        data_span: Optional[DataFrame] = None,
        density: Optional[Series] = None,
        trim_steps: int = 10,
        step_size: float = 10.0,
        inlier_pct: float = 0.95,
        solver_options: Optional[dict] = None,
    ) -> "tuple[JAXArray, JAXArray]":
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        if solver_options is None:
            solver_options = {}
        solver_options["detach"] = False

        logger.info(f"Fitting with trimming: steps={trim_steps}, inlier_pct={inlier_pct}")
        y = self.fit(
            data, data_span=data_span, density=density, **solver_options
        )[0]

        if inlier_pct < 1.0:
            num_inliers = int(inlier_pct * len(data))
            counter = 0
            success = False
            while (counter < trim_steps) and (not success):
                counter += 1
                logger.debug(f"Trimming step {counter}/{trim_steps}")
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
                logger.info(f"Trimming did not converge after {trim_steps} steps, forcing final trimming")
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
        logger.info(f"Trimming complete, kept {int(jnp.sum(trim_weights))} of {len(trim_weights)} observations")

        return y, trim_weights

    @memory_profiled
    def predict(
        self, data, x: Optional[NDArray] = None, from_kernel: bool = False
    ) -> NDArray:
        logger.debug(f"Making predictions with data shape {data.shape}, from_kernel={from_kernel}")
        x = self.x if x is None else x
        if from_kernel:
            self.kernel.attach(data)
            kernel_components = self.kernel.kernel_components
            rows = [
                jnp.asarray(data[kc.dim_names].to_numpy())
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
            pred = self.likelihood.inv_link(
                data[self.likelihood.offset].to_numpy()
                + predict_rows(*rows).ravel()
            )
        else:
            self.attach(data, train=False)
            pred = self.likelihood.get_param(x)
            self.detach()
        logger.debug(f"Predictions complete, shape: {pred.shape}")
        return np.asarray(pred)
