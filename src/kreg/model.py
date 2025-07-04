import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag
from msca.optim.prox import proj_capped_simplex

from kreg.likelihood import Likelihood
from kreg.precon import PlainPreconBuilder
from kreg.solver.newton import NewtonCG, NewtonDirect, OptimizationResult
from kreg.term import Term
from kreg.typing import Callable, DataFrame, JAXArray, NDArray, Series

# TODO: Inexact solve, when to quit
jax.config.update("jax_enable_x64", True)


class KernelRegModel:
    def __init__(self, terms: list[Term], likelihood: Likelihood) -> None:
        """Kernel regression model

        Parameters
        ----------
        terms
            The regression terms for the kernel regression
        likelihood
            The likelihood model to use

        """
        self.terms = terms
        self.likelihood = likelihood

        self.x: JAXArray
        self.solver_info: OptimizationResult

    def objective_prior(self, x: JAXArray) -> JAXArray:
        start, val = 0, 0.0
        for v in self.terms:
            val += v.objective(x[start : start + v.size])
            start += v.size
        return val

    def gradient_prior(self, x: JAXArray) -> JAXArray:
        start, val = 0, []
        for v in self.terms:
            val.append(v.gradient(x[start : start + v.size]))
            start += v.size
        return jnp.hstack(val)

    def hessian_prior_op(self, x: JAXArray) -> JAXArray:
        start, val = 0, []
        for v in self.terms:
            val.append(v.hessian_op(x[start : start + v.size]))
            start += v.size
        return jnp.hstack(val)

    @functools.cache
    def hessian_prior_matrix(self) -> JAXArray:
        mats = [v.hessian_matrix() for v in self.terms]
        return block_diag(*mats)

    def objective(self, x: JAXArray) -> JAXArray:
        return self.likelihood.objective(x) + self.objective_prior(x)

    def gradient(self, x: JAXArray) -> JAXArray:
        return self.likelihood.gradient(x) + self.gradient_prior(x)

    def hessian(self, x: JAXArray) -> Callable:
        """
        Return a function that computes the Hessian-vector product.

        Parameters
        ----------
        x : JAXArray
            The parameter vector

        Returns
        -------
        Callable
            A function that computes the Hessian-vector product

        """
        likelihood_hessian = self.likelihood.hessian(x)

        def hessian_vector_product(z: JAXArray) -> JAXArray:
            """Compute the Hessian-vector product."""
            return likelihood_hessian(z) + self.hessian_prior_op(z)

        return hessian_vector_product

    def hessian_matrix(self, x: JAXArray) -> JAXArray:
        return self.likelihood.hessian_matrix(x) + self.hessian_prior_matrix()

    def attach(
        self,
        data: DataFrame,
        data_span: DataFrame | None = None,
        density: Series | None = None,
        train: bool = True,
    ) -> None:
        """
        Attach data to the model for computation.

        Parameters
        ----------
        data : DataFrame
            The data for the likelihood
        data_span : DataFrame, optional
            The data for the kernel span, by default None
        density : Series, optional
            The density weights, by default None
        train : bool, optional
            Whether in training mode, by default True

        """
        for v in self.terms:
            v.attach(data if data_span is None else data_span)
        self.likelihood.attach(data, self.terms, train=train, density=density)

    def detach(self) -> None:
        """Release resources by detaching data from the model."""
        self.likelihood.detach()
        for v in self.terms:
            v.clear_matrices()

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
        nystroem_rank: int = 0,
        lam: float | dict[str, float] | None = None,
        detach: bool = True,
        use_direct: bool = False,
        grad_decrease: float = 0.5,
        verbose: bool = True,
    ) -> tuple[JAXArray, OptimizationResult]:
        """
        Fit the model to data.

        Parameters
        ----------
        data
            The data to fit, by default None
        data_span
            The data for the kernel span, by default None
        density
            The density weights, by default None
        x0
            Initial parameter vector, by default None
        gtol
            Gradient tolerance for convergence, by default 1e-3
        max_iter
            Maximum number of iterations, by default 25
        cg_maxiter
            Maximum CG iterations, by default 100
        cg_maxiter_increment
            CG iteration increment per Newton step, by default 25
        nystroem_rank
            Rank for Nyström approximation, by default 25
        lam
            Override regularization parameter, by default None
        detach
            Whether to detach data after fitting, by default True
        use_direct
            Whether to use direct solver instead of CG, by default False
        grad_decrease
            Required gradient decrease factor, by default 0.5
        verbose
            If True print iteration information.

        Returns
        -------
        tuple[JAXArray, OptimizationResult]
            The fitted parameter vector and solver information

        """
        if lam is not None:
            if not isinstance(lam, dict):
                lam = {v.label: lam for v in self.terms}
            for v in self.terms:
                if v.label in lam:
                    v.lam = lam[v.label]

        # attach dataframe
        if data is not None:
            self.attach(data, data_span=data_span, density=density, train=True)

        if x0 is None:
            if hasattr(self, "x"):
                x0 = self.x
            else:
                size = sum([v.size for v in self.terms])
                x0 = jnp.zeros(size)

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
                grad_decrease=grad_decrease,
                verbose=verbose,
            )
        else:
            if nystroem_rank > 0:
                raise ValueError(
                    "Do not support preconditioner until further development"
                )
            else:
                precon_builder = PlainPreconBuilder(self.terms)
            solver = NewtonCG(
                jax.jit(self.objective),
                jax.jit(self.gradient),
                self.hessian,
                precon_builder=precon_builder,
            )
            self.x, self.solver_info = solver.solve(
                x0,
                max_iter=max_iter,
                gtol=gtol,
                cg_maxiter=cg_maxiter,
                cg_maxiter_increment=cg_maxiter_increment,
                precon_build_freq=10,
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
        """
        Fit the model with trimming to handle outliers.

        Parameters
        ----------
        data : DataFrame
            The data to fit
        data_span : DataFrame, optional
            The data for the kernel span, by default None
        density : Series, optional
            The density weights, by default None
        trim_steps : int, optional
            Number of trimming steps, by default 10
        step_size : float, optional
            Step size for trimming, by default 10.0
        inlier_pct : float, optional
            Percentage of inliers to keep (0.0-1.0), by default 0.95
        solver_options : dict, optional
            Additional options for the solver, by default None

        Returns
        -------
        tuple[JAXArray, JAXArray]
            The fitted parameter vector and trimming weights
        """
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
        self,
        data: DataFrame,
        x: NDArray | None = None,
        from_kernel: bool = False,
    ) -> NDArray:
        """
        Make predictions for new data.

        Parameters
        ----------
        data : DataFrame
            The data to predict for
        x : NDArray, optional
            Parameter vector to use for prediction, by default None
        from_kernel : bool, optional
            Whether to use kernel matrix for prediction, by default False

        Returns
        -------
        NDArray
            The predictions
        """
        x = self.x if x is None else x
        if from_kernel:
            raise ValueError(
                "from_kernel option is not supported until further development"
            )
        else:
            self.attach(data, train=False)
            pred = self.likelihood.get_param(x)
            self.detach()
        return np.asarray(pred)
