import time
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.sparse.linalg import cg

from kreg.solver.line_search import build_armijo_linesearch
from kreg.solver.opt_logger import Logger
from kreg.typing import Callable, JAXArray


@dataclass
class OptimizationResult:
    """Outcome of a Newton solve.

    ``converged`` is True when the gradient norm reached ``gtol``, or
    reached ``10 * gtol`` after a line-search failure.
    ``convergence_criterion`` names which of those happened, or says that
    the solver did not converge. ``log.to_dataframe()`` has one row per
    iteration: objective value, infinity and 2-norms of the gradient, step
    size and the Armijo ratio of the line search.
    """

    converged: bool
    convergence_criterion: str
    log: Logger


class NewtonCG:
    """Newton's method with conjugate-gradient inner solves.

    Each step solves ``H p = g`` by CG using Hessian-vector products only,
    then back-tracks along ``p`` with an Armijo line search. The CG
    iteration budget grows by ``cg_maxiter_increment`` every Newton step so
    later steps, closer to the optimum, are solved more accurately.

    Parameters
    ----------
    objective, gradient
        Callables of ``x``.
    hessian
        Callable of ``x`` returning the Hessian-vector product function.
    precon_builder
        Callable of ``x`` returning a preconditioner function for CG,
        rebuilt every ``precon_build_freq`` iterations.

    """

    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian: Callable,
        precon_builder: Callable | None = None,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.precon_builder = precon_builder

    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        precon_build_freq: int = 10,
        grad_decrease=0.25,
        verbose=True,
    ) -> tuple[JAXArray, OptimizationResult]:
        """Minimise from ``x0``; ``grad_decrease`` is unused."""
        start = time.time()
        converged = False

        x = x0.copy()
        precon = None
        linesearch_dec = 0.01
        linesearch = jax.jit(
            build_armijo_linesearch(self.objective, slope=0.01)
        )
        opt_logger = Logger(verbose=verbose)

        val, g, hess = self.objective(x), self.gradient(x), self.hessian(x)
        opt_logger.log(
            iter=0,
            obj_val=val,
            gnorm_inf=jnp.max(jnp.abs(g)),
            gnorm2=jnp.linalg.norm(g),
            Δx=0.0,
            step=0.0,
            time=time.time() - start,
            armijo_rat=0.0,
        )

        for i in range(max_iter):
            # update preconditioner
            if self.precon_builder is not None and i % precon_build_freq == 0:
                precon = self.precon_builder(x)

            p, _info = jax.block_until_ready(
                cg(hess, g, M=precon, maxiter=cg_maxiter, tol=1e-16)
            )
            cg_maxiter += cg_maxiter_increment

            x, step, armijo_rat_final = jax.block_until_ready(
                linesearch(x, val, p, g, t0=1.0)
            )

            # Calculate new vals
            val, g, hess = self.objective(x), self.gradient(x), self.hessian(x)

            # Log results
            opt_logger.log(
                iter=i + 1,
                obj_val=val,
                gnorm_inf=jnp.max(jnp.abs(g)),
                gnorm2=jnp.linalg.norm(g),
                Δx=jnp.max(jnp.abs(step * p)),
                step=step,
                time=time.time() - start,
                armijo_rat=armijo_rat_final,
            )

            # Check for failed linesearch
            if armijo_rat_final < linesearch_dec:
                warnings.warn(
                    f"Line search failed on iteration {i}, achieved gnorm = {jnp.linalg.vector_norm(g):.2f}"
                )
                if jnp.linalg.vector_norm(g) <= gtol * 10:
                    converged = True
                    conv_crit = (
                        "approximate_convergence_after_line_search_failure"
                    )
                break

            if jnp.linalg.vector_norm(g) <= gtol:
                converged = True
                conv_crit = "grad_norm"
                break

        if not converged:
            conv_crit = "Did not converge"
            warnings.warn(
                f"Convergence wasn't achieved in {max_iter} iterations"
            )

        convergence_data = OptimizationResult(
            converged=converged, convergence_criterion=conv_crit, log=opt_logger
        )

        return x, convergence_data


class NewtonDirect:
    """Newton's method with a dense Hessian and a Cholesky solve per step.

    Parameters
    ----------
    objective, gradient
        Callables of ``x``.
    hessian_mat
        Callable of ``x`` returning the dense Hessian matrix.

    """

    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian_mat: Callable,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian_mat = hessian_mat

    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        grad_decrease: float = 0.25,
        verbose: bool = True,
    ) -> tuple[JAXArray, OptimizationResult]:
        """Minimise from ``x0``; ``grad_decrease`` is unused."""
        start = time.time()
        converged = False

        x = x0.copy()
        linesearch_dec = 0.01
        linesearch = jax.jit(
            build_armijo_linesearch(self.objective, slope=0.01)
        )

        opt_logger = Logger(verbose=verbose)
        val, g, hess = self.objective(x), self.gradient(x), self.hessian_mat(x)
        opt_logger.log(
            iter=0,
            obj_val=val,
            gnorm_inf=jnp.max(jnp.abs(g)),
            gnorm2=jnp.linalg.norm(g),
            Δx=0.0,
            step=0.0,
            time=time.time() - start,
            armijo_rat=0.0,
        )
        for i in range(max_iter):
            # Check for convergence
            if jnp.linalg.vector_norm(g) <= gtol:
                converged = True
                conv_crit = "grad_norm"
                break

            p = solve(hess, g, assume_a="pos")
            x, step, armijo_rat_final = jax.block_until_ready(
                linesearch(x, val, p, g, t0=1.0)
            )

            val, g, hess = (
                self.objective(x),
                self.gradient(x),
                self.hessian_mat(x),
            )
            opt_logger.log(
                iter=i + 1,
                obj_val=val,
                gnorm_inf=jnp.max(jnp.abs(g)),
                gnorm2=jnp.linalg.norm(g),
                Δx=jnp.max(jnp.abs(step * p)),
                step=step,
                time=time.time() - start,
                armijo_rat=armijo_rat_final,
            )

            if armijo_rat_final < linesearch_dec:
                warnings.warn(
                    f"Line search failed on iteration {i}, achieved gnorm = {jnp.linalg.vector_norm(g):.2f}"
                )
                if jnp.linalg.vector_norm(g) <= gtol * 10:
                    converged = True
                    conv_crit = (
                        "approximate_convergence_after_line_search_failure"
                    )
                break

        if not converged:
            conv_crit = "Did not converge"
            warnings.warn(
                f"Convergence wasn't achieved in {max_iter} iterations"
            )

        convergence_data = OptimizationResult(
            converged=converged, convergence_criterion=conv_crit, log=opt_logger
        )

        return x, convergence_data
