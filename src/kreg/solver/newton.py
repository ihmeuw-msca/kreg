import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.sparse.linalg import cg
from tqdm.auto import tqdm
import warnings

from kreg.solver.line_search import armijo_line_search
from kreg.typing import Callable, JAXArray, Optional
from kreg.utils import memory_profiled, logger


class NewtonCG:
    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian: Callable,
        precon_builder: Optional[Callable] = None,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.precon_builder = precon_builder

    @memory_profiled
    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        precon_build_freq: int = 10,
        disable_tqdm=False,
        grad_decrease=0.25,
    ) -> "tuple[JAXArray, dict]":
        logger.info(f"Starting Newton-CG solver with max_iter={max_iter}, gtol={gtol}, cg_maxiter={cg_maxiter}")
        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        converged = False

        x = x0.copy()
        precon = None

        for i in tqdm(range(max_iter), disable=disable_tqdm):
            val, g, hess = self.objective(x), self.gradient(x), self.hessian(x)

            # Check for convergence
            if jnp.linalg.vector_norm(g) <= gtol:
                logger.info(f"Converged at iteration {i}: gradient norm = {jnp.linalg.vector_norm(g):.6f} <= {gtol}")
                converged = True
                conv_crit = "grad_norm"
                break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))
            logger.debug(f"Iteration {i}: loss = {val:.6f}, |grad| = {jnp.linalg.vector_norm(g):.6f}")

            # update preconditioner
            if self.precon_builder is not None and i % precon_build_freq == 0:
                logger.debug(f"Building preconditioner at iteration {i}")
                precon = self.precon_builder(x)

            cg_maxiter += cg_maxiter_increment
            logger.debug(f"Running CG with maxiter={cg_maxiter}")
            p, info = cg(hess, g, M=precon, maxiter=cg_maxiter, tol=1e-16)
            logger.debug(f"CG completed with info={info}")

            newton_decrements.append(jnp.sqrt(jnp.dot(g, p)))
            # Hard coded line search
            try:
                step, armijo_ratio, gradnorm_ratio = armijo_line_search(
                    x,
                    p,
                    g,
                    self.objective,
                    self.gradient,
                    grad_decrease=grad_decrease,
                )
            except RuntimeError as e:
                warnings.warn(
                    f"Line search failed on iteration {i}, achieved gnorm = {jnp.linalg.vector_norm(g)}: {str(e)}"
                )
                if jnp.linalg.vector_norm(g) <= gtol * 10:
                    converged = True
                    conv_crit = (
                        "approximate_convergence_after_line_search_failure"
                    )
                    break
                else:
                    break

            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step * p)))
            x = x - step * p
        if not converged:
            conv_crit = "Did not converge"
            warnings.warn(
                f"Convergence wasn't achieved in {max_iter} iterations"
            )

        val, g = self.objective(x), self.gradient(x)
        loss_vals.append(val)
        grad_norms.append(jnp.linalg.vector_norm(g))
        logger.info(f"Final loss = {val:.6f}, |grad| = {jnp.linalg.vector_norm(g):.6f}")

        loss_vals = jnp.array(loss_vals)
        grad_norms = jnp.array(grad_norms)
        newton_decrements = jnp.array(newton_decrements)
        iterate_maxnorm_distances = jnp.array(iterate_maxnorm_distances)
        convergence_data = {
            "loss_vals": loss_vals,
            "gnorms": grad_norms,
            "converged": converged,
            "convergence_criterion": conv_crit,
            "newton_decrements": newton_decrements,
            "iterate_maxnorm_distances": iterate_maxnorm_distances,
        }

        return x, convergence_data


class NewtonDirect:
    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian_mat: Callable,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian_mat = hessian_mat

    @memory_profiled
    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        disable_tqdm=False,
        grad_decrease=0.25,
    ) -> "tuple[JAXArray, dict]":
        logger.info(f"Starting Newton-Direct solver with max_iter={max_iter}, gtol={gtol}")
        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        armijo_ratios = []
        gradnorm_ratios = []
        stepsizes = []
        converged = False

        x = x0.copy()

        for i in tqdm(range(max_iter), disable=disable_tqdm):
            val, g, hess = (
                self.objective(x),
                self.gradient(x),
                self.hessian_mat(x),
            )

            # Check for convergence
            if jnp.linalg.vector_norm(g) <= gtol:
                logger.info(f"Converged at iteration {i}: gradient norm = {jnp.linalg.vector_norm(g):.6f} <= {gtol}")
                converged = True
                conv_crit = "grad_norm"
                break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))
            logger.debug(f"Iteration {i}: loss = {val:.6f}, |grad| = {jnp.linalg.vector_norm(g):.6f}")

            # Log the Hessian dimensions and condition number for memory profiling
            logger.debug(f"Hessian shape: {hess.shape}")
            
            p = solve(hess, g, assume_a="pos")

            newton_decrements.append(jnp.sqrt(jnp.dot(g, p)))
            # Hard coded line search
            step, armijo_ratio, gradnorm_ratio = armijo_line_search(
                x,
                p,
                g,
                self.objective,
                self.gradient,
                grad_decrease=grad_decrease,
            )
            logger.debug(f"Line search: step={step:.6f}, armijo_ratio={armijo_ratio:.6f}")
            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step * p)))
            armijo_ratios.append(armijo_ratio)
            gradnorm_ratios.append(gradnorm_ratio)
            stepsizes.append(step)
            x = x - step * p
        if not converged:
            conv_crit = "Did not converge"
            logger.warning(f"Convergence wasn't achieved in {max_iter} iterations")

        val, g = self.objective(x), self.gradient(x)
        loss_vals.append(val)
        grad_norms.append(jnp.linalg.vector_norm(g))
        logger.info(f"Final loss = {val:.6f}, |grad| = {jnp.linalg.vector_norm(g):.6f}")

        loss_vals = jnp.array(loss_vals)
        grad_norms = jnp.array(grad_norms)
        newton_decrements = jnp.array(newton_decrements)
        iterate_maxnorm_distances = jnp.array(iterate_maxnorm_distances)
        armijo_ratios = jnp.array(armijo_ratios)
        gradnorm_ratios = jnp.array(gradnorm_ratios)
        stepsizes = jnp.array(stepsizes)
        convergence_data = {
            "loss_vals": loss_vals,
            "gnorms": grad_norms,
            "converged": converged,
            "convergence_criterion": conv_crit,
            "newton_decrements": newton_decrements,
            "iterate_maxnorm_distances": iterate_maxnorm_distances,
            "armijo_ratios": armijo_ratios,
            "gradnorm_ratios": gradnorm_ratios,
            "stepsizes": stepsizes,
        }

        return x, convergence_data
