import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from pykronecker import KroneckerProduct
from pykronecker import KroneckerDiag
from tqdm.auto import tqdm
from functools import partial
from jax.scipy.sparse.linalg import cg
from numpy.typing import ArrayLike
import numpy as np
from functools import reduce

#TODO: Inexact solve, when to quit

def cartesian_prod(x:jax.Array,y:jax.Array):
    """
    Computes Cartesian product of two arrays x,y
    """
    a,b= jnp.meshgrid(y,x)
    full_X = jnp.vstack([b.flatten(),a.flatten()]).T
    return full_X

def outer_fold(x: ArrayLike, y: ArrayLike) -> jax.Array:
    """Compute the outer product of two arrays 
    in a way that's amenable to functional reduction operations

    Parameters
    ----------
    x
        First array.
    y
        Second array.

    Returns
    -------
    jax.Array
        Outer product of the two arrays.

    """
    return jnp.array(np.multiply.outer(np.array(x), np.array(y)))

def randomized_nystrom(
    vmapped_A: Callable, input_shape: int, rank: int, key: int
):
    """Create a randomized Nystrom approximation of a matrix.

    Parameters
    ----------
    vmapped_A
        Function that computes the matrix-vector product of the matrix to be approximated.
    input_shape
        Number of rows of the matrix.
    rank
        Rank of the approximation.
    key
        Random key.

    Returns
    -------
    tuple[Array, Array]
        Approximation matrices U and E.

    """
    X = jax.random.normal(key, (input_shape, rank))
    Q = jnp.linalg.qr(X).Q
    AQ = vmapped_A(Q)
    eps = 1e-8 * jnp.linalg.norm(AQ, ord="fro")
    C = jnp.linalg.cholesky(Q.T @ AQ + eps * jnp.identity(rank))
    B = jax.scipy.linalg.solve_triangular(C, AQ.T, lower=True).T
    U, S = jax.scipy.linalg.svd(B, full_matrices=False)[:2]
    E = jnp.maximum(0, S**2 - eps * jnp.ones(rank))
    return U, E

def build_ny_precon(U,E,lam):
    def precon(x):
        inner_diag = 1/(E+lam)- (1/lam)
        return (1/lam) *x + U@(inner_diag * (U.T@x))
    return precon


class KroneckerKernel:
    """Kronecker product of all kernel functions to form a complete kernel
    linear mapping.

    Parameters
    ----------
    kernels
        List of kernel functions.
    value_grids
        List of value grids, unique values for each dimension.
    nugget
        Regularization for the kernel matrix.

    """

    def __init__(
        self,
        kernels: list[Callable],
        value_grids: list[jax.Array],
        nugget: float = 5e-8,
    ) -> None:
        """
        TODO: Abstract this to lists of kernels and grids, kronecker out sex, age and time
        """
        self.kmats = [k(x,x) for k,x in zip(kernels,value_grids)]
        if len(self.kmats)==1:
            nugget_vals = jnp.ones([len(m) for m in self.kmats]).reshape(-1,1)
        else:
            nugget_vals = jnp.ones([len(m) for m in self.kmats])

        self.K = KroneckerProduct(self.kmats)+ nugget * KroneckerDiag(nugget_vals)
        eigvals,eigvecs = zip(*[jnp.linalg.eigh(Ki) for Ki in self.kmats])
        self.left = KroneckerProduct(eigvecs)
        self.right = self.left.T
        self.kronvals = reduce(outer_fold,eigvals) + nugget
        self.P = self.left@(KroneckerDiag(1/self.kronvals))@self.right

        self.left_etimes_left = KroneckerProduct([e*e for e in eigvecs])
        self.shapes = [len(grid) for grid in value_grids]

        self.rootK = (
            self.left @ (KroneckerDiag(jnp.sqrt(self.kronvals))) @ self.right
        )
        self.rootP = (
            self.left
            @ (KroneckerDiag(1 / jnp.sqrt(self.kronvals)))
            @ self.right
        )

    def get_preconditioners(
        self, lam: float, beta: float
    ) -> tuple[KroneckerOperator, KroneckerOperator]:
        PC = (
            self.left
            @ KroneckerDiag(
                jnp.sqrt(self.kronvals / (lam + beta * self.kronvals))
            )
            @ self.right
        )
        PC_inv = (
            self.left
            @ KroneckerDiag(
                1 / jnp.sqrt(self.kronvals / (lam + beta * self.kronvals))
            )
            @ self.right
        )
        return PC, PC_inv
    
    def get_M(self, lam: float, beta: float) -> KroneckerOperator:
        middle = self.kronvals / (lam + beta * self.kronvals)
        return self.left @ KroneckerDiag(middle) @ self.right

class LogisticLikelihood:
    def __init__(
        self,
        obs_counts: jax.Array,
        sample_sizes: jax.Array,
    ) -> None:
        assert (obs_counts<=sample_sizes).all()
        self.sample_sizes = sample_sizes
        self.obs_counts = obs_counts
        self.beta_smoothness = jnp.max(sample_sizes)/4
        self.N = len(obs_counts)
        

    def loss_single(y: jax.Array, k: jax.Array, n: jax.Array) -> jax.Array:
        return n * jnp.log(1 + jnp.exp(-y)) + (n - k) * y

    @partial(jax.jit, static_argnums=0)
    def f(self, y: jax.Array) -> jax.Array:
        return jnp.sum(
            LogisticLikelihood.loss_single(
                y, self.obs_counts, self.sample_sizes
            )
        )

    grad_f = jax.jit(jax.grad(f,argnums=1),static_argnums = 0)
    val_grad_f = jax.jit(jax.value_and_grad(f,argnums=1),static_argnums = 0)

    @partial(jax.jit,static_argnums = 0)
    def H_diag(self,y):
        z = jnp.exp(y)
        return self.sample_sizes * (z/((z+1)**2))
    
class KernelRegModel():
    def __init__(
            self,
            kernel,
            likelihood,
            lam,
            offset,
            ) -> None:
        self.kernel = kernel
        self.likelihood = likelihood
        self.lam = lam
        self.offset = offset
    
    @partial(jax.jit, static_argnums=0)
    def reg_term(self, y: jax.Array) -> jax.Array:
        return self.lam * y.T @ self.kernel.P @ y / 2

    @partial(jax.jit, static_argnums=0)
    def full_loss(self, y: jax.Array) -> jax.Array:
        return self.likelihood.f(y + self.offset) + self.reg_term(y)

    grad_loss = jax.jit(jax.grad(full_loss, argnums=1), static_argnums=0)
    val_grad_loss = jax.jit(
        jax.value_and_grad(full_loss, argnums=1), static_argnums=0
    )

    def D(self, y: jax.Array) -> jax.Array:
        return self.likelihood.H_diag(y + self.offset)

    def H(self, y: jax.Array) -> Callable:
        Hd = self.D(y)
        P_part = self.lam * self.kernel.P
        def H_apply(x):
            return Hd*x + P_part@x
        return H_apply

    def compute_nystroem(
        self, D: jax.Array, key: int, rank: int = 50
    ) -> Callable:
        rootK = self.kernel.rootK
        root_KDK =jax.vmap(
            lambda x:rootK@(D*(rootK@x)),
            in_axes = 1,
            out_axes = 1
            )
        U, E = randomized_nystrom(root_KDK, self.likelihood.N, rank, key)
        return U,E
    
    @partial(jax.jit,static_argnames = 'self')
    def nys_pc_newton_step(
        self,
        y:jax.Array,
        g:jax.Array,
        U:jax.Array,
        E:jax.Array,
        maxiter:int,
        ):
        ny_PC = build_ny_precon(U,E,self.lam)
        def full_PC(x):
            return self.kernel.rootK@(ny_PC(self.kernel.rootK@x))
        H = self.H(y)
        step,info = cg(H,g,M = full_PC,maxiter = maxiter, tol = 1e-16)
        return step,info
    
    @partial(jax.jit,static_argnames = 'self')
    def K_pc_newton_step(
        self,
        y:jax.Array,
        g:jax.Array,
        maxiter:int,
        ):
        H = self.H(y)
        M = lambda x:self.kernel.K@x
        step,info = cg(H,g,M = M,maxiter = maxiter, tol = 1e-16)
        return step,info
    
    def optimize(
        self,
        y0: jax.Array | None = None,
        max_newton_cg: int = 25,
        grad_tol: float = 1e-3,
        max_cg_iter: int = 100,
        scaling_cg_iter: int = 25,
        nystrom_rank: int = 25,
    ) -> tuple[jax.Array, dict]:
        """
        """
        rng_key = jax.random.PRNGKey(101)
        rng_key,*split_keys = jax.random.split(rng_key,2*max_newton_cg)

        
        if y0 is None:
            y0 = jnp.zeros(len(self.kernel.K))

        y = y0.copy()        

        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        converged = False
        M = lambda x:self.kernel.K@x
        for i in tqdm(range(max_newton_cg)):
            val,g = self.val_grad_loss(y)

            #Check for convergence
            if (jnp.linalg.vector_norm(g)<=grad_tol):
                converged = True
                conv_crit = "grad_norm"
                break
            # elif i>1 and val>=min(loss_vals)-1e-12:
            #     converged = True
            #     conv_crit = "loss_val"
            #     break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))

            #M = self.kernel.get_M(self.lam,beta)
            num_cg_iter = max_cg_iter + scaling_cg_iter * i
            if (nystroem_rank>0 and i%10==0):
                D = self.D(y)
                U,E = self.compute_nystroem(D,split_keys[i],rank = nystroem_rank)
            if nystroem_rank>0:
                step,info = self.nys_pc_newton_step(y,g,U,E,num_cg_iter)
            else:
                step,info = self.K_pc_newton_step(y,g,maxiter=num_cg_iter)

            newton_decrements.append(jnp.sqrt(jnp.dot(g,step)))
            #Hard coded line search
            step_size = 1.
            alpha = 0.1
            new_y = y-step_size*step
            new_val = self.full_loss(new_y)
            while new_val-val>=step_size * alpha * jnp.dot(g,step) or jnp.isnan(new_val):
                step_size = 0.2 * step_size
                new_y = y - step_size * step
                new_val = self.full_loss(new_y)
            
            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step_size * step)))
            y = new_y
        if not converged:
            conv_crit = "Did not converge"
            print(f"Convergence wasn't achieved in {max_newton_cg} iterations")

        val, g = self.val_grad_loss(y)
        loss_vals.append(val)
        grad_norms.append(jnp.linalg.vector_norm(g))

        loss_vals=jnp.array(loss_vals)
        grad_norms = jnp.array(grad_norms)
        newton_decrements = jnp.array(newton_decrements)
        iterate_maxnorm_distances = jnp.array(iterate_maxnorm_distances)
        convergence_data = {
            "loss_vals": loss_vals,
            "gnorms": grad_norms,
            "converged": converged,
            "convergence_criterion": conv_crit,
            "newton_decrements": newton_decrements,
            "iterate_maxnorm_distances": iterate_maxnorm_distances
        }
        return y, convergence_data

def run_solver(solver,x0):
    """
    Wraps a jaxopt solver to track convergence data
    """
    state = solver.init_state(x0)
    sol = x0
    values,errors,stepsizes = [state.value],[state.error],[state.stepsize]
    update = lambda sol,state:solver.update(sol,state)
    jitted_update = jax.jit(update)
    for _ in tqdm(range(solver.maxiter)):
        sol,state = jitted_update(sol,state)
        values.append(state.value)
        errors.append(state.error)
        stepsizes.append(state.stepsize)
        if solver.verbose > 0:
            print("Gradient Norm: ",state.error)
            print("Loss Value: ",state.value)
        if state.error<=solver.tol:
            break
        if stepsizes[-1]==0:
            state = solver.init_state(sol)
    convergence_data = {
        "values":np.array(values),
        "gradnorms":np.array(errors),
        "stepsizes":np.array(stepsizes)
    }
    return sol,convergence_data,state
