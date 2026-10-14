# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `kreg.kernel.kron_op.KroneckerOperator`, an in-house lazy Kronecker product
  with matvec, matmat, transpose, scalar scaling and `to_array()` (#35).
- `kreg` exports `KernelRegModel`, `Term`, the likelihoods,
  `KroneckerKernel`, `KernelComponent`, `Dimension` and `__version__` at the
  package level.
- GitHub Actions workflow running ruff and pytest on Python 3.11 and 3.12.
- Tests for `KroneckerKernel` operators, the Kronecker operator, and the
  fixed model behaviours below (#35, #36).
- Tests for `Dimension` (#38); likelihood derivatives against `jax.grad` and
  `jax.hessian`, `Term.encode` and `Likelihood.encode` with interval
  dimensions, and the default Newton-CG fit path (#39).
- `README.md` with install instructions, the object model and a runnable
  example; class and constructor docstrings throughout the package.
- Tests for both Newton solvers on a quadratic: convergence, result fields,
  the non-convergence warning and the CG iteration budget.

### Changed

- `requires-python` is `>=3.11`. The code already used `typing.NotRequired`
  and now uses `typing.Self` (#35).
- Dependency floors `jax>=0.4.35`, `numpy>=1.26`, `pandas>=2.2`,
  `matfree>=0.1.0`, `msca>=0.2.0`, the oldest versions the test suite passes
  with. `numpy` is now declared explicitly.
- Package metadata: real description, PEP 639 license expression and
  `license-files`.
- `fit(lam={...})` raises `ValueError` on keys that match no term label
  instead of silently ignoring them (#36).
- `KernelComponent` accepts a single dimension configuration without
  wrapping it in a list (#38).
- Data values outside a dimension's span, or intervals outside its grid,
  raise `ValueError` at encoding time (#38).
- `Likelihood.size` before `attach` and `KernelRegModel.predict` before
  `fit` raise `ValueError` instead of `AttributeError` (#38).
- `KroneckerKernel.span` labels interval dimensions by their grid cells,
  lower and upper bound, so a density for such a kernel is indexed by the
  two interval columns (#39).

### Removed

- The `pykronecker` dependency. It is unmaintained and its JAX warning was
  about to become an error (#35).
- Unused dependencies `tqdm` and `matplotlib`.
- `.pre-commit-config.yaml`, in favour of CI.
- The `examples/` notebooks, which targeted a pre-0.3 API and did not run.
  The README example replaces them.

### Fixed

- `Likelihood.attach` crashed with `KeyError: None` when `weights=None`, so
  no model without a weights column could be fitted (#34).
- The dense solver reused a cached prior Hessian after `fit(lam=...)` changed
  the regularisation, so refits with `use_direct=True` used the wrong
  curvature. Both `functools.cache` decorators on methods are gone; they also
  kept every model and term alive (#36).
- `fit(verbose=False)` still printed the iteration table on the default CG
  path (#36).
- `fit_trimming` wrote `detach` and `x0` into the caller's `solver_options`
  dict (#36).
- `KroneckerKernel.__len__` returns a plain `int` (#35).
- `KernelComponent("age", kfunc)` with a bare string created one dimension
  per character of the name (#38).
- An interval entirely below the grid was silently mapped to the last
  column instead of raising (#38).
- `KroneckerKernel.span` raised a shape error for any kernel with an
  interval dimension, so density normalisation never worked there (#39).
- `fit` docstring: the Nyström rank default is 0, not 25, and
  `grad_decrease` is unused; `fit_trimming` documents the hard-threshold
  return when trimming does not converge.
- `NewtonCG` added `cg_maxiter_increment` before the first CG solve, so the
  first Newton step used a larger budget than `cg_maxiter`.
- `Likelihood.hessian_matrix` built an observations-by-observations
  diagonal matrix (quadratic memory in the number of rows) before the
  product; it now scales the rows of the sparse design matrix.
- `BinomialLikelihood` used a hand-rolled sigmoid in `inv_link` and
  `hessian_diag` and the JAX one in `gradient`; the hand-rolled version has
  a NaN derivative for large negative inputs. All three use
  `jax.scipy.special.expit`.

## [0.4.0] - 2025-11-24

Last release before this changelog was started.
