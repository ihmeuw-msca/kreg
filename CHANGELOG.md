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

### Removed

- The `pykronecker` dependency. It is unmaintained and its JAX warning was
  about to become an error (#35).
- Unused dependencies `tqdm` and `matplotlib`.
- `.pre-commit-config.yaml`, in favour of CI.

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

## [0.4.0] - 2025-11-24

Last release before this changelog was started.
