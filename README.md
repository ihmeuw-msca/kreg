# kreg

Kernel regression with Kronecker-structured priors.

`kreg` fits a generalised linear model (binomial, Gaussian or Poisson) whose
linear predictor is a sum of terms. A term is either a plain coefficient or a
smooth surface over a grid. The grid is the Cartesian product of one or more
dimensions, each with its own kernel function, and the prior on the surface is
the Kronecker product of the per-dimension kernel matrices. The Kronecker
structure is never materialised: products with the kernel and its inverse go
through per-factor eigendecompositions, so a grid of 200 locations by 100 ages
by 50 years costs three small eigendecompositions, not one of size one
million.

Observations are mapped onto the grid by a sparse encoding matrix. A
dimension can match data exactly (a location id), through a tuple of columns
(a hierarchy such as region and country), or through an interval (an age
range, which is split over the grid cells it covers with integration
weights). Fitting uses Newton's method with conjugate-gradient inner solves,
or a dense Newton solver for small problems, and `fit_trimming` gives a
robust fit by trimming a fixed fraction of outlying observations.

## Install

The package is not on PyPI. Install from GitHub:

```bash
pip install "kreg @ git+https://github.com/ihmeuw-msca/kreg.git"
```

Append `@<tag>` to the URL to pin a release.

Python 3.11 or 3.12 is required. The runtime dependencies are `jax`,
`numpy`, `pandas`, `matfree` and `msca`.

Importing `kreg` enables 64-bit floating point in JAX for the whole process
(`jax_enable_x64`). Every kreg computation relies on it, and it changes the
default dtype of any other JAX code you run in the same process.

## The object model

| Class | Role |
| --- | --- |
| `Dimension` | One axis of a grid. Plain (`"location_id"`), coordinate tuple (`coords=("region", "country")`), or interval (`interval=("age_start", "age_end")`). The span is read from the data at fit time. |
| `KernelComponent` | One or more dimensions plus a kernel function. Builds one kernel matrix over the product of its dimensions' spans. |
| `KroneckerKernel` | A list of components. Its kernel matrix is the Kronecker product of the component matrices, in list order. |
| `Term` | A block of coefficients. With a kernel it is a surface penalised by `0.5 * lam * x' K^{-1} x`; without one it is a single coefficient on a covariate column named by the term. The term named `"intercept"` is not multiplied by any column. |
| `Likelihood` | `BinomialLikelihood`, `GaussianLikelihood` or `PoissonLikelihood`. Names the observation, optional weight and optional offset columns. |
| `KernelRegModel` | Terms plus a likelihood. `fit`, `fit_trimming`, `predict`. |

Kernel functions come from `kreg.kernel.factory`: Matérn 3/2 and 5/2,
Gaussian RBF, rational quadratic, exponential similarity and a shifted linear
kernel. Each builder returns a scalar kernel; wrap it with `vectorize_kfunc`
before passing it to a component.

## Example

Prevalence by year and age interval, with a binomial likelihood.

```python
import numpy as np
import pandas as pd

from kreg import (
    BinomialLikelihood,
    KernelComponent,
    KernelRegModel,
    KroneckerKernel,
    Term,
)
from kreg.kernel.factory import build_matern_three_half_kfunc, vectorize_kfunc

# Synthetic data: five years, sixteen five-year age cells.
rng = np.random.default_rng(0)
grid = np.arange(0.0, 85.0, 5.0)
data = pd.DataFrame(
    [
        (year, a0, a1)
        for year in range(2000, 2005)
        for a0, a1 in zip(grid[:-1], grid[1:])
    ],
    columns=["year_id", "age_start", "age_end"],
)
age_mid = 0.5 * (data["age_start"] + data["age_end"])
truth = 1 / (
    1 + np.exp(-(-3.0 + 0.05 * age_mid + 0.2 * (data["year_id"] - 2000)))
)
data["sample_size"] = 200.0
data["obs"] = rng.binomial(200, truth) / data["sample_size"]

# A smooth surface over year x age.
kernel = KroneckerKernel(
    [
        KernelComponent(
            "year_id",
            vectorize_kfunc(build_matern_three_half_kfunc(rho=2.0)),
        ),
        KernelComponent(
            {"name": "age", "interval": ("age_start", "age_end")},
            vectorize_kfunc(build_matern_three_half_kfunc(rho=20.0)),
        ),
    ]
)
model = KernelRegModel(
    terms=[Term("intercept", kernel=kernel, lam=1.0)],
    likelihood=BinomialLikelihood(obs="obs", weights="sample_size"),
)

x, info = model.fit(data, verbose=False)
assert info.converged

pred = data[["year_id", "age_start", "age_end"]].copy()
pred["pred"] = model.predict(pred)
```

Points to note:

- **Spans come from the data.** The grid of each dimension is read from the
  data frame passed to `fit`. For an interval dimension the rows must be the
  grid cells themselves, contiguous and without gaps. Pass `data_span=` to
  `fit` when the observations use other intervals, for example wider age
  ranges, and the span data frame holds the cells.
- **Prediction data** carries only the dimension columns and any covariate or
  offset columns. Values outside the span raise.
- **`lam` can be changed per fit** without rebuilding the model. A scalar
  applies to every term. A dict is keyed by `Term.label`, which is the term
  name for a plain term and `name/kernel_label` for a kernel term, with the
  kernel label joining its dimension labels by `*`:

  ```python
  print(model.terms[0].label)
  # intercept/year_id*(age_start-age_end)
  x, info = model.fit(
      data,
      lam={"intercept/year_id*(age_start-age_end)": 10.0},
      verbose=False,
  )
  ```

  Unknown keys raise a `ValueError` listing the available labels.
- **Covariate terms.** `Term("sdi")` adds a single unpenalised coefficient on
  the `sdi` column. Pass `lam_ridge=` for a ridge penalty.
- **Solver choice.** `fit` uses Newton-CG by default. `use_direct=True`
  forms the dense Hessian; fine for a few thousand grid points, not beyond.
  `info.log.to_dataframe()` has the per-iteration objective, gradient norms
  and step sizes.
- **Robust fitting.** `fit_trimming(data, inlier_pct=0.95)` iteratively
  down-weights the observations with the largest negative log-likelihood
  until the trimming weights are all zero or one, and returns the fit and the
  weights.

## Development

```bash
git clone https://github.com/ihmeuw-msca/kreg.git
cd kreg
python -m venv env && source env/bin/activate
pip install -e ".[test]"
pytest -W error
ruff check src tests && ruff format --check src tests
```

Continuous integration runs the same commands on Python 3.11 and 3.12.
Changes are recorded in `CHANGELOG.md`.
