# Validation report — v0.2.0

Validation date: 2026-09-15

This report records implementation-level checks for `reppi_ols.py`. It is not a substitute for the asymptotic theory in Ji, Lei and Zrnic (2026).

## 1. Formula adjudication

The previous implementation used an evaluation-fold-specific labeled/unlabeled ratio inside the correction and equal averaging across three rotations. Local Codex correctly flagged that this differs from arXiv v2 Algorithm 1.

v0.2.0 therefore follows the June 2026 paper revision:

- global augmentation scale `N/(n+N)` in every rotation;
- Equation (13) `M` without absorbing the sample-ratio factor;
- final aggregation weights `|D_k|/n`;
- plug-in covariance after Remark 5 using out-of-fold fitted scores.

See `ALGORITHM-NOTE.md`.

## 2. Unit tests

Command:

```bash
python -m pytest -q
```

Result:

```text
10 passed
```

The tests cover:

- output dimensions and finite values;
- deterministic reproducibility under a fixed random seed;
- coefficient recovery in a large deterministic simulation;
- Ridge and linear recalibration;
- NaN/Inf rejection;
- design-matrix dimension mismatch rejection;
- prevention of duplicate intercept insertion;
- near-collinearity: warning + pseudoinverse stabilization rather than hard crash;
- independent paper-v2 reference calculation with `n=451` to force unequal fold sizes;
- explicit checking of the v2 point estimate, variance matrix, global augmentation scale, fold weights, and summary metadata.

## 3. Recalibrator smoke tests

The optional `histgb` and `random_forest` recalibrators were rerun after the v0.2.0 correction and returned finite coefficients and standard errors.

## 4. End-to-end example

`python example_simulation.py` runs successfully. With the fixed seed:

```text
True beta:        [ 1.0000,  0.5000, -0.3000]
Human-only OLS:   [ 0.9966,  0.5342, -0.2687]
Surrogate-only:   [ 1.0050,  0.3873, -0.2326]
RePPI:            [ 1.0101,  0.5265, -0.3144]
augmentation      = 0.909091 = 6000/(600+6000)
fold weights      = [0.3333, 0.3333, 0.3333]
```

## 5. Monte Carlo audit

Default design:

- repetitions: 200;
- labeled observations: 600;
- unlabeled observations: 6000;
- true beta: `[1.0, 0.5, -0.3]`;
- surrogate contamination: shrinkage + nonlinear distortion + additional noise.

Fixed-seed result:

```text
bias              = [ 0.0094, -0.0002, -0.0007]
empirical SD      = [ 0.0316,  0.0348,  0.0328]
mean reported SE  = [ 0.0323,  0.0361,  0.0325]
95% coverage      = [ 0.955,   0.955,   0.955 ]
human-only SD     = [ 0.0406,  0.0392,  0.0428]
RePPI/Human SD    = [ 0.777,   0.886,   0.767 ]
```

Interpretation: in this audit DGP, reported SEs track empirical SDs reasonably well, coverage is close to nominal, and RePPI has lower sampling SD than human-only OLS. These numbers are DGP-specific regression checks, not universal claims.

## 6. Small-labeled-sample stress test

With `n=300`, `N=3000`, `R=200`:

```text
bias              = [ 0.0079, -0.0008, -0.0023]
empirical SD      = [ 0.0470,  0.0535,  0.0516]
mean reported SE  = [ 0.0461,  0.0506,  0.0463]
95% coverage      = [ 0.960,   0.955,   0.895 ]
RePPI/Human SD    = [ 0.845,   0.915,   0.819 ]
```

The third coefficient still shows material under-coverage. Three-way cross-fitting reduces the effective labeled sample available to each nuisance/evaluation role; the normal interval is asymptotic and should not be treated as exact in small samples.

## 7. Remaining scope limits

This validation does not establish correctness for:

- nonrandom verification samples without sampling correction;
- generated covariates/regressors;
- fixed effects or panel dependence;
- clustered standard errors;
- survey/frequency weights;
- logistic, Poisson, quantile, or other targets.

Those require separate derivation and tests.
