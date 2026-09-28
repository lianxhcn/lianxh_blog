import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from reppi_ols import fit_reppi_ols


def make_data(seed=123, n=450, N=3000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n + N, 2))
    beta = np.array([0.7, 0.4, -0.25])
    y = beta[0] + X @ beta[1:] + rng.normal(size=n + N)
    yhat = (
        0.3
        + 0.8 * y
        + 0.25 * (X[:, 0] ** 2 - 1)
        + rng.normal(scale=0.6, size=n + N)
    )
    return X[:n], y[:n], yhat[:n], X[n:], yhat[n:], beta


def _cov(A, B=None):
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    Ac = A - A.mean(axis=0, keepdims=True)
    if B is None:
        return Ac.T @ Ac / (A.shape[0] - 1)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    Bc = B - B.mean(axis=0, keepdims=True)
    return Ac.T @ Bc / (A.shape[0] - 1)


def _nearest_psd(A):
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)
    return (vecs * vals) @ vecs.T


def reference_v2_linear(Xl_raw, y, yhl, Xu_raw, yhu, seed, ridge=1e-8, rcond=1e-10):
    """Independent direct transcription of paper v2 Algorithm 1 for OLS.

    Used only as a regression oracle in tests. It deliberately does not call
    implementation-private helpers from reppi_ols.py.
    """
    Xl = np.column_stack([np.ones(len(Xl_raw)), Xl_raw])
    Xu = np.column_stack([np.ones(len(Xu_raw)), Xu_raw])
    n, p = Xl.shape
    N = len(Xu)
    alpha_aug = N / (n + N)

    rng = np.random.default_rng(seed)
    folds = [np.asarray(v, dtype=int) for v in np.array_split(rng.permutation(n), 3)]
    roles = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    theta_parts = []
    weights = []
    score_oof = np.zeros((n, p))

    for a, b, c in roles:
        i0, ical, ieval = folds[a], folds[b], folds[c]
        G0 = Xl[i0].T @ Xl[i0] / len(i0)
        rhs0 = Xl[i0].T @ y[i0] / len(i0)
        theta0 = np.linalg.pinv(G0, rcond=rcond, hermitian=True) @ rhs0

        model = LinearRegression().fit(np.column_stack([Xl_raw[ical], yhl[ical]]), y[ical])
        pred_e = model.predict(np.column_stack([Xl_raw[ieval], yhl[ieval]]))
        pred_u = model.predict(np.column_stack([Xu_raw, yhu]))

        s_e = Xl[ieval] * (Xl[ieval] @ theta0 - pred_e)[:, None]
        s_u = Xu * (Xu @ theta0 - pred_u)[:, None]
        l_e = Xl[ieval] * (Xl[ieval] @ theta0 - y[ieval])[:, None]

        C = _cov(l_e, s_e)
        Vs = _cov(s_e)
        scale = max(float(np.trace(Vs) / p), 1.0)
        Vs_reg = Vs + ridge * scale * np.eye(p)
        M = C @ np.linalg.pinv(Vs_reg, rcond=rcond, hermitian=True)

        corr = alpha_aug * M @ (s_e.mean(axis=0) - s_u.mean(axis=0))
        G = Xl[ieval].T @ Xl[ieval] / len(ieval)
        rhs = Xl[ieval].T @ y[ieval] / len(ieval) + corr
        theta_parts.append(np.linalg.pinv(G, rcond=rcond, hermitian=True) @ rhs)
        weights.append(len(ieval) / n)
        score_oof[ieval] = s_e

    weights = np.asarray(weights)
    theta = np.average(np.vstack(theta_parts), axis=0, weights=weights)

    grad = Xl * (Xl @ theta - y)[:, None]
    Vl = _cov(grad)
    Vs = _cov(score_oof)
    C = _cov(grad, score_oof)
    scale = max(float(np.trace(Vs) / p), 1.0)
    Vs_reg = Vs + ridge * scale * np.eye(p)
    H = Xl.T @ Xl / n
    Hinv = np.linalg.pinv(H, rcond=rcond, hermitian=True)
    meat = Vl - alpha_aug * C @ np.linalg.pinv(Vs_reg, rcond=rcond, hermitian=True) @ C.T
    meat = _nearest_psd(meat)
    vcov = _nearest_psd(Hinv @ meat @ Hinv / n)
    return theta, vcov, alpha_aug, weights


def test_shapes_and_finite_output():
    Xl, y, yhl, Xu, yhu, _ = make_data()
    out = fit_reppi_ols(Xl, y, yhl, Xu, yhu, random_state=9)
    assert out.coef.shape == (3,)
    assert out.se.shape == (3,)
    assert out.vcov.shape == (3, 3)
    assert np.all(np.isfinite(out.coef))
    assert np.all(np.isfinite(out.se))
    assert np.all(out.se >= 0)
    assert np.all(out.ci_lower <= out.ci_upper)
    assert len(out.fold_diagnostics) == 3
    assert np.isclose(out.augmentation_scale, len(Xu) / (len(Xl) + len(Xu)))
    assert np.isclose(out.fold_weights.sum(), 1.0)


def test_reproducibility():
    Xl, y, yhl, Xu, yhu, _ = make_data(seed=456)
    a = fit_reppi_ols(Xl, y, yhl, Xu, yhu, random_state=2026)
    b = fit_reppi_ols(Xl, y, yhl, Xu, yhu, random_state=2026)
    np.testing.assert_allclose(a.coef, b.coef, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.se, b.se, rtol=0, atol=1e-12)


def test_reasonable_recovery_in_large_sample():
    Xl, y, yhl, Xu, yhu, beta = make_data(seed=7, n=1200, N=8000)
    out = fit_reppi_ols(
        Xl,
        y,
        yhl,
        Xu,
        yhu,
        random_state=77,
        recalibrator="ridge",
    )
    # This is not a coverage test. It catches sign, scaling, and intercept mistakes.
    assert np.max(np.abs(out.coef - beta)) < 0.12


def test_linear_recalibrator_also_runs():
    Xl, y, yhl, Xu, yhu, _ = make_data(seed=88)
    out = fit_reppi_ols(
        Xl,
        y,
        yhl,
        Xu,
        yhu,
        recalibrator="linear",
        random_state=1,
    )
    assert np.all(np.isfinite(out.coef))
    assert np.all(np.isfinite(out.se))


def test_input_validation_nonfinite():
    Xl, y, yhl, Xu, yhu, _ = make_data()
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_reppi_ols(Xl, y, yhl, Xu, yhu)


def test_input_validation_dimension_mismatch():
    Xl, y, yhl, Xu, yhu, _ = make_data()
    with pytest.raises(ValueError, match="columns"):
        fit_reppi_ols(Xl, y, yhl, Xu[:, :1], yhu)


def test_existing_intercept_not_duplicated():
    Xl, y, yhl, Xu, yhu, _ = make_data()
    Xl2 = np.column_stack([np.ones(len(Xl)), Xl])
    Xu2 = np.column_stack([np.ones(len(Xu)), Xu])
    out = fit_reppi_ols(Xl2, y, yhl, Xu2, yhu, add_intercept=True)
    assert out.coef.shape == (3,)


def test_near_collinearity_does_not_crash():
    rng = np.random.default_rng(901)
    n, N = 450, 2000
    x1 = rng.normal(size=n + N)
    x2 = x1 + 1e-7 * rng.normal(size=n + N)
    X = np.column_stack([x1, x2])
    y = 1 + 0.4 * x1 - 0.2 * x2 + rng.normal(size=n + N)
    yhat = y + rng.normal(scale=0.5, size=n + N)
    with pytest.warns(RuntimeWarning):
        out = fit_reppi_ols(X[:n], y[:n], yhat[:n], X[n:], yhat[n:], random_state=4)
    assert np.all(np.isfinite(out.coef))
    assert np.all(np.isfinite(out.se))
    assert out.warnings


def test_v2_algorithm_reference_with_unequal_folds():
    # n=451 makes fold sizes unequal, so this catches both the global n/N scaling
    # and Algorithm 1 Step 7 fold-size aggregation.
    Xl, y, yhl, Xu, yhu, _ = make_data(seed=310, n=451, N=2500)
    seed = 991
    out = fit_reppi_ols(
        Xl, y, yhl, Xu, yhu, recalibrator="linear", random_state=seed
    )
    theta_ref, vcov_ref, scale_ref, weights_ref = reference_v2_linear(
        Xl, y, yhl, Xu, yhu, seed
    )
    np.testing.assert_allclose(out.coef, theta_ref, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(out.vcov, vcov_ref, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(out.augmentation_scale, scale_ref, rtol=0, atol=1e-15)
    np.testing.assert_allclose(out.fold_weights, weights_ref, rtol=0, atol=1e-15)
    assert len(np.unique(out.fold_weights)) > 1


def test_summary_reports_baselines_and_algorithm_metadata():
    Xl, y, yhl, Xu, yhu, _ = make_data(seed=777)
    out = fit_reppi_ols(Xl, y, yhl, Xu, yhu, random_state=7)
    text = out.summary()
    assert "human-only coef" in text
    assert "surrogate-only coef" in text
    assert "augmentation_scale=N/(n+N)" in text
