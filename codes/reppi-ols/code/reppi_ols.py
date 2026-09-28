"""Engineering-hardened OLS implementation of Recalibrated PPI (RePPI).

This module implements the three-way cross-fitted OLS construction in Algorithm 1
of Ji, Lei and Zrnic, "Predictions as Surrogates" (arXiv v2 / accepted-manuscript
algorithm).  It intentionally follows the paper when the older public research code
differs from the revised algorithm.  It is intended for settings with:

    * a labeled sample: X, true outcome Y, and surrogate prediction Yhat;
    * an unlabeled/cheap sample: X and surrogate prediction Yhat only.

The target is the population OLS coefficient of Y on X.  The implementation is
unweighted by design: adding survey/frequency weights requires additional care
in the cross-fitted covariance calculations and is therefore not silently
supported here.

The code is written for research transparency rather than maximal speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import copy
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]


@dataclass
class FoldDiagnostics:
    """Diagnostics for one of the three cross-fitting rotations."""

    rotation: int
    n_initial: int
    n_calibration: int
    n_evaluation: int
    recalibration_r2: float
    recalibration_rmse: float
    eval_gram_condition: float
    score_cov_condition: float
    m_frobenius_norm: float


@dataclass
class RePPIOLSResult:
    """Result returned by :func:`fit_reppi_ols`."""

    coef: FloatArray
    se: FloatArray
    ci_lower: FloatArray
    ci_upper: FloatArray
    pvalue: FloatArray
    vcov: FloatArray
    human_only_coef: FloatArray
    surrogate_only_coef: FloatArray
    feature_names: List[str]
    n_labeled: int
    n_unlabeled: int
    alpha: float
    random_state: Optional[int]
    recalibrator: str
    augmentation_scale: float
    fold_weights: FloatArray
    fold_diagnostics: List[FoldDiagnostics] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self, digits: int = 4) -> str:
        """Return a compact plain-text coefficient table."""
        z = norm.ppf(1 - self.alpha / 2)
        lines = [
            "RePPI OLS",
            f"n_labeled={self.n_labeled}, n_unlabeled={self.n_unlabeled}, "
            f"CI={100 * (1 - self.alpha):.1f}%, z={z:.3f}",
            "",
            f"{'term':<18}{'coef':>12}{'se':>12}{'lower':>12}{'upper':>12}{'p':>12}",
        ]
        for name, b, se, lo, hi, p in zip(
            self.feature_names,
            self.coef,
            self.se,
            self.ci_lower,
            self.ci_upper,
            self.pvalue,
        ):
            lines.append(
                f"{name:<18}{b:>12.{digits}f}{se:>12.{digits}f}"
                f"{lo:>12.{digits}f}{hi:>12.{digits}f}{p:>12.{digits}g}"
            )
        lines.append("")
        lines.append(
            f"augmentation_scale=N/(n+N)={self.augmentation_scale:.6f}; "
            f"fold_weights={np.array2string(self.fold_weights, precision=4)}"
        )
        lines.append(
            "human-only coef:    "
            + np.array2string(self.human_only_coef, precision=digits)
        )
        lines.append(
            "surrogate-only coef:"
            + np.array2string(self.surrogate_only_coef, precision=digits)
        )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {msg}" for msg in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation and linear-algebra helpers
# ---------------------------------------------------------------------------


def _as_2d_float(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (or a 1-D single covariate).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _as_1d_float(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _add_intercept_if_needed(X: FloatArray, tol: float = 1e-12) -> tuple[FloatArray, bool]:
    has_ones = np.any(np.max(np.abs(X - 1.0), axis=0) <= tol)
    if has_ones:
        return X.copy(), False
    return np.column_stack([np.ones(X.shape[0]), X]), True


def _pinv(A: FloatArray, rcond: float) -> FloatArray:
    return np.linalg.pinv(A, rcond=rcond, hermitian=True)


def _safe_condition_number(A: FloatArray) -> float:
    try:
        value = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _sample_cov(X: FloatArray) -> FloatArray:
    """Sample covariance with variables in columns; always return a 2-D matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < 2:
        raise ValueError("At least two observations are required for covariance estimation.")
    centered = X - X.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (X.shape[0] - 1)


def _cross_cov(A: FloatArray, B: FloatArray) -> FloatArray:
    """Sample cross-covariance Cov(A, B), variables in columns."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of observations.")
    if A.shape[0] < 2:
        raise ValueError("At least two observations are required for covariance estimation.")
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    return (Ac.T @ Bc) / (A.shape[0] - 1)


def _nearest_psd(A: FloatArray, floor: float = 0.0) -> FloatArray:
    """Symmetrize and clip numerical negative eigenvalues."""
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, floor)
    return (vecs * vals) @ vecs.T


def _ols_coef(X: FloatArray, y: FloatArray, rcond: float) -> FloatArray:
    gram = X.T @ X / X.shape[0]
    rhs = X.T @ y / X.shape[0]
    return _pinv(gram, rcond) @ rhs


# ---------------------------------------------------------------------------
# Recalibration model
# ---------------------------------------------------------------------------


def _build_recalibrator(
    specification: Union[str, Any],
    random_state: Optional[int],
) -> tuple[Any, str]:
    if not isinstance(specification, str):
        try:
            return clone(specification), specification.__class__.__name__
        except Exception:
            return copy.deepcopy(specification), specification.__class__.__name__

    key = specification.lower().strip()
    if key == "linear":
        return LinearRegression(), "linear"
    if key in {"ridge", "ridgecv"}:
        # Scaling is fitted only on the calibration fold, so cross-fitting is preserved.
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 4, 17)),
        )
        return model, "ridgecv"
    if key in {"rf", "random_forest", "randomforest"}:
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "random_forest"
    if key in {"histgb", "hist_gradient_boosting", "hgb"}:
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=random_state,
        )
        return model, "hist_gradient_boosting"
    raise ValueError(
        "Unknown recalibrator. Use 'linear', 'ridge', 'random_forest', "
        "'histgb', or pass a scikit-learn compatible regressor."
    )


def _recalibration_features(X_raw: FloatArray, yhat: FloatArray) -> FloatArray:
    return np.column_stack([X_raw, yhat])


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def fit_reppi_ols(
    X_labeled: ArrayLike,
    y_labeled: ArrayLike,
    yhat_labeled: ArrayLike,
    X_unlabeled: ArrayLike,
    yhat_unlabeled: ArrayLike,
    *,
    alpha: float = 0.05,
    add_intercept: bool = True,
    recalibrator: Union[str, Any] = "ridge",
    random_state: Optional[int] = 12345,
    decorrelation_ridge: float = 1e-8,
    pinv_rcond: float = 1e-10,
    feature_names: Optional[Sequence[str]] = None,
    min_fold_buffer: int = 3,
) -> RePPIOLSResult:
    """Fit the three-way cross-fitted OLS RePPI estimator.

    Parameters
    ----------
    X_labeled, y_labeled, yhat_labeled
        Covariates, true outcome, and surrogate prediction for observations with
        gold-standard labels.
    X_unlabeled, yhat_unlabeled
        Covariates and surrogate prediction for observations without the true
        outcome. "Unlabeled" means the expensive outcome is unavailable; the
        covariates and cheap prediction must be observed.
    alpha
        Two-sided confidence interval error rate. ``alpha=0.05`` gives 95% CIs.
    add_intercept
        Add an intercept column unless X already contains a column of ones.
    recalibrator
        ``'ridge'`` (default), ``'linear'``, ``'random_forest'``, ``'histgb'``,
        or a scikit-learn compatible regressor implementing fit/predict.
    random_state
        Seed for the three-way split and stochastic recalibrators.
    decorrelation_ridge
        Relative ridge added to the covariance matrix used to estimate the
        decorrelation matrix M.  This is a numerical stabilization device.
    pinv_rcond
        Relative cutoff used by Moore-Penrose pseudoinverses.
    feature_names
        Optional names for user-supplied X columns.  The intercept name is added
        automatically when needed.
    min_fold_buffer
        Require each evaluation fold to contain at least p + this many
        observations, where p is the number of OLS coefficients.

    Returns
    -------
    RePPIOLSResult
        Point estimates, sandwich standard errors, confidence intervals,
        p-values, baselines, and fold-level diagnostics.

    Notes
    -----
    The target is an OLS coefficient vector for a surrogate *outcome*.  This
    function is not a correction for an AI-generated regressor/covariate.

    Cross-fitting uses exactly three folds and the cyclic roles in Algorithm 1:
    initial OLS fit, recalibration fit, and evaluation/correction.  The global
    augmentation factor N/(n+N), fold-size aggregation, and plug-in covariance
    formula follow arXiv v2 (June 2026).  Pseudoinverses and a tiny covariance
    ridge are used to avoid hard failures under near-collinearity; diagnostics
    record when conditioning is poor.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if decorrelation_ridge < 0:
        raise ValueError("decorrelation_ridge must be nonnegative.")
    if pinv_rcond <= 0:
        raise ValueError("pinv_rcond must be positive.")

    Xl_raw = _as_2d_float(X_labeled, "X_labeled")
    Xu_raw = _as_2d_float(X_unlabeled, "X_unlabeled")
    y = _as_1d_float(y_labeled, "y_labeled")
    yh_l = _as_1d_float(yhat_labeled, "yhat_labeled")
    yh_u = _as_1d_float(yhat_unlabeled, "yhat_unlabeled")

    n, k = Xl_raw.shape
    N, ku = Xu_raw.shape
    if ku != k:
        raise ValueError(
            f"X_labeled has {k} columns but X_unlabeled has {ku}; they must match."
        )
    if y.size != n or yh_l.size != n:
        raise ValueError("Labeled X, y, and yhat must have the same number of rows.")
    if yh_u.size != N:
        raise ValueError("X_unlabeled and yhat_unlabeled must have the same number of rows.")
    if n < 9:
        raise ValueError("At least 9 labeled observations are required for three-way cross-fitting.")
    if N < 2:
        raise ValueError("At least 2 unlabeled observations are required.")

    if add_intercept:
        Xl, intercept_added = _add_intercept_if_needed(Xl_raw)
        Xu, intercept_added_u = _add_intercept_if_needed(Xu_raw)
        if intercept_added != intercept_added_u:
            raise ValueError(
                "Intercept detection differs between labeled and unlabeled X. "
                "Make the design matrices consistent or set add_intercept=False."
            )
    else:
        Xl, Xu = Xl_raw.copy(), Xu_raw.copy()
        intercept_added = False

    p = Xl.shape[1]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    folds = [np.asarray(x, dtype=int) for x in np.array_split(perm, 3)]
    min_fold = min(len(f) for f in folds)
    if min_fold < p + min_fold_buffer:
        raise ValueError(
            f"Each fold should have at least p + {min_fold_buffer} observations. "
            f"Here p={p}, smallest fold={min_fold}. Increase the labeled sample "
            "or reduce the number of regressors."
        )

    # Feature names refer to user-supplied X, not the added intercept.
    if feature_names is None:
        raw_names = [f"x{j + 1}" for j in range(k)]
    else:
        raw_names = list(feature_names)
        if len(raw_names) != k:
            raise ValueError("feature_names must have one entry per column of X_labeled.")
    names = (["Intercept"] + raw_names) if intercept_added else raw_names

    warning_messages: List[str] = []
    fold_diags: List[FoldDiagnostics] = []
    theta_parts: List[FloatArray] = []
    fold_weights: List[float] = []
    score_labeled_oof = np.zeros((n, p), dtype=float)

    # Algorithm 1 uses the full labeled/unlabeled ratio in every rotation:
    # 1/(1+n/N) = N/(n+N).  This must not be recomputed from the evaluation fold.
    augmentation_scale = N / (n + N)

    # Cyclic three-way cross-fitting roles:
    # (initial, calibration, evaluation) = (0,1,2), (1,2,0), (2,0,1).
    role_triplets = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    for rotation, (a, b, c) in enumerate(role_triplets, start=1):
        idx_initial = folds[a]
        idx_cal = folds[b]
        idx_eval = folds[c]

        theta0 = _ols_coef(Xl[idx_initial], y[idx_initial], pinv_rcond)

        model, model_name = _build_recalibrator(recalibrator, random_state)
        Z_cal = _recalibration_features(Xl_raw[idx_cal], yh_l[idx_cal])
        model.fit(Z_cal, y[idx_cal])

        Z_eval = _recalibration_features(Xl_raw[idx_eval], yh_l[idx_eval])
        Z_u = _recalibration_features(Xu_raw, yh_u)
        pred_eval = np.asarray(model.predict(Z_eval), dtype=float).reshape(-1)
        pred_u = np.asarray(model.predict(Z_u), dtype=float).reshape(-1)
        if pred_eval.size != idx_eval.size or pred_u.size != N:
            raise ValueError("The recalibrator's predict() output has an invalid shape.")
        if not np.isfinite(pred_eval).all() or not np.isfinite(pred_u).all():
            raise ValueError("The recalibrator produced NaN or infinite predictions.")

        # Estimated conditional score s(X, Yhat) at theta0.
        score_eval = Xl[idx_eval] * (
            Xl[idx_eval] @ theta0 - pred_eval
        )[:, None]
        score_u = Xu * (Xu @ theta0 - pred_u)[:, None]

        # Gold-standard score on the independent evaluation fold.
        true_score_eval = Xl[idx_eval] * (
            Xl[idx_eval] @ theta0 - y[idx_eval]
        )[:, None]

        cov_ls = _cross_cov(true_score_eval, score_eval)
        cov_ss = _sample_cov(score_eval)
        scale = max(float(np.trace(cov_ss) / p), 1.0)
        cov_ss_reg = cov_ss + decorrelation_ridge * scale * np.eye(p)
        score_cov_cond = _safe_condition_number(cov_ss_reg)

        # Equation (13): M itself is the unscaled covariance ratio.  Algorithm 1
        # Step 5 applies the global N/(n+N) factor to the linear imputed loss.
        M = cov_ls @ _pinv(cov_ss_reg, pinv_rcond)
        A = augmentation_scale * M

        mean_eval = score_eval.mean(axis=0)
        mean_u = score_u.mean(axis=0)
        correction = A @ (mean_eval - mean_u)

        gram_eval = Xl[idx_eval].T @ Xl[idx_eval] / idx_eval.size
        rhs_eval = Xl[idx_eval].T @ y[idx_eval] / idx_eval.size + correction
        theta_c = _pinv(gram_eval, pinv_rcond) @ rhs_eval
        theta_parts.append(theta_c)
        fold_weights.append(idx_eval.size / n)

        # Store the *unscaled* out-of-fold fitted score.  The revised paper gives
        # a direct plug-in covariance estimator in terms of this score.
        score_labeled_oof[idx_eval] = score_eval

        gram_cond = _safe_condition_number(gram_eval)
        r2 = float(r2_score(y[idx_eval], pred_eval))
        rmse = float(np.sqrt(mean_squared_error(y[idx_eval], pred_eval)))
        m_norm = float(np.linalg.norm(M, ord="fro"))
        fold_diags.append(
            FoldDiagnostics(
                rotation=rotation,
                n_initial=idx_initial.size,
                n_calibration=idx_cal.size,
                n_evaluation=idx_eval.size,
                recalibration_r2=r2,
                recalibration_rmse=rmse,
                eval_gram_condition=gram_cond,
                score_cov_condition=score_cov_cond,
                m_frobenius_norm=m_norm,
            )
        )

        if gram_cond > 1e10:
            warning_messages.append(
                f"Rotation {rotation}: evaluation design matrix is ill-conditioned "
                f"(condition number {gram_cond:.2e}); pseudoinverse was used."
            )
        if score_cov_cond > 1e10:
            warning_messages.append(
                f"Rotation {rotation}: recalibrated score covariance is ill-conditioned "
                f"(condition number {score_cov_cond:.2e}); ridge/pseudoinverse stabilization was used."
            )
        if m_norm > 1e4:
            warning_messages.append(
                f"Rotation {rotation}: decorrelation matrix has a very large norm ({m_norm:.2e}); "
                "inspect collinearity and recalibration quality."
            )

    # Algorithm 1 Step 7: weight the fold-specific estimators by the size of
    # their evaluation fold.  Equal averaging is only identical when n is exactly
    # divisible by three.
    fold_weights_arr = np.asarray(fold_weights, dtype=float)
    theta_hat = np.average(np.vstack(theta_parts), axis=0, weights=fold_weights_arr)

    # Revised-paper plug-in variance estimator (arXiv v2, immediately after
    # Remark 5).  For OLS, H is the labeled-sample average of X X'.
    true_grad = Xl * (Xl @ theta_hat - y)[:, None]
    V_l = _sample_cov(true_grad)
    V_s = _sample_cov(score_labeled_oof)
    C_ls = _cross_cov(true_grad, score_labeled_oof)

    vscale = max(float(np.trace(V_s) / p), 1.0)
    V_s_reg = V_s + decorrelation_ridge * vscale * np.eye(p)
    V_s_cond = _safe_condition_number(V_s_reg)

    H = Xl.T @ Xl / n
    H_cond = _safe_condition_number(H)
    H_inv = _pinv(H, pinv_rcond)
    meat = V_l - augmentation_scale * C_ls @ _pinv(V_s_reg, pinv_rcond) @ C_ls.T
    meat = 0.5 * (meat + meat.T)

    eigvals = np.linalg.eigvalsh(meat)
    meat_scale = max(float(np.trace(V_l) / p), 1.0)
    if float(eigvals.min()) < -1e-8 * meat_scale:
        warning_messages.append(
            "Finite-sample plug-in covariance meat is not PSD; a nearest-PSD "
            "projection was applied. Inspect sample size and score conditioning."
        )
    meat = _nearest_psd(meat)
    vcov_scaled = H_inv @ meat @ H_inv
    vcov = _nearest_psd(vcov_scaled / n)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    if V_s_cond > 1e10:
        warning_messages.append(
            f"Out-of-fold score covariance is ill-conditioned "
            f"(condition number {V_s_cond:.2e}); ridge/pseudoinverse stabilization was used."
        )
    if H_cond > 1e10:
        warning_messages.append(
            f"Labeled-sample Hessian/design second moment is ill-conditioned "
            f"(condition number {H_cond:.2e}); inference uses a pseudoinverse."
        )

    zcrit = norm.ppf(1.0 - alpha / 2.0)
    ci_lower = theta_hat - zcrit * se
    ci_upper = theta_hat + zcrit * se
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.divide(theta_hat, se, out=np.full_like(theta_hat, np.nan), where=se > 0)
    pvalue = 2.0 * norm.sf(np.abs(zstat))

    human_coef = _ols_coef(Xl, y, pinv_rcond)
    surrogate_coef = _ols_coef(Xu, yh_u, pinv_rcond)

    # Duplicate warnings are not useful to callers.
    warning_messages = list(dict.fromkeys(warning_messages))
    for msg in warning_messages:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return RePPIOLSResult(
        coef=np.asarray(theta_hat),
        se=np.asarray(se),
        ci_lower=np.asarray(ci_lower),
        ci_upper=np.asarray(ci_upper),
        pvalue=np.asarray(pvalue),
        vcov=np.asarray(vcov),
        human_only_coef=np.asarray(human_coef),
        surrogate_only_coef=np.asarray(surrogate_coef),
        feature_names=names,
        n_labeled=n,
        n_unlabeled=N,
        alpha=alpha,
        random_state=random_state,
        recalibrator=model_name,
        augmentation_scale=float(augmentation_scale),
        fold_weights=np.asarray(fold_weights_arr),
        fold_diagnostics=fold_diags,
        warnings=warning_messages,
    )


__all__ = ["fit_reppi_ols", "RePPIOLSResult", "FoldDiagnostics"]
