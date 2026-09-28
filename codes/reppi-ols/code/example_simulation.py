"""Minimal reproducible example for the robust RePPI OLS implementation."""

import numpy as np
from reppi_ols import fit_reppi_ols


def main() -> None:
    rng = np.random.default_rng(20260915)

    n_labeled = 600
    n_unlabeled = 6000
    beta = np.array([1.0, 0.50, -0.30])  # intercept, x1, x2

    X_all = rng.normal(size=(n_labeled + n_unlabeled, 2))
    eps = rng.normal(scale=1.0, size=n_labeled + n_unlabeled)
    y_all = beta[0] + X_all @ beta[1:] + eps

    # A deliberately imperfect surrogate: shrinkage + nonlinear distortion + noise.
    # The surrogate is cheap and observed for everyone; the true y is costly.
    yhat_all = (
        0.25
        + 0.75 * y_all
        + 0.35 * (X_all[:, 0] ** 2 - 1.0)
        + rng.normal(scale=0.7, size=n_labeled + n_unlabeled)
    )

    X_l = X_all[:n_labeled]
    y_l = y_all[:n_labeled]
    yhat_l = yhat_all[:n_labeled]
    X_u = X_all[n_labeled:]
    yhat_u = yhat_all[n_labeled:]

    result = fit_reppi_ols(
        X_l,
        y_l,
        yhat_l,
        X_u,
        yhat_u,
        recalibrator="ridge",
        random_state=20260915,
        feature_names=["x1", "x2"],
    )

    print(result.summary())
    print("\nTrue beta:       ", beta)
    print("Human-only OLS:  ", np.round(result.human_only_coef, 4))
    print("Surrogate-only:  ", np.round(result.surrogate_only_coef, 4))

    print("\nFold diagnostics:")
    for d in result.fold_diagnostics:
        print(
            f"rotation={d.rotation}, R2={d.recalibration_r2:.3f}, "
            f"RMSE={d.recalibration_rmse:.3f}, "
            f"cond(score)={d.score_cov_condition:.2e}"
        )


if __name__ == "__main__":
    main()
