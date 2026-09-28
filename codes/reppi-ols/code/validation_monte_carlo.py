"""Monte Carlo audit for bias, SE calibration, coverage, and efficiency.

Run:
    python validation_monte_carlo.py

The default design is deliberately moderate rather than tiny because RePPI uses
three-way cross-fitting.  This script is an audit, not a formal proof.
"""

import argparse
import numpy as np
from reppi_ols import fit_reppi_ols


def main(R: int = 200, n: int = 600, N: int = 6000, base_seed: int = 30000) -> None:
    beta = np.array([1.0, 0.5, -0.3])
    estimates = []
    ses = []
    cover = []
    human = []

    for rep in range(R):
        rng = np.random.default_rng(base_seed + rep)
        X = rng.normal(size=(n + N, 2))
        y = beta[0] + X @ beta[1:] + rng.normal(size=n + N)

        # Imperfect surrogate with shrinkage, nonlinear distortion and extra noise.
        yhat = (
            0.25
            + 0.75 * y
            + 0.35 * (X[:, 0] ** 2 - 1.0)
            + rng.normal(scale=0.7, size=n + N)
        )

        out = fit_reppi_ols(
            X[:n],
            y[:n],
            yhat[:n],
            X[n:],
            yhat[n:],
            recalibrator="linear",
            random_state=900 + rep,
        )
        estimates.append(out.coef)
        ses.append(out.se)
        cover.append((out.ci_lower <= beta) & (beta <= out.ci_upper))
        human.append(out.human_only_coef)

    estimates = np.asarray(estimates)
    ses = np.asarray(ses)
    cover = np.asarray(cover)
    human = np.asarray(human)

    empirical_sd = estimates.std(axis=0, ddof=1)
    human_sd = human.std(axis=0, ddof=1)

    print(f"R={R}, n_labeled={n}, n_unlabeled={N}")
    print("true beta:       ", beta)
    print("mean estimate:   ", np.round(estimates.mean(axis=0), 4))
    print("bias:            ", np.round(estimates.mean(axis=0) - beta, 4))
    print("empirical SD:    ", np.round(empirical_sd, 4))
    print("mean reported SE:", np.round(ses.mean(axis=0), 4))
    print("95% coverage:    ", np.round(cover.mean(axis=0), 3))
    print("human-only SD:   ", np.round(human_sd, 4))
    print("RePPI/Human SD:  ", np.round(empirical_sd / human_sd, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monte Carlo audit for the RePPI OLS teaching implementation."
    )
    parser.add_argument("--reps", type=int, default=200, help="Monte Carlo repetitions")
    parser.add_argument("--n", type=int, default=600, help="labeled sample size")
    parser.add_argument("--N", type=int, default=6000, help="unlabeled sample size")
    parser.add_argument(
        "--seed", type=int, default=30000, help="base random seed for reproducibility"
    )
    args = parser.parse_args()
    if args.reps <= 0 or args.n <= 0 or args.N <= 0:
        parser.error("--reps, --n, and --N must all be positive integers")
    main(R=args.reps, n=args.n, N=args.N, base_seed=args.seed)
