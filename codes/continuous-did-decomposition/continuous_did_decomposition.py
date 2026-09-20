"""连续 DID 系数分解：两期、无额外协变量、等权样本的教学实现。

运行示例：
    python continuous_did_decomposition.py
    python continuous_did_decomposition.py --self-test
    python continuous_did_decomposition.py --self-test --output-dir results

输入：每个单位一行；dose 为当前比较的有效处理强度；dy 为同一两期的结果差分。
零强度代表未处理，处理单位的处理强度严格为正。不自动删除缺失值。
只计算统计点估计，不提供因果识别保证、标准误或多期联合推断。
方法来源：Yu (2026), arXiv:2609.09488v2, Theorem 3。
教学推导、示例和检查程序为本文编写，并非作者官方复现程序。
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def decompose_did(dose: ArrayLike, dy: ArrayLike) -> dict[str, float]:
    """返回原连续 DID 系数、水平对比、响应指数、权重及重构结果。"""
    # 不允许复数。缺失值、非法维度和不同长度均报错，避免悄悄改变样本。
    if np.iscomplexobj(dose) or np.iscomplexobj(dy):
        raise ValueError("处理强度和结果变化量必须是实数。")
    d = np.asarray(dose, dtype=float)
    y = np.asarray(dy, dtype=float)
    if d.ndim != 1 or y.ndim != 1 or d.size != y.size or d.size == 0:
        raise ValueError("dose 和 dy 必须是非空、等长的一维数组。")
    if not (np.isfinite(d).all() and np.isfinite(y).all()):
        raise ValueError("输入含缺失值或无穷值；请先统一确定估计样本。")
    if np.any(d < 0):
        raise ValueError("本实现只接受零强度控制组与正强度处理组。")

    treated = d > 0
    if treated.sum() < 2 or (~treated).sum() == 0:
        raise ValueError("完整分解需要控制组，且至少有两个处理单位。")

    # 所有方差、协方差均按经验分布计算：分母为本组样本数。
    dt, yt = d[treated], y[treated]
    mu = dt.mean()
    vd = dt.var(ddof=0)
    if not np.isfinite(vd) or vd <= 0:
        raise ValueError("处理组内的强度方差为零或溢出，响应指数无法计算。")
    if np.sqrt(vd) <= 100 * np.finfo(float).eps * abs(mu):
        raise ValueError("处理强度差异接近浮点精度；请检查数据及计量单位。")

    rho = (~treated).mean()
    delta = yt.mean() - y[~treated].mean()
    theta = np.mean((dt - mu) * (yt - yt.mean())) / vd
    beta = np.mean((d - d.mean()) * (y - y.mean())) / d.var(ddof=0)
    weight = vd / (vd + rho * mu**2)
    response_component = weight * theta
    level_component = (1 - weight) * delta / mu
    rebuilt = response_component + level_component

    result = {
        "beta": float(beta), "delta_L": float(delta),
        "theta_R": float(theta), "lambda_R": float(weight),
        "mu_D": float(mu), "var_D": float(vd), "rho": float(rho),
        "level_component": float(level_component),
        "response_component": float(response_component),
        "beta_rebuilt": float(rebuilt),
    }
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("中间运算溢出；请检查数值范围并合理调整单位。")

    # 容许浮点舍入误差。出现严重不一致时，不静默返回错误结果。
    scale = max(1.0, abs(beta), abs(level_component), abs(response_component))
    np.testing.assert_allclose(beta, rebuilt, rtol=1e-10, atol=1e-12 * scale)
    return result


def example_rows() -> list[dict]:
    """固定的四个教学情形；所有数字均由函数实际计算。"""
    cases = [
        ("处理效应不随正强度变化", [0, 0, 1, 3], [0, 0, 10, 10]),
        ("处理效应与强度成正比", [0, 0, 1, 3], [0, 0, 2, 6]),
        ("组间差异与组内系数异号", [0, 0, 1, 3], [0, 0, 14, 6]),
        ("增加控制组", [0, 0, 0, 0, 0, 0, 1, 3], [0, 0, 0, 0, 0, 0, 10, 10]),
    ]
    return [{"case": name, **decompose_did(d, y)} for name, d, y in cases]


def run_tests() -> dict:
    """独立 OLS 对照、随机样本恒等式、变换性质及非法输入检查。"""
    rng = np.random.default_rng(20260916)
    max_rebuild_error = 0.0
    max_ols_error = 0.0
    max_centered_error = 0.0
    for _ in range(1000):
        n0, n1 = rng.integers(2, 150, size=2)
        d = np.r_[np.zeros(n0), rng.lognormal(0.3, 0.8, size=n1)]
        # 随机非线性结果：没有人为强制平行趋势或线性因果效应。
        y = 2.0 * (d > 0) + 0.4 * d - 0.03 * d**2 + rng.normal(size=d.size)
        y[:n0] += rng.normal()
        r = decompose_did(d, y)
        x = np.column_stack([np.ones(d.size), d])
        b_ols = np.linalg.lstsq(x, y, rcond=None)[0][1]
        np.testing.assert_allclose(r["beta"], b_ols, rtol=1e-10, atol=1e-10)
        max_ols_error = max(max_ols_error, abs(r["beta"] - b_ols))
        max_rebuild_error = max(max_rebuild_error, abs(r["beta"] - r["beta_rebuilt"]))

        t = (d > 0).astype(float)
        z = t * (d - r["mu_D"])
        x_two = np.column_stack([np.ones(d.size), t, z])
        b_two = np.linalg.lstsq(x_two, y, rcond=None)[0]
        np.testing.assert_allclose(b_two[1:], [r["delta_L"], r["theta_R"]], atol=1e-10)
        max_centered_error = max(max_centered_error, float(np.max(np.abs(b_two[1:] - [r["delta_L"], r["theta_R"]]))))

    base = decompose_did([0, 0, 1, 3], [0, 0, 10, 10])
    for scale in (0.001, 0.1, 100, 100000):
        r = decompose_did(np.array([0, 0, 1, 3]) * scale, [0, 0, 10, 10])
        np.testing.assert_allclose(r["lambda_R"], base["lambda_R"])
        np.testing.assert_allclose(r["beta"] * scale, base["beta"])
        np.testing.assert_allclose(r["delta_L"], base["delta_L"])
    shift = decompose_did([0, 0, 1, 3], [7, 7, 17, 17])
    for key in ("beta", "delta_L", "theta_R", "lambda_R"):
        np.testing.assert_allclose(shift[key], base[key], atol=1e-12)

    expected = [(10/3, 10, 0), (2, 4, 2), (2, 10, -4), (3.75, 10, 0)]
    for row, (beta, delta, theta) in zip(example_rows(), expected):
        np.testing.assert_allclose([row["beta"], row["delta_L"], row["theta_R"]], [beta, delta, theta], atol=1e-12)

    invalid = [
        ([], []), ([0, 1, 3], [0, 1]), ([[0, 1, 3]], [[0, 1, 2]]),
        ([0, -1, 3], [0, 1, 2]), ([0, 1, np.nan], [0, 1, 2]),
        ([0, 1, 3], [0, np.inf, 2]), ([1, 2, 3], [0, 1, 2]),
        ([0, 0, 1], [0, 1, 2]), ([0, 1, 1], [0, 1, 2]),
        ([0, 1, 3 + 1j], [0, 1, 2]),
    ]
    for d, y in invalid:
        try:
            decompose_did(d, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法输入未被拒绝: {d}, {y}")

    return {
        "status": "PASS", "random_datasets": 1000,
        "deterministic_examples": 4, "invalid_input_cases": len(invalid),
        "max_reconstruction_abs_error": max_rebuild_error,
        "max_independent_OLS_abs_error": max_ols_error,
        "max_centered_OLS_abs_error": max_centered_error,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scope": "两期等权点估计和数值恒等式；未验证因果识别、标准误或多期联合推断。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="运行独立 OLS 等数值检查")
    parser.add_argument("--output-dir", type=Path, help="可选：保存结果 CSV 与测试 JSON")
    args = parser.parse_args()
    rows = example_rows()
    fields = ("beta", "delta_L", "theta_R", "lambda_R", "level_component", "response_component", "beta_rebuilt")
    print("情形 | " + " | ".join(fields))
    for row in rows:
        print(row["case"] + " | " + " | ".join(f"{row[k]:.6f}" for k in fields))
    report = run_tests() if args.self_test else None
    if report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "teaching_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if report:
            (args.output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
