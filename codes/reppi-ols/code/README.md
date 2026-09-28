# RePPI OLS robust Python implementation v0.2.0

这是一个面向实证研究者的 RePPI OLS 最小可用实现。它针对如下数据结构：

- labeled sample: 有 `X`、人工/金标准 outcome `Y`、AI surrogate `Yhat`；
- unlabeled sample: 有 `X` 和 `Yhat`，但没有昂贵的真实 `Y`。

目标参数是 `Y` 关于 `X` 的 population OLS 系数。**v0.2.0 以 Ji, Lei and Zrnic (2026) 的 arXiv v2 / accepted-manuscript Algorithm 1 为规范基准**。当作者较早公开的 research code 与 2026 年修订论文在缩放或 fold 汇总上存在差异时，本实现以论文 v2 的书面算法为准。详见 `ALGORITHM-NOTE.md`。

## 安装

```bash
python -m pip install -r requirements.txt
```

也可以在当前目录安装：

```bash
python -m pip install -e .
python -m pip install pytest
```

## 最小调用

```python
from reppi_ols import fit_reppi_ols

result = fit_reppi_ols(
    X_labeled,
    y_labeled,
    yhat_labeled,
    X_unlabeled,
    yhat_unlabeled,
    recalibrator="ridge",
    random_state=12345,
)

print(result.summary())
```

`result` 包含：

- `coef`: RePPI OLS 系数；
- `se`: v2 plug-in covariance 对应的标准误；
- `ci_lower`, `ci_upper`: 默认 95% 置信区间；
- `pvalue`: 双侧正态近似 p 值；
- `human_only_coef`: 只用人工真值样本的 OLS；
- `surrogate_only_coef`: 直接把 AI surrogate 当 outcome 的 naive OLS；
- `augmentation_scale`: 全样本比例 `N/(n+N)`；
- `fold_weights`: Algorithm 1 Step 7 的 evaluation-fold 权重；
- `fold_diagnostics`: 三次 cross-fitting 的 recalibration 和数值稳定性诊断。

## v0.2.0 对齐的三个关键公式

定义

$$
\alpha_{n}=\frac{N}{n+N}.
$$

每个 rotation 先在独立 evaluation fold 上估计

$$
\widehat M
=\widehat{\operatorname{Cov}}(\nabla\ell,\widehat s)
\widehat{\operatorname{Cov}}(\widehat s)^{-1},
$$

再用 `alpha_n * M` 进入 PPI correction。**不能用 evaluation fold 大小重新计算 `alpha_n`**。

最终点估计按 Algorithm 1 Step 7 汇总：

$$
\widehat\theta^{CF}=\sum_{k=1}^{3}\frac{|D_k|}{n}\widehat\theta^{k}.
$$

标准误使用 arXiv v2 在 Remark 5 后给出的 plug-in covariance：

$$
\widehat\Sigma
=\widehat H^{-1}
\left\{
\widehat V_{\ell}
-\frac{N}{n+N}
\widehat C_{\ell s}
\widehat V_s^{-1}
\widehat C_{\ell s}^{\top}
\right\}
\widehat H^{-1},
$$

最终 `vcov = Sigma / n`。

## 运行示例与测试

```bash
python example_simulation.py
python -m pytest -q
python validation_monte_carlo.py
python validation_monte_carlo.py --reps 200 --n 300 --N 3000
```

当前测试为 `10 passed`。其中新增的 reference test 独立转写论文 v2 Algorithm 1，并使用 `n=451` 制造不等长三折，从而同时检查：

- 全样本 `N/(n+N)` 缩放；
- fold-size weighted aggregation；
- v2 plug-in covariance。

## 为什么默认使用 `ridge`

作者论文允许用 flexible learner 学习 imputed score。对 OLS 而言，本实现通过拟合 `E[Y|X,Yhat]` 来构造条件 score。默认使用带标准化的 `RidgeCV`，是因为 labeled sample 往往不大，`[X, Yhat]` 容易出现强共线性。也支持：

```python
recalibrator="linear"
recalibrator="random_forest"
recalibrator="histgb"
```

或者传入任意兼容 scikit-learn `fit/predict` 接口的回归器。

## 数值稳定性处理

本实现额外加入：

- shape、NaN、Inf 检查；
- 固定 `random_state`；
- 近奇异矩阵使用 Moore-Penrose pseudoinverse；
- score covariance 使用小幅 relative ridge；
- plug-in covariance 做对称化与必要的 PSD 数值投影；
- condition number、out-of-fold R2/RMSE、`M` 矩阵范数诊断。

这些属于工程防护，不改变论文目标 estimand。

## 适用边界

当前版本只针对 **AI 生成/预测 outcome**。如果 AI 生成的是解释变量 `X`，属于 imputed covariate / measurement error 问题，不能直接使用本函数。

当前版本有意不支持 sample weights、fixed effects、cluster-robust SE 或 panel-specific dependence。相关扩展需要重新推导 point estimate 与 influence-function covariance，不能只给每行乘权重或先残差化就宣称推断有效。

当前无权重实现要求两批样本来自相同的联合总体分布，例如随机抽取人工标注样本。仅有给定 X 后随机缺失 (MAR) 不足以保证适用。如果人工核验样本按模型置信度、文本难度、行业等非随机抽取，应另行处理 sampling/propensity correction。

## 当前 Monte Carlo 审计快照

`validation_monte_carlo.py` 默认运行 200 次模拟，`n=600, N=6000`：

```text
bias              = [ 0.0094, -0.0002, -0.0007]
empirical SD      = [ 0.0316,  0.0348,  0.0328]
mean reported SE  = [ 0.0323,  0.0361,  0.0325]
95% coverage      = [ 0.955,   0.955,   0.955 ]
RePPI/Human SD    = [ 0.777,   0.886,   0.767 ]
```

`n=300, N=3000` 的 stress test：

```text
95% coverage      = [0.960, 0.955, 0.895]
RePPI/Human SD    = [0.845, 0.915, 0.819]
```

第三个系数仍有明显 under-coverage，因此不能把渐近理论解释为“小样本一定稳健”。

## 文献与代码

- Ji, W., Lei, L., & Zrnic, T. (2026). Predictions as Surrogates: Revisiting Surrogate Outcomes in the Age of AI. *Biometrika*, asag053. https://doi.org/10.1093/biomet/asag053
- arXiv v2: https://arxiv.org/html/2501.09731v2
- 作者公开 research code: https://github.com/Wenlong2000/RePPI

本文实现为独立教学/审计代码，不是作者官方发行版。
