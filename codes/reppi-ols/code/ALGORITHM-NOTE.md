# Algorithm adjudication note

Date: 2026-09-15

## 1. Why this note exists

本项目第一次本地 Codex 审计发现：旧版教学实现和作者当前 GitHub `reppi.py` 与 2026 年 6 月修订的论文 Algorithm 1 在两个有限样本实现细节上不一致。该报告已保存在 `00-input/CODEX-REPORT-20260915.md`。

本次裁定以 **arXiv v2 / Biometrika accepted-manuscript algorithm** 为规范基准，而不是以较早公开 research code 的具体行实现为规范。

## 2. Paper v2 requirements

Source: https://arxiv.org/html/2501.09731v2

- Equation (13):
  `M = Cov(grad_l, s) Cov(s)^(-1)`，其中 `M` 本身不包含样本比例缩放。
- Algorithm 1 Step 5:
  linear imputed loss 使用 `1/(1+n/N) = N/(n+N)`。这里的 `n` 是全体 labeled sample size，不是某个 evaluation fold 的大小。
- Algorithm 1 Step 7:
  三轮估计按 `|D_k|/n` 加权汇总，而不是一般意义上的等权平均。
- Remark 5 之后：
  论文直接给出 cross-fitted plug-in covariance，使用全体 labeled sample 上的 out-of-fold score covariance 与 labeled Hessian。

## 3. Difference from the older public research code

Retrieved: 2026-09-15

Source: https://github.com/Wenlong2000/RePPI/blob/main/reppi.py

当前公开 `reppi.py` 中：

- `ppi_opt_ols_pointestimate()` 从传入 evaluation fold 再取 `n = Y.shape[0]`，因此内部比例实际变为 `n_eval/N`；
- 三个 fold-specific estimates 使用简单平均；
- CI 代码沿用较早 influence decomposition，而 v2 论文已经提供更直接的 plug-in variance 公式。

这不意味着作者论文理论有问题。更合理的解释是：公开 repository 是较早 research-code snapshot，而论文在后续修订中明确了算法与 variance estimator。当前 repo 的首个公开 commit 日期为 2025-01-16，而 arXiv v2 修订日期为 2026-06-21。

## 4. v0.2.0 implementation decision

`reppi_ols.py` v0.2.0 做出以下调整：

1. 全部 rotation 固定使用 `augmentation_scale = N/(n+N)`。
2. `M` 按 Equation (13) 计算，不提前吸收比例。
3. fold-specific point estimates 按 evaluation fold 大小除以总 `n` 加权。
4. variance 改为论文 v2 的 plug-in formula。
5. 保存每个 labeled observation 的 out-of-fold score，用于 `V_s` 与 `C_ls`。
6. 继续保留 pseudoinverse、ridge stabilization 与 PSD numerical projection。

## 5. Tests specifically added for this issue

新增测试 `test_v2_algorithm_reference_with_unequal_folds()`：

- 使用独立 reference transcription，不调用主模块的内部 helper；
- 使用 `n=451`，确保三个 fold 大小不完全相等；
- 同时比较 point estimate、variance matrix、global augmentation scale 与 fold weights。

因此，若未来代码再次误用 `n_eval/N` 或恢复简单平均，该测试应直接失败。
