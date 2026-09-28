> **作者：** 舒清扬 (连享会)
> **邮箱：** <lianxhcn@163.com>

&emsp;

- **Title**: AI 生成的因变量怎样进入回归？RePPI 的原理与 Python 实操
- **Keywords**: 预测驱动推断, 替代结果变量, 交叉拟合, 统计效率

>**提要**: 人工智能 (artificial intelligence，AI) 中的大语言模型 (large language model，LLM) 可以低成本为成千上万条文本生成评分，但这些预测值通常不能直接替代人工真值进入回归。本文介绍再校准预测驱动推断 (recalibrated prediction-powered inference，RePPI)：把模型预测值 (prediction) 当作替代指标 (surrogate)，用少量人工真值学习系统偏差，再借助交叉拟合 (cross-fitting) 完成统计推断。文中给出普通最小二乘法 (ordinary least squares，OLS) 的直觉、适用边界和一套经过单元测试与模拟审计的 Python 实现。

* 来源：Ji, W., Lei, L., & Zrnic, T. (**2026**). Predictions as surrogates: Revisiting surrogate outcomes in the age of AI. *Biometrika*, asag053. [Link](https://doi.org/10.1093/biomet/asag053), [PDF](https://arxiv.org/pdf/2501.09731v2), [Google](<https://scholar.google.com/scholar?q=Predictions+as+Surrogates%3A+Revisiting+Surrogate+Outcomes+in+the+Age+of+AI>), [github](https://github.com/Wenlong2000/RePPI).


<section style="box-sizing:border-box;width:100%;max-width:100%;margin:18px 0;padding:14px 16px;background:#F0FDF4;border-left:4px solid #5A8F69;border-radius:4px;color:#333;line-height:1.9;font-size:1em;overflow-wrap:anywhere;word-break:break-word;">
<div style="margin-bottom:6px;font-weight:600;color:#356342;">代码入口</div>
作者研究代码：<a href="https://github.com/Wenlong2000/RePPI">Wenlong2000/RePPI</a><br>
基础 PPI Python 软件包：<a href="https://github.com/aangelopoulos/ppi_py">aangelopoulos/ppi_py</a><br>
本文配套包：<a href="https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/reppi-ols/downloads/reppi-ols-v0.2.0.zip">下载代码附件 (ZIP)</a>；<a href="https://github.com/lianxhcn/lianxh_blog/tree/main/codes/reppi-ols">浏览配套仓库</a>；<a href="https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/reppi-ols/downloads/reppi-publication-20260928.zip">下载完整材料</a>，内含源代码、示例、测试、算法说明与本地验证日志。该实现用于教学、审计与后续扩展，不是作者官方软件发行版。
</section>

## 1. AI 打出的分数，为什么不能直接替代真实 $Y$？

假设我们想研究企业某种文本特征的决定因素。真实变量 $Y_{i}$ 需要人工阅读全文后给出 1–10 分，成本很高；用 LLM 批量评分却很便宜。于是，一个很自然的方案是：人工标注 300 条，确认 LLM 与人工评分相关性不错，然后让模型把剩余 5,000 条全部打分，最后把 LLM 生成的 $\hat{Y}_{i}$ 当作因变量放进 OLS。

这一步看似合理，却悄悄改变了我们正在估计的对象。

原本的目标是 **真实结果变量 (outcome) 关于 $X$ 的总体 OLS 系数**：

$$
\beta^{*}=\arg\min_{\beta}\mathbb{E}\left[(Y-X'\beta)^{2}\right].
$$

对应的一阶条件是

$$
\mathbb{E}\left[U_{\beta^{*}}(X,Y)\right]=0,
\qquad
U_{\beta}(X,Y)=X(X'\beta-Y).
$$

如果直接把 $Y$ 换成 AI 预测值 $\hat{Y}$，求解的却是

$$
\mathbb{E}\left[X(X'\beta-\hat{Y})\right]=0.
$$

除非 $\hat{Y}$ 与 $Y$ 的误差具有很强的特殊结构，否则这两个参数并不相同。比如，模型可能系统性压缩极端评分，也可能在不同年份、行业或文本风格下发生分布漂移 (distribution shift)。即使整体相关系数很高，或者分类准确率达到 90%，下游回归系数仍可能偏离我们真正关心的参数。

此前连享会的 [「LLM 生成的变量，能直接放进回归吗？」](https://www.lianxh.cn/details/1800.html) 已经讨论过这一类风险。Ji、Lei 和 Zrnic ([2026](https://doi.org/10.1093/biomet/asag053)) 的贡献更进一步：**既然 AI 预测值不是金标准结果变量 (gold-standard outcome)，就不要把它冒充 $Y$；但也没有必要把它全部丢掉。** 可以把它看成一个便宜的替代指标，再用一小部分真实 $Y$ 把替代指标中有用的信息提取出来。

![AI 预测值、人工真值与 RePPI 的关系](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/reppi-fig01-workflow-20260928-171943.png)

> 图注：大量样本只有 $X$ 和 AI 替代指标 $\hat{Y}$，少量样本同时拥有真实 $Y$。直接用 $\hat{Y}$ 替代 $Y$ 会改变目标参数；RePPI 用少量金标准标签 (gold labels) 学习替代指标与真实结果变量的关系，再把大量预测值转化为推断信息。图内流程是本文对方法的教学化整理。

这里需要先划清一个边界。本文讨论的是 **AI 生成或预测的因变量 $Y$**。如果 LLM 生成的是解释变量、处理变量或控制变量 $X$，问题会变成插补协变量 (imputed covariate) 与测量误差 (measurement error)，不能直接套用本文的 RePPI OLS。第 6 节会再回到这个区别。

## 2. 从 PPI 到 RePPI：关键在于怎样利用预测值

Angelopoulos、Bates 等 ([2023](https://doi.org/10.1126/science.adi6000)) 提出的预测驱动推断 (Prediction-Powered Inference，PPI) 的出发点并不复杂。我们有两批来自同一目标总体的数据：

- 有标签样本 (labeled sample) 较小，有 $(X_{i},Y_{i},\hat{Y}_{i})$；
- 无标签样本 (unlabeled sample) 较大，只有 $(X_{j},\hat{Y}_{j})$。

如果只用有标签样本，当然可以做普通 OLS，推断是围绕真实 $Y$ 展开的，但样本小、标准误可能很大。如果只用大样本中的 $\hat{Y}$，精度看起来很高，却可能估错目标。PPI 的基本思想是把两部分组合起来：先利用大量预测值形成一个低方差的近似，再用有标签样本中的 $Y-\hat{Y}$ 修正预测值的系统误差。

用估计方程 (estimating equation) 的语言看，PPI 可以写成

$$
0=\mathbb{P}_{n}U_{\beta}
+\mathbb{P}_{N}h_{\beta}
-\mathbb{P}_{n}h_{\beta},
$$

其中，$\mathbb{P}_{n}$ 和 $\mathbb{P}_{N}$ 分别表示有标签样本与无标签样本的经验平均，$n$ 和 $N$ 为相应样本量；$h_{\beta}=\nabla g_{\beta}$ 是真正进入 PPI 修正项的插补得分 (imputed score)。这里的得分 (score) 指估计方程中的向量函数，与 AI 给文本打出的评分不同。第一项始终用金标准标签锚定目标参数，后两项利用有标签与无标签两批样本上同一个基于替代指标的得分的均值差来降方差。因此，预测值可以很有用，但不会被当成真实 $Y$ 直接替换进去。

这里有一个容易混淆的比例因子。Ji 等先定义原始条件得分

$$
s^{*}_{\beta}(X,\hat{Y})
=
\mathbb{E}\left[U_{\beta}(X,Y)\mid X,\hat{Y}\right],
$$

而理论上最优的 PPI 增广项 (augmentation) 是

$$
h^{*}_{\beta}(X,\hat{Y})
=
\frac{N}{n+N}s^{*}_{\beta}(X,\hat{Y}).
$$

因此，RePPI 实际学习的是 $s^{*}_{\beta}$，再乘以 $N/(n+N)$ 以及一个用于安全调节的矩阵 $M$。这个区分在软件实现中不能省略。

对于 OLS，

$$
U_{\beta}(X,Y)=X(X'\beta-Y),
$$

因此

$$
s^{*}_{\beta}(X,\hat{Y})
=
X\left[X'\beta-\mathbb{E}(Y\mid X,\hat{Y})\right].
$$

这给出了一个很直观的解释：**RePPI 允许 AI 对 $Y$ 的原始预测存在误差。它通过再校准 (recalibration)，学习在当前研究样本中 $Y$ 如何随 $(X,\hat{Y})$ 变化。** 对 OLS 而言，可以把这一步粗略理解为估计 $\mathbb{E}(Y\mid X,\hat{Y})$；对更一般的估计方程，论文则直接学习更合适的插补损失函数或得分。

Angelopoulos、Duchi 和 Zrnic ([2023](https://doi.org/10.48550/arXiv.2311.01453)) 提出的 PPI++ 已经会根据预测值的质量调节预测值在估计中的权重。RePPI 再向前走一步：不仅问“预测值应该用多少”，还问“**预测值应该经过怎样的函数变换后再用**”。这正是再校准带来的额外效率来源。

这也解释了为什么不能只盯着预测值的准确率、均方误差 (mean squared error) 或与 $Y$ 的相关系数。对下游回归而言，更关键的是预测值能否解释估计得分中与参数有关的那部分波动。一个存在明显水平偏差的模型，只要这种偏差能通过 $(X,\hat{Y})$ 被稳定学习，仍然可能给 RePPI 提供大量有效信息；反过来，一个总体均方误差很漂亮的预测值，如果在研究者真正关心的参数方向上几乎没有额外信息，效率增益也可能很小。**预测质量是上游指标，参数估计效率才是下游推断指标。**

因此，RePPI 把辅助函数学习 (nuisance learning) 纳入估计方程，即学习目标参数之外、用于校正的函数。机器学习负责学习“哪里可以借力”，金标准标签负责保证最终目标仍然锚定真实结果变量。这个分工对 LLM 标注尤其合适：模型可以很便宜地覆盖大样本，但研究者仍保留一小块人工标注样本作为统计锚点。

论文给出的理论结论也需要准确理解。在其假设和渐近框架下，即使再校准没有完美学到最优插补损失函数，经过安全的去相关处理和加权后，RePPI 的渐近效率仍不劣于只用真实标签的估计；如果最优插补损失函数能被一致估计，则可以达到该类预测驱动估计量中的最小渐近方差。这里说的是 **渐近方差**，不是“小样本下每次都一定更好”。后面我们的模拟会看到，这个区别不能省略。

## 3. 三折交叉拟合到底在做什么？

如果我们用同一批少量金标准标签既估计初始回归，又训练再校准模型，再用同一批数据评价修正项，很容易把第一阶段过拟合带进最终推断。Ji 等的实现采用三折交叉拟合，把有标签样本分成三份，让三个任务在不同样本上完成：

- 一份估计初始参数 $\hat\beta_{0}$；
- 一份学习再校准，即 $(X,\hat{Y})\rightarrow Y$ 或相应得分；
- 一份计算修正项和本轮 RePPI 参数。

这里每一折就是一个样本组 (fold)，用于计算修正项的一组称为评价折 (evaluation fold)。三折轮换角色，最后按各评价折的样本量占有标签样本的比例加权汇总。三折完全等长时，这一步才与简单平均相同。

![RePPI 的三折交叉拟合](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/reppi-fig02-crossfit-20260928-171943.png)

> 图注：同一个有标签观测在一轮中只承担一种角色。三轮依次轮换初始估计、再校准和评价，最后汇总三个估计。这样做的目的不是“多跑三次提高精度”，而是把辅助函数学习与最终估计方程尽量分开。

以 OLS 为例，记

$$
\alpha_{n}=\frac{N}{n+N}.
$$

在第 $k$ 轮的评价折上，点估计可以写成

$$
\hat\beta_{k}
=
\hat\Sigma_{k}^{-1}
\left[
\frac{1}{n_{k}}\sum_{i\in I_{k}}X_{i}Y_{i}
+
\alpha_{n}\hat M_{k}
\left(
\bar s_{L,k}-\bar s_{U,k}
\right)
\right].
$$

其中，$\hat\Sigma_{k}$ 是评价折上的格拉姆矩阵 (Gram matrix)；$\bar s_{L,k}$ 和 $\bar s_{U,k}$ 分别是再校准得分在有标签评价样本与大规模无标签样本中的平均；$\hat M_{k}$ 由得分与真实梯度的交叉协方差构造。**这里的 $\alpha_{n}=N/(n+N)$ 使用全体有标签与无标签样本量，而不是把 $n$ 换成某个评价折的 $n_{k}$。**

三折得到 $\hat\beta_{1},\hat\beta_{2},\hat\beta_{3}$ 后，算法 1 的最终汇总为

$$
\hat\beta^{CF}=\sum_{k=1}^{3}\frac{n_{k}}{n}\hat\beta_{k}.
$$

2026 年 6 月的 arXiv 第 2 版还直接给出了基于交叉拟合的代入式 (cross-fitted plug-in) 方差估计。记 $\widehat H=n^{-1}\sum_{i=1}^{n}X_{i}X_{i}^{\top}$，其中 $X_{i}$ 包含截距项。若 $\tilde s_{i}$ 是每个有标签观测在未参与训练时得到的原始得分 (尚未乘以 $M$ 或 $N/(n+N)$)，定义 $\widehat V_{\ell}$、$\widehat V_{s}$ 和 $\widehat C_{\ell s}$ 为最终估计值处的真实梯度、上述原始得分的样本协方差及二者的交叉协方差，则

$$
\widehat\Sigma_{RePPI}
=
\widehat H^{-1}
\left[
\widehat V_{\ell}
-
\frac{N}{n+N}
\widehat C_{\ell s}
\widehat V_{s}^{-1}
\widehat C_{\ell s}^{\top}
\right]
\widehat H^{-1}.
$$

第 $j$ 个系数的标准误为 $\sqrt{\widehat\Sigma_{RePPI,jj}/n}$。本文配套代码 v0.2.0 正是按这组公式实现，并在 $\widehat V_{s}$ 接近奇异时使用小幅岭正则化 (ridge regularization)。

这一结构带来一个实际代价：如果只有 90 个金标准标签，三折以后每一轮能用于再校准或评价的样本只有大约 30 个。理论上的“少量标签”并不意味着几十个观测值就一定够用；参数维数、再校准学习器的复杂度和 $Y$ 的噪声都会影响有限样本表现。

## 4. 再校准什么时候最有用？

如果 $\hat{Y}$ 已经与 $Y$ 极其接近，而且误差几乎没有系统结构，那么普通 PPI/PPI++ 已经能把预测值用得很好，RePPI 的增量不会太大。论文重点讨论的是预测值 **系统性偏离**真实结果变量的情形。

一个常见来源是 **模态不匹配** (modality mismatch)。预训练模型只能看到文本或图像，而人工结果变量还利用了模型看不到的信息。第二种是 **分布漂移**：模型在一个训练分布上表现很好，进入新的年份、行业或人群后产生系统偏差。第三种尤其贴近 LLM 标注，即 **离散预测** (discrete prediction)：真实标签是连续值，但模型为了稳定输出，只给出几个离散等级。

这些情形的共同点是：$\hat{Y}$ 仍然包含有用信号，却不能被视为 $Y$ 的无偏复制品。把 $X$ 一起放入再校准，可以利用研究样本自身的信息修正这种结构性偏离。

论文的三个应用很能说明这一点。

在葡萄酒评分应用中，真实评分位于 80–100，但 AI 只输出 5 个有序类别。作者研究价格与真实葡萄酒评分之间的回归关系。在论文给定的实验设定和区间长度下，RePPI 相对 PPI++ 所需的人工标签减少约 **17.6%–21.5%**。这里的改善来自再校准把粗糙的离散预测重新映射到与真实评分更接近的条件结构。

在美国人口普查应用中，作者研究对数收入与年龄等变量的关系，论文补充材料报告的某些设定下，RePPI 相对 PPI++ 的标签需求减少比例约为 **24.8%–26.2%**。在礼貌程度数据中，真实结果变量是多位人工评估者的礼貌程度评分，AI 生成 1–25 的预测值；对应实验中的标签需求减少比例较小，大约 **3.1%–6.6%**。

这些数字应该理解为论文特定任务、模型和区间设定下的结果，而不是“RePPI 一般都能节约 20% 人工标注”。更可靠的判断标准是：**如果你已经发现预测误差会随 $X$、样本群体或预测值水平呈现系统结构，再校准才最有机会带来明显增益。**

在自己的数据上，可以先做一个很朴素的诊断。只看有标签样本，画出 $Y-\hat{Y}$ 对 $\hat{Y}$ 的散点或分箱均值，再按年份、行业、文本长度和几个关键 $X$ 分组比较预测误差。如果误差的条件均值基本没有可学习的结构，更复杂的再校准未必带来额外增益；如果误差随预测值水平呈现压缩、弯曲或分组漂移，那么 $(X,\hat{Y})\rightarrow Y$ 很可能存在可利用结构。本文代码返回的各评价折上的样本外 (out-of-fold) $R^{2}$ 和均方根误差 (root mean squared error)，就是为这类判断准备的，而不是只给出一个最终系数就结束。

这一步还可以反过来帮助安排人工标注预算。金标准标签不只是为了报告“人工与 AI 一致率”，而是用来估计预测误差的结构并校准下游参数。因此，验证样本是否覆盖研究总体、是否包含极端值和关键子群体，比单纯追求一个漂亮的总体准确率更有意义。

## 5. Python 实操：一个可以直接运行的 RePPI OLS

作者提供了公开的 [RePPI 研究代码](https://github.com/Wenlong2000/RePPI)，其中包含 OLS 交叉拟合函数，以及人口普查、礼貌程度和葡萄酒评分等应用的交互式笔记本。它足以帮助理解论文与复现实验，但整体更接近论文研究代码，而不是面向一般实证用户的安装包。

为便于教学和后续审计，本文配套材料另外整理了一份 **独立实现** `reppi-ols-python-v2`。它没有复制作者源码，而是以 2026 年 6 月 arXiv 第 2 版及录用稿算法为基准重新写成一个最小 Python 模块，并额外加入输入检查、固定随机种子、伪逆、协方差正则化、半正定 (positive semidefinite) 数值修正及分折诊断。目前只实现无权重 OLS，尚未支持逻辑回归 (logit)、固定效应、聚类稳健标准误 (cluster-robust standard error) 或 AI 生成的协变量。

这里有一个代码审计中发现的细节。截至 2026-09-27 核验时，作者 GitHub 中的 `reppi.py` 来自较早的研究代码版本，其 OLS 实现会在评价折内重新计算有标签样本与无标签样本的比例，并对三轮估计做等权平均；而论文第 2 版的算法 1 明确使用全样本 $N/(n+N)$，并按评价折大小加权。本文代码以**论文第 2 版的书面算法**为准。配套目录中的 `ALGORITHM-NOTE.md` 保留了这一差异及测试办法，便于后续复核。

### 5.1 一个故意让 AI 预测值出错的例子

模拟数据的真实模型设为

$$
Y_{i}=1+0.5X_{1i}-0.3X_{2i}+\varepsilon_{i}.
$$

我们只给 600 个观测保留真实 $Y$，另外 6,000 个观测只有 AI 替代指标。为了不让例子过于理想，替代指标被故意加入收缩、非线性扭曲和额外噪声：

$$
\hat{Y}_{i}
=0.25+0.75Y_{i}+0.35(X_{1i}^{2}-1)+u_{i}.
$$

这里使用 $Y$ 构造 $\hat{Y}$，是模拟中设定二者联合分布的方式，用于刻画带有相关信息但存在系统误差的预测。实际应用中，预测器不能读取待推断样本的真实 $Y$；估计函数不接收无标签样本的真实 $Y$。

如果直接把 $\hat{Y}$ 当结果变量，$X_{1}$ 的回归系数会从真实的 0.5 被压到约 0.375；在有限样本中会有波动。我们的固定随机种子示例得到以下结果 (展示标签已译为中文，数值保持原样)：

```text
真实系数：        [ 1.0000,  0.5000, -0.3000]
仅人工标签 OLS：   [ 0.9966,  0.5342, -0.2687]
仅替代指标 OLS：   [ 1.0050,  0.3873, -0.2326]
RePPI：           [ 1.0101,  0.5265, -0.3144]
```

这个单次模拟只用于检查程序是否朝合理方向工作，不能用来证明一般效率性质。更重要的是重复模拟中的偏差、标准误校准和置信区间覆盖率 (coverage)。

### 5.2 最小调用

下载[代码附件](https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/reppi-ols/downloads/reppi-ols-v0.2.0.zip) (`reppi-ols-v0.2.0.zip`)，解压后进入 `reppi-ols-v0.2.0/` 目录，在已激活的 Python 环境中执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python example_simulation.py
python validation_monte_carlo.py
```

本地使用 Python 3.11.14 验证，10 项测试全部通过 (原始输出为 `10 passed`)。如提示找不到测试模块 (`No module named pytest`)，请确认安装依赖和运行测试使用同一个 `python`；如找不到示例脚本，请确认已进入解压后的代码目录。测试中另有独立对照计算，专门检查全样本比例、按折样本量加权汇总和第 2 版的代入式方差；测试样本量故意设为不能被 3 整除，以避免“等权平均恰好没出错”的假阳性。在自己的数据中，核心调用只有下面几行：

```python
from reppi_ols import fit_reppi_ols

# X_labeled:      有人工真值的样本对应的解释变量
# y_labeled:      人工标注或金标准结果变量
# yhat_labeled:   同一批样本上的 AI 预测值
# X_unlabeled:    大样本解释变量
# yhat_unlabeled: 大样本 AI 预测值
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

`result.summary()` 同时报告 RePPI 系数、三明治标准误 (sandwich standard error)、置信区间，以及仅使用人工标签、仅使用替代指标的两组 OLS 结果，便于快速检查“预测值是否真的提供了额外信息”。`fold_diagnostics` 还会报告各轮再校准的样本外 $R^{2}$、均方根误差和得分协方差矩阵的条件数 (condition number)。

### 5.3 200 次蒙特卡洛模拟：检查区间覆盖率

我们通过蒙特卡洛模拟 (Monte Carlo simulation) 检查程序表现，重新运行了包内 `validation_monte_carlo.py`，固定 $n=600$、$N=6000$、200 次重复。结果如下。最后一列的分子是 RePPI 的经验标准差，分母是仅使用人工标签的 OLS 的经验标准差：

| 参数 | 95% 置信区间覆盖率 | 经验标准差之比 |
| --- | ---: | ---: |
| 截距 | 0.955 | 0.777 |
| $X_{1}$ | 0.955 | 0.886 |
| $X_{2}$ | 0.955 | 0.767 |

程序报告的标准误与重复模拟得到的经验标准差也比较接近。例如 $X_{1}$ 的经验标准差为 0.0348，平均标准误为 0.0361。这组结果说明，在**这个特定数据生成过程 (data-generating process) 与样本规模下**，程序没有通过牺牲覆盖率来制造更窄区间，同时相对于仅使用人工标签的 OLS 获得了不同程度的效率改善。

![RePPI 蒙特卡洛模拟审计](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/reppi-fig03-monte-carlo-20260928-171943.png)

> 图注：200 次模拟，$n=600$、$N=6000$。柱高是 RePPI 估计量与仅使用人工标签的 OLS 估计量的经验标准差之比，低于 1 表示经验标准差更小；柱内同时标出 95% 置信区间覆盖率。该图来自本文配套代码的实际运行，不是 Ji 等论文中的结果。

需要说明的是，把有标签样本减到 $n=300$、无标签样本减到 $N=3000$ 后，我们用同一版本重新做了 200 次压力测试，三项参数的 95% 置信区间覆盖率分别为 0.960、0.955 和 0.895。第三个系数已经出现明显覆盖率不足。这个现象并不否定论文的渐近理论，反而说明一个实操要点：**三折交叉拟合会消耗金标准标签，有限样本推断必须单独检查。**

## 6. 什么时候不能直接套这套代码？

RePPI 很有用，但它解决的是一个相当明确的问题。下面几种情况如果混在一起，反而容易把“校准预测值”误解成万能的生成变量修正。

### 6.1 AI 生成的是 $X$，不是 $Y$

如果 LLM 生成的是解释变量、处理变量或控制变量，例如“管理层战略激进程度”被放在回归右边，那么我们面对的是插补协变量。Kluger 等 ([2025](https://arxiv.org/abs/2501.18577)) 专门研究这一问题，并讨论先预测后去偏 (Predict-Then-Debias) 及完整观测样本的抽取概率不均匀时如何推断。**当前 `reppi_ols.py` 不支持这种场景。**

<section style="box-sizing:border-box;width:100%;max-width:100%;margin:18px 0;padding:14px 16px;background:#FFFBEB;border-left:4px solid #D6A84B;border-radius:4px;color:#333;line-height:1.9;font-size:1em;overflow-wrap:anywhere;word-break:break-word;">
<div style="margin-bottom:6px;font-weight:600;color:#7A5A13;">先看生成变量在回归哪一边</div>
AI 生成的<strong>结果变量</strong>：RePPI 是直接相关的工具。<br>
AI 生成的<strong>解释变量或协变量</strong>：转向插补协变量或先预测后去偏等框架。<br>
两类问题都不能因为“人工验证准确率很高”就忽略生成变量的不确定性。
</section>

### 6.2 金标准标签不是随机抽的

当前无权重代码要求两批观测来自相同的联合总体分布，例如从目标总体中随机抽取人工标注样本。仅有给定 $X$ 后标签随机缺失 (missing at random，MAR) 并不足够：若标注概率随 $X$ 变化，两批样本的得分均值通常仍不相同。如果人工只挑“模型最没把握的样本”、某些行业或某些年份去标注，有标签样本就不再代表目标总体。此时需要显式处理抽样概率、权重或分层/聚类设计，而不能直接解释当前程序给出的置信区间。

### 6.3 预测模型偷看过本次的真实 $Y$

如果产生 $\hat{Y}$ 的模型使用当前推断样本的真实 $Y$ 做过训练、微调或提示词选择，又把同一批观测当作金标准标签参与最终推断，就可能发生重复使用数据 (double dipping)。最简单的处理是使用真正的外部预训练模型，或者再做一层样本划分或交叉拟合，把预测模型训练与推断数据分开。

### 6.4 论文框架比本文代码宽，软件边界要单独看

Ji 等的理论并不只限于连续结果变量的普通 OLS；论文从一般估计方程和凸损失函数出发讨论 RePPI。但“理论上可以扩展”与“当前这份代码已经正确实现”是两回事。本文配套模块只对连续结果变量的无权重 OLS 做了实现与模拟审计。若改用逻辑回归、泊松回归 (Poisson regression) 或分位数等估计目标，需要相应定义得分、海塞矩阵 (Hessian) 和方差，并设计测试。二元 $Y$ 若仍以总体 OLS 投影 (线性概率模型) 为目标，OLS 得分的形式不变；但本文尚未单独验证该情形的有限样本表现。

### 6.5 面板数据、固定效应与聚类稳健标准误不能靠“先跑起来”代替推导

本文代码目前只实现独立同分布观测下的 OLS 三明治方差推断。经管实证研究中常见的企业年度面板、高维固定效应、组内相关误差、抽样权重都会改变影响函数和方差计算。把 `reghdfe` 的残差化数据塞进当前函数，并不自动等于获得了正确的聚类稳健 RePPI 推断。要用于正式论文，这些扩展应当逐项推导和模拟验证。

### 6.6 RePPI 修正统计推断，不替你定义研究概念

如果人工金标准标签自己就不能稳定测量“创新”“企业文化”“政策支持强度”等概念，RePPI 不能凭空创造构念效度 (construct validity)。提示词操纵 (prompt hacking)、标签标准漂移、不同人群间测量不等价等问题仍然存在。相关问题可参见 [「Prompt-Hacking：比 p-hacking 更隐蔽的显著性幻觉」](https://www.lianxh.cn/details/1804.html)。

同样，RePPI 给出的是某个统计目标的有效估计与推断。若 $X$ 与误差项存在内生性，得到更准确的 OLS 系数也不会自动变成因果效应。**生成的结果变量的测量问题与因果识别是两道不同的关。**

简言之，RePPI 最适合这样的研究流程：先明确金标准结果变量；用便宜的 AI 模型为大样本生成替代指标；随机抽取一部分观测获得真实结果变量；用交叉拟合再校准把预测值转化为推断信息；最后对有限样本、抽样设计与下游识别条件分别做检查。它让研究者利用有误差但有信息的替代指标，同时将目标参数锚定在人工真值上。

## 7. 参考资料

### 7.1 核心文献

1. Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (**2023**). Prediction-powered inference. *Science*, 382(6671), 669–674. [Link](https://doi.org/10.1126/science.adi6000), [PDF](https://clarafy.github.io/data/ABFJZ2023Science.pdf), [Google](<https://scholar.google.com/scholar?q=Prediction-Powered+Inference>).
2. Angelopoulos, A. N., Duchi, J. C., & Zrnic, T. (**2023**). PPI++: Efficient prediction-powered inference. *arXiv:2311.01453*. [Link](https://doi.org/10.48550/arXiv.2311.01453), [PDF](https://arxiv.org/pdf/2311.01453), [Google](<https://scholar.google.com/scholar?q=PPI%2B%2B%3A+Efficient+Prediction-Powered+Inference>).
3. Ji, W., Lei, L., & Zrnic, T. (**2026**). Predictions as surrogates: Revisiting surrogate outcomes in the age of AI. *Biometrika*, asag053. [Link](https://doi.org/10.1093/biomet/asag053), [PDF](https://arxiv.org/pdf/2501.09731v2), [Google](<https://scholar.google.com/scholar?q=Predictions+as+Surrogates%3A+Revisiting+Surrogate+Outcomes+in+the+Age+of+AI>).
4. Kluger, D. M., Lu, K., Zrnic, T., Wang, S., & Bates, S. (**2025**). Prediction-powered inference with imputed covariates and nonuniform sampling. *arXiv:2501.18577*. [Link](https://doi.org/10.48550/arXiv.2501.18577), [PDF](https://arxiv.org/pdf/2501.18577), [Google](<https://scholar.google.com/scholar?q=Prediction-Powered+Inference+with+Imputed+Covariates+and+Nonuniform+Sampling>).

### 7.2 相关连享会推文

- [LLM 生成的变量，能直接放进回归吗？](https://www.lianxh.cn/details/1800.html)。这篇文章讨论生成变量进入下游分析的一般风险；本文进一步聚焦 AI 生成的结果变量的 RePPI 解决方案。
- [Prompt-Hacking：比 p-hacking 更隐蔽的显著性幻觉](https://www.lianxh.cn/details/1804.html)。用于理解预测流程本身怎样成为新的研究者自由度来源。
- [用 AI 做文本分析：从文本标注到回归变量](https://www.lianxh.cn/details/1924.html)。用于衔接从文本标注、人工核验到下游实证变量构造的完整流程。
