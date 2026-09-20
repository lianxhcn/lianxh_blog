# 连续 DID 系数分解：原理、适用条件与实现

> **作者：** 连小白 (连享会)  
> **邮箱：** <lianxhcn@163.com>

**关键词：** 连续处理 DID，处理强度，组间差异，组内回归系数，平行趋势，Stata，Python

**配套程序：** [Stata 完整示例](https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/continuous-did-decomposition/continuous_did_stata_demo.do) · [Stata 最简示例](https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/continuous-did-decomposition/continuous_did_stata_minimal.do) · [Python 示例](https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/continuous-did-decomposition/continuous_did_decomposition.py)

> **提要：** 将二元处理变量替换为连续的处理强度，不仅保留了处理组内部的强度差异，也改变了回归中的比较基准。此时，回归既比较处理组与控制组，也比较处理组内接受不同强度政策的单位。本文通过一个可以手算的例子，推导连续 DID 系数的分解公式，解释两个组成部分及其权重，并讨论识别条件、适用场景和软件实现。

## 1. 为什么需要分解连续 DID 系数？

研究一项产业扶持政策时，你原本打算比较试点城市与非试点城市的就业变化。整理数据后发现，各试点城市获得的扶持额度并不相同，于是将模型中的「是否试点」替换为「扶持额度」。这样做似乎既保留了更多信息，又能进一步回答：扶持额度越高，就业增长是否越多？

在只有政策前、政策后两个时期的平衡面板中，相应的模型可以写为：

$$
Y_{it}=\alpha_{i}+\gamma_{t}+\beta D_{i}Post_{t}+\varepsilon_{it}.
$$

其中，未处理城市的 $D_{i}=0$，处理城市的 $D_{i}>0$；$Post_{t}$ 在政策后取值为 1，在政策前取值为 0。如果估计得到 $\widehat{\beta}=3.33$，你可能准备写：扶持额度每增加一个单位，就业增加约 3.33 人。

但在作出这种解释前，需要弄清楚：**回归系数是根据哪些单位之间的差异估计出来的？**

对上述模型作两期差分，个体固定效应被消去，模型变为：

$$
\Delta Y_{i}=a+\beta D_{i}+v_{i}.
$$

处理强度 $D_{i}$ 包含两类差异。一类是零值与正值之间的差异，即未处理单位与处理单位之间的差异；另一类是不同正值之间的差异，即处理组内部接受不同强度政策的单位之间的差异。因而，将二元变量替换为连续变量，不只是改变了解释变量的计量单位，也使回归增加了一类比较。

连续处理 DID 文献已经区分了实际处理强度下的平均处理效应，以及提高处理强度的因果效应。Callaway et al. ([in press](https://doi.org/10.1257/aer.20240137)) 还指出，通常的平行趋势假设不能自动排除不同强度组之间的处理收益选择问题。

本文介绍的**连续 DID 系数分解**，就是将原系数所包含的两类比较分别表示出来。研究者可以据此判断原系数主要反映哪一类差异，也可以分别报告两类比较的结果，而不再要求一个系数同时回答两个问题。下面依据 Yu ([2026](https://doi.org/10.48550/arXiv.2609.09488)) 的分解框架，从组间变异和组内变异出发说明其原理。

为便于理解，先讨论两期、观测等权、没有额外协变量的情形。多期设计、协变量调整及统计推断的问题放在后文讨论。文献中的 `dose` 在本文统一译为「处理强度」，不另用「剂量」指代同一个变量。

## 2. 平均政策效应与处理组内的回归系数是两回事

假设四个城市的扶持额度及就业变化如下。处理强度的一个单位代表 100 万元扶持额度，结果变量的单位是新增就业人数。

| 城市 | 处理强度 $D_{i}$ | 就业变化 $\Delta Y_{i}$ |
|---|---:|---:|
| 甲 | 0 | 0 |
| 乙 | 0 | 0 |
| 丙 | 1 | 10 |
| 丁 | 3 | 10 |

为了明确这个例子的因果含义，再作如下设定：如果没有政策，四个城市的就业变化都为零；在所考察的正处理强度范围内，每个城市接受政策后都增加 10 个岗位，提高扶持额度不会再增加就业。

这些是人为指定的数据生成条件，不是根据表中的四个观测推断出来的结论。在真实研究中，两个城市的就业变化恰好相同，并不足以证明提高扶持额度没有作用。

将表中的四个观测画在同一坐标系中，就能看清这两类比较的区别。下图的横轴是处理强度，纵轴是就业变化；甲、乙位于同一个位置 $(0,0)$，丙、丁分别位于 $(1,10)$ 和 $(3,10)$。因此，图上虽然只看到三个位置，回归中仍然有四个观测，原点的两个观测均参与估计。

![同一组四城市数据中的全样本回归与处理组内回归](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/continuous-did-fig01-between-within-20260920-161338.png)

*图中使用上表的教学数据。蓝色虚线只使用两个处理城市估计，橙色实线使用全部四个城市估计；两次回归都包含常数项。*

对照表中的丙、丁两行，沿蓝色虚线从左向右看：扶持额度从 100 万元增加到 300 万元，两个城市的就业变化却同为 10 人。横坐标相差 2，纵坐标相差 0，因此这条线是水平的，处理组内的回归系数为零。

再看处理城市与控制城市的位置。两个处理城市都位于高度 10，两个控制城市都位于高度 0，所以两组平均就业变化相差 10 人。这个**组间均值之差**对应的是两组观测在纵轴上的平均距离，而不是蓝色虚线或橙色实线的斜率。

橙色实线使用全部四个城市，既要拟合原点的两个控制观测，也要拟合位于上方的两个处理观测。零强度与正强度之间存在明显的就业变化差异，因此这条线仍然向上倾斜。按照 OLS 的计算公式，有：

$$
\widehat{\beta}=\frac{\sum_{i}(D_{i}-\overline{D})(\Delta Y_{i}-\overline{\Delta Y})}{\sum_{i}(D_{i}-\overline{D})^{2}}=\frac{10}{3}\approx 3.33.
$$

图中的全样本拟合式为 $\widehat{\Delta Y}=5/3+(10/3)D$。它的截距为 $5/3$，并不经过原点。这不是绘图误差：含常数项的 OLS 最小化全部观测的残差平方和，不要求拟合线穿过任何一个观测点。

现在可以把表格和图形对应起来：表中的 10 人是两组平均变化之差，蓝色虚线的斜率 0 是处理组内的回归系数，橙色实线的斜率 3.33 则是全样本回归系数。**它们描述的是不同的统计关系，不能互相替代。** 因而，问题不是 OLS 计算有误，而是不能把橙色实线向上倾斜，直接解释成对同一个城市提高扶持额度就能增加就业。

这个例子只有两种正处理强度，仍然可以用于说明下文的代数分解。将组内回归系数进一步解释为连续处理强度的平均因果导数，则需要另外的条件。

## 3. 连续 DID 系数如何分解？

### 3.1 将处理强度分为组间部分和组内部分

令处理状态和处理组平均强度分别为：

$$
T_{i}=\mathbf{1}\{D_{i}>0\},\qquad \mu_{D}=E[D_{i}\mid T_{i}=1].
$$

对处理组内的强度进行中心化，定义：

$$
Z_{i}=T_{i}(D_{i}-\mu_{D}).
$$

于是，原处理强度可以表示为：

$$
D_{i}=\mu_{D}T_{i}+Z_{i}.
$$

$\mu_{D}T_{i}$ 将每个处理单位的强度都替换为处理组均值，控制单位仍为零。它保留了两组之间的强度差异，但不再反映处理组内部的强度差异。

$Z_{i}$ 则表示每个处理单位的强度偏离本组均值多少。控制单位的 $Z_{i}=0$，处理组的 $Z_{i}$ 均值也为零，因此它只反映处理组内部的强度差异。

在四城市例子中，$\mu_{D}=2$，上述分解可以直接写成下表。

| 城市 | 原处理强度 $D_{i}$ | 组间部分 $\mu_{D}T_{i}$ | 组内部分 $Z_{i}$ |
|---|---:|---:|---:|
| 甲 | 0 | 0 | 0 |
| 乙 | 0 | 0 | 0 |
| 丙 | 1 | 2 | −1 |
| 丁 | 3 | 2 | 1 |

对照第 2 节的散点图，处理组的平均强度为 2，恰好位于丙、丁两个点的横坐标中间。丙的强度比均值低 1，丁的强度比均值高 1，因此表中的 $Z_{i}$ 分别为 $-1$ 和 $1$。两者的就业变化却同为 10，这正是处理组内回归线为水平线的原因。中心化只是将组内强度差异单独表示出来，并没有改变原数据。

根据这个构造，有：

$$
E[Z_{i}]=0,\qquad \operatorname{Cov}(T_{i},Z_{i})=0.
$$

这里的正交性是指 $T_{i}$ 与 $Z_{i}$ 不相关。因而，我们可以将它们同时放入回归，用两个系数分别描述两类差异，而不必将两者合并为一个系数。

### 3.2 两个组成部分分别表示什么？

考虑下面的总体线性投影：

$$
\Delta Y_{i}=a+\delta_{L}T_{i}+\theta_{R}Z_{i}+e_{i}.
$$

利用前面的正交性，可以得到 $T_{i}$ 的系数：

$$
\delta_{L}=E[\Delta Y_{i}\mid T_{i}=1]-E[\Delta Y_{i}\mid T_{i}=0].
$$

它是处理组与控制组的平均变化之差，原文称为 **level contrast，本文译为「水平对比」**。在四城市例子中，$\delta_{L}=10$，单位是就业人数。

$Z_{i}$ 的系数为：

$$
\theta_{R}=\frac{\operatorname{Cov}(D_{i},\Delta Y_{i}\mid T_{i}=1)}{\operatorname{Var}(D_{i}\mid T_{i}=1)}.
$$

它等于在处理组内，将结果变化量对常数项和处理强度进行 OLS 回归时，处理强度对应的回归系数。原文将其命名为 **response index，本文译为「响应指数」**。下文在解释其统计含义时，也将它称为「处理组内的回归系数」；这是对其计算方式的描述，不是另外定义一个参数。四城市例子中的 $\theta_{R}=0$，单位是「就业人数/处理强度单位」 (Yu, [2026](https://doi.org/10.48550/arXiv.2609.09488), 第 2.2 节)。

回看第 2 节的图，$\delta_{L}$ 对应两组观测的平均高度差，$\theta_{R}$ 对应蓝色虚线的斜率；橙色实线的斜率则是尚待分解的 $\beta$。三个符号因此各有明确的图形含义。

使用「对比」和「指数」，是为了避免在尚未讨论识别条件时，就将统计量称为因果效应。两个统计量都可以直接根据数据计算，但是否具有因果含义，需要分别论证。

线性投影也不要求真实的条件均值函数一定是线性的。如果结果变化与处理强度之间存在非线性关系，$\theta_{R}$ 仍然有定义，但它只概括其中的线性关联，不能完整描述整条反应曲线。

### 3.3 为什么原回归只能得到一个加权平均？

将 $D_{i}=\mu_{D}T_{i}+Z_{i}$ 代入原模型的拟合式：

$$
a+\beta D_{i}=a+\beta\mu_{D}T_{i}+\beta Z_{i}.
$$

这意味着，原模型用同一个 $\beta$ 决定两个部分的拟合结果：处理组与控制组的拟合均值相差 $\beta\mu_{D}$，处理组内的拟合斜率为 $\beta$。

更明确地说，如果将同时包含 $T_{i}$ 和 $Z_{i}$ 的模型写成 $a+b_{L}T_{i}+b_{R}Z_{i}$，原来的单一强度回归就相当于施加了下面的约束：

$$
b_{L}=\mu_{D}b_{R}.
$$

不加这个约束时，两个系数分别为 $\delta_{L}$ 和 $\theta_{R}$。它们不一定满足上述比例关系。例如，在四城市例子中，一个系数为 10，另一个为零。因此，原模型无法同时保持两个不受约束的拟合结果，只能根据两类变异的大小确定最终系数。

令 $p=P(T_{i}=1)$、$\rho=1-p$，并将处理组内部的强度方差记为 $V_{D}=\operatorname{Var}(D_{i}\mid T_{i}=1)$。由变量定义可得：

$$
\operatorname{Var}(T_{i})=p\rho,\qquad \operatorname{Var}(Z_{i})=pV_{D}.
$$

利用 $T_{i}$ 与 $Z_{i}$ 的正交性，可以将原回归系数写为：

$$
\beta=\frac{\mu_{D}\delta_{L}\operatorname{Var}(T_{i})+\theta_{R}\operatorname{Var}(Z_{i})}{\mu_{D}^{2}\operatorname{Var}(T_{i})+\operatorname{Var}(Z_{i})}.
$$

代入两个方差并约去 $p$，得到分解公式：

$$
\beta=\lambda\theta_{R}+(1-\lambda)\frac{\delta_{L}}{\mu_{D}},\qquad \lambda=\frac{V_{D}}{V_{D}+\rho\mu_{D}^{2}}.
$$

这就是 Yu ([2026](https://doi.org/10.48550/arXiv.2609.09488), 定理 3) 给出的两期分解。推导只使用了变量定义和线性投影的性质，因此它是一个**统计恒等式，不以平行趋势成立为前提**。

原系数是 $\theta_{R}$ 和 $\delta_{L}/\mu_{D}$ 的加权平均。将水平对比除以平均处理强度，是为了让两项具有相同的计量单位，不能因此就把 $\delta_{L}/\mu_{D}$ 解释为提高一单位处理强度的因果效应。

样本分解也有相同形式。使用样本均值、样本中的控制组比例，以及分母为处理组样本数的经验方差，可以精确重构样本 OLS 系数；计算机运算中只需容许很小的浮点误差。

将第 2 节的四城市数据代入，处理组平均强度为 2，组内经验方差为 1，控制组占比为 $1/2$，所以响应指数的权重为 $\widehat{\lambda}=1/3$。原系数可以重构为 $\widehat{\beta}=(1/3)\times0+(2/3)\times(10/2)=10/3$。蓝色虚线对应的组内系数虽然为零，归一化水平对比仍为 5；两项加权后，就得到橙色实线的斜率 3.33。这说明了图中那条向上倾斜的线是怎样形成的。

### 3.4 与直接报告一个连续系数相比，分解增加了什么？

二元 DID 报告处理组与控制组的平均变化之差。连续强度回归则利用同一条拟合线，同时描述两组之间的差异和处理组内部的强度关系。分解方法将这两类信息分别报告，并明确说明原系数怎样对二者加权。

它并不要求使用比 OLS 更复杂的算法。水平对比可以直接用两组均值差计算，响应指数可以通过处理组内回归获得；也可以利用前面的中心化变量，在同一个回归中估计二者。

不过，分解没有提供新的外生变异。它能说明一个系数包含什么，却不能自动解决未处理趋势不同、处理收益选择或处理强度测量误差等问题。是否具有因果解释，仍取决于研究设计。

## 4. 如何解读分解结果和权重？

### 4.1 相同的连续 DID 系数可能对应不同的组间和组内关系

保留四个城市的处理强度 $0,0,1,3$，只改变两个处理城市的就业变化，可以得到以下三种结果。

| 教学设定 | 两个处理城市的就业变化 | $\delta_{L}$ | $\theta_{R}$ | $\beta$ |
|---|---|---:|---:|---:|
| 处理效应不随正强度变化 | 10、10 | 10 | 0 | 3.33 |
| 处理效应与强度成正比 | 2、6 | 4 | 2 | 2.00 |
| 组间差异与组内系数异号 | 14、6 | 10 | −4 | 2.00 |

第二行和第三行的原连续系数都等于 2，但水平对比和组内回归系数并不相同。下图将这两行情形分别放在上、下两组中，展示各自进入分解公式的两个加权项。由于两种情形的处理强度和样本比例相同，处理组平均强度均为 $\mu_{D}=2$，响应指数的权重均为 $\lambda=1/3$。

![表中第二行与第三行对应的加权组成部分](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/continuous-did-fig02-same-beta-20260920-161338.png)

*蓝色条形表示组内加权项，橙色条形表示水平加权项，菱形表示两项之和。横轴上的数值都是进入原系数的加权值，不是表中未经加权的水平对比或响应指数。*

看图的上半部分，它对应表中的第二行。表中的 $\theta_{R}=2$ 乘以 $1/3$，得到蓝色条形所表示的 $2/3$；$\delta_{L}=4$ 先除以平均强度 2，再乘以 $2/3$，得到橙色条形所表示的 $4/3$。两个条形都从零向右延伸，两项相加为 2，因此菱形位于横轴的 2 处。若每个单位的处理效应都满足 $\tau_{i}(d)=2d$，并且相应的趋势条件成立，两个未加权的统计量 $\theta_{R}$ 与 $\delta_{L}/\mu_{D}$ 就都等于 2，无论权重如何变化，原系数都保持为 2。

图的下半部分对应表中的第三行。处理城市平均比控制城市多增加 10 个岗位，但处理组内扶持额度较高的城市反而增加了较少的就业，所以水平对比为正，组内系数为负。相应的两个加权项为：

$$
\lambda\theta_{R}=\frac{1}{3}\times(-4)=-\frac{4}{3},\qquad (1-\lambda)\frac{\delta_{L}}{\mu_{D}}=\frac{2}{3}\times5=\frac{10}{3}.
$$

因此，蓝色条形从零向左延伸到 $-4/3$，橙色条形向右延伸到 $10/3$。两项相加仍然等于 2，菱形便与上半部分的菱形处于同一横坐标。两个菱形说明原系数相同，条形的方向和长度则说明这个相同的系数来自不同的组间与组内关系。

阅读时，需要将表格中的统计量与图中的加权项区分开。例如，第三行的水平对比是 10，但进入原系数的水平加权项为 $10/3$，两者并不矛盾。**表格便于查看原始统计量，条形图则展示这些统计量如何经过加权形成原系数。** 图中的正负方向只表示代数分解，不代表已经识别了不同经济机制的贡献。

同样，$\theta_{R}=0$ 也不一定意味着处理强度与结果完全无关。例如，正处理强度为 $1,2,3$，相应的结果变化为 $10,14,10$，组内线性回归系数为零，但两者显然存在非线性关系。因此，报告响应指数时，还应查看处理组内的散点图或分箱均值，判断单个线性系数能否合理概括数据中的关系。

### 4.2 混合权重由处理组内的强度变异和控制组占比决定

将权重写成变异系数的形式，更容易理解：

$$
\lambda=\frac{CV_{D}^{2}}{CV_{D}^{2}+\rho},\qquad CV_{D}=\frac{\sqrt{V_{D}}}{\mu_{D}}.
$$

在控制组占比不变时，处理组内部的强度相对差异越大，$\lambda$ 越大，响应指数在原系数中的权重就越高。相反，如果处理单位获得的扶持额度都很接近，原系数主要反映的就可能是处理组与控制组之间的差异。

在处理组的强度分布不变时，控制组占比越高，$\lambda$ 越小。回到第一个例子，将控制城市从 2 个增加到 6 个，并保持新增城市的强度与就业变化均为零，可以得到：

| 控制城市数量 | $\rho$ | $\lambda$ | $\delta_{L}$ | $\theta_{R}$ | $\beta$ |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.50 | 1/3 | 10 | 0 | 3.33 |
| 6 | 0.75 | 1/4 | 10 | 0 | 3.75 |

下图将上表两行结果标在同一条曲线上。这里的横轴已经不是城市的处理强度，而是全样本中的控制组占比 $\rho$；纵轴也不是就业变化，而是用整个样本估计得到的连续 DID 系数 $\beta$。图中的两个圆点分别对应上表的 2 个控制城市和 6 个控制城市。

![控制组占比表中的两种样本构成及其连续 DID 系数](https://fig-lianxh.oss-cn-shenzhen.aliyuncs.com/continuous-did-fig03-control-share-20260920-161338.png)

*两个标记点对应上表的样本设定。曲线表示保持两组各自的结果分布和处理组内的强度分布不变、仅改变控制组占比时，分解公式给出的系数变化关系。*

从左侧圆点看起：两个处理城市加上两个控制城市，控制组占比为 $\rho=0.50$，响应指数的权重为 $1/3$。增加四个同样位于 $(0,0)$ 的控制城市后，样本移到右侧圆点，控制组占比变为 $\rho=0.75$，响应指数的权重降为 $1/4$，水平项的权重则由 $2/3$ 升至 $3/4$。

在这个过程中，表中的水平对比始终为 10，组内回归系数始终为零。由于归一化水平对比为 $10/2=5$，两个圆点的纵坐标分别等于 $(2/3)\times5=10/3$ 和 $(3/4)\times5=15/4$，即约 3.33 和 3.75。**系数上升来自混合权重的变化，而不是处理城市的就业增幅扩大。** 这也解释了为什么表中两个分项系数没有变化，曲线上的两个圆点却处于不同高度。

本例中，归一化水平对比 5 大于组内回归系数 0，所以前者的权重上升会推高原系数。若两者的大小关系相反，原系数的变化方向也会相反；不能将图中的上升曲线理解成所有连续 DID 研究都具有的规律。

这个算例提示我们：调整控制组之后，不应只比较 $\widehat{\beta}$，还应分别检查 $\delta_{L}$、$\theta_{R}$ 和 $\lambda$。在真实研究中，新增控制单位还可能具有不同的平均趋势，因此系数变化的来源可能不止权重这一项。样本选择应当依据研究设计，而不是为了得到某个系数值而调整控制组。

### 4.3 代数权重不能解释为经济机制的贡献比例

$\lambda=1/3$ 表示响应指数在分解公式中的权重，不能据此声称三分之一的政策效应来自提高处理强度。两个加权项可能异号，也可能相互抵消，将它们解释为经济机制的贡献比例并不合适。

由于两项权重均非负且合计为 1，原系数位于 $\theta_{R}$ 与 $\delta_{L}/\mu_{D}$ 之间。但这只是加权平均的代数性质，并不保证其中任何一项具有因果含义。

计量单位的变化也不能解决这个问题。例如，将扶持额度从「百万元」改为「万元」，相当于将处理强度乘以 100。此时，$\lambda$ 和 $\delta_{L}$ 不变，$\theta_{R}$、$\delta_{L}/\mu_{D}$ 以及 $\beta$ 均缩小为原来的百分之一。变化的是单位，不是回归所包含的信息。

## 5. 什么条件支持因果解释？

### 5.1 水平对比：处理组与控制组的平均未处理趋势应相同

将没有政策时的结果变化，以及政策后接受强度 $d$ 的处理效应分别记为：

$$
U_{i}=Y_{i1}(0)-Y_{i0}(0),\qquad \tau_{i}(d)=Y_{i1}(d)-Y_{i1}(0).
$$

在一致性和无预期效应条件下，处理单位满足 $\Delta Y_{i}=U_{i}+\tau_{i}(D_{i})$，控制单位的结果变化为 $U_{i}$。代入水平对比的定义，得到：

$$
\delta_{L}=E[\tau_{i}(D_{i})\mid T_{i}=1]+E[U_{i}\mid T_{i}=1]-E[U_{i}\mid T_{i}=0].
$$

如果满足水平平行趋势条件：

$$
E[U_{i}\mid T_{i}=1]=E[U_{i}\mid T_{i}=0],
$$

后两项相互抵消，水平对比就等于实际处理强度下的平均处理效应：

$$
\tau_{L}=E[\tau_{i}(D_{i})\mid T_{i}=1].
$$

它回答的是：各城市按照现实中获得的额度接受政策，相对于这些城市都不接受政策，平均产生了多大的影响。

这不是给所有城市发放同一平均额度的效果，也不是从零强度略微提高一点所产生的效果。因此，在解释 $\delta_{L}$ 时，需要保留「实际处理强度」这一限定，而不能笼统地将其称为参与政策的边际效应。

由此也可以看出，将正强度单位编码为处理组，并不必然意味着丢弃了有用信息。如果研究目标就是现行方案的平均影响，二元 DID 所估计的组间平均差异可能恰好对应这个目标。但将正强度单位按中位数切分为高、低两组，是另一种研究设计，不能与零强度和正强度之间的比较混为一谈。

### 5.2 响应指数：还需排除处理组内部与强度相关的未处理趋势

对响应指数作同样的代入，可以得到：

$$
\theta_{R}=\frac{\operatorname{Cov}(D_{i},U_{i}\mid T_{i}=1)}{V_{D}}+\frac{\operatorname{Cov}(D_{i},\tau_{i}(D_{i})\mid T_{i}=1)}{V_{D}}.
$$

第一项表示处理强度与未处理趋势之间的相关性。例如，原本发展较快的城市可能获得更多扶持，即使没有政策，其就业也会增长得更多。

响应平行趋势要求：

$$
E[U_{i}\mid D_{i},T_{i}=1]=E[U_{i}\mid T_{i}=1].
$$

这个条件约束的是处理组内部的关系：没有政策时的平均结果变化，不应随处理强度系统变化。对当前线性回归系数而言，$\operatorname{Cov}(D_{i},U_{i}\mid T_{i}=1)=0$ 就足以消除第一项；条件均值不随强度变化是一个更强的限制 (Yu, [2026](https://doi.org/10.48550/arXiv.2609.09488), 第 3 节)。

两种平行趋势并不互相包含。假设控制组的未处理变化为零，处理组满足 $U_{i}=b(D_{i}-\mu_{D})$。处理组的平均未处理变化也为零，因此水平平行趋势成立；但只要 $b\neq0$，处理组内部的趋势就随强度变化。反过来，如果所有处理单位的未处理变化都等于同一个非零常数，组内趋势不随强度变化，却与控制组的平均趋势不同。

因此，利用政策前数据进行诊断时，应分别考察处理组与控制组的平均变化之差，以及处理组内结果变化对未来处理强度的回归系数。两个检验针对不同的限制。不拒绝某个零假设，也不能证明政策后的识别条件成立。前趋势检验还可能存在检验力不足，以及依据检验结果筛选分析所引起的推断问题 (Roth, [2022](https://doi.org/10.1257/aeri.20210236))。

### 5.3 相同趋势不能排除不同强度组之间的处理收益选择

即使上式中的第一项为零，第二项也未必等于对同一个单位提高处理强度的因果效应。

考虑一个纯粹用于说明问题的企业例子。企业的项目吸收能力为 $A_{i}>0$，高能力企业获批的名义额度更高，即 $D_{i}=A_{i}$。假设项目收益主要来自入选后获得的固定技术服务：在所考察的正强度范围内，企业 $i$ 获得的收益始终为 $A_{i}$，进一步提高名义额度不会增加收益。所有企业在没有政策时的结果变化都为零。

于是，观测数据满足 $\Delta Y_{i}=A_{i}=D_{i}$，处理组内的回归系数等于 1。但是，对同一家企业提高额度，其收益仍然为 $A_{i}$，对应的因果响应为零。

这里的问题不是高能力企业原本增长得更快，因为未处理趋势已经设定为零。问题在于：**从政策中获益更多的企业，同时获得了更高的处理强度**。这种分配机制被称为基于处理收益的选择 (selection on gains)。它不同于未处理趋势差异，不能仅靠平行趋势假设予以排除 (Callaway et al., [in press](https://doi.org/10.1257/aer.20240137))。

因此，响应指数不能直接改称「处理强度的边际效应」。只有在另外约束处理收益选择，并满足相应的光滑性和支持集条件后，才能讨论其作为某种加权平均因果导数的解释。局部因果响应也不能不加限制地用于预测补贴翻倍等大幅调整的效果。

## 6. 适用场景和使用要点

### 6.1 先根据研究问题选择需要报告的参数

考虑一项三年期企业技术改造项目。企业在获批时确定固定扶持额度，研究期间内额度不变；另有一部分企业尚未获批。研究者可以提出两个问题：现行资助方案是否提高了处理企业的生产率？对已经获批的企业增加额度，是否还能进一步提高生产率？

前一个问题对应实际处理强度下的平均处理效应，后一个问题涉及改变强度的因果响应。分别报告水平对比和响应指数，可以将两类统计关系呈现出来，但不能代替识别论证。尤其当项目质量决定资助额度时，第二个问题需要认真讨论处理收益选择。

预先确定的减排目标、固定期限内的税收优惠幅度，也可以考虑采用这种分解。判断是否适用时，应检查处理强度的含义、政策实施时间和处理路径，而不能只根据政策领域作决定。

政策前暴露程度也常被用于衡量连续处理强度。不过，暴露程度更高的地区出现更大的结果变化，不一定意味着主动提高政策力度会产生同样的效果。能否作出后一种解释，取决于暴露指标能否对应一个定义清楚的干预。

### 6.2 错位实施时，应先确定每个批次—时期的比较样本

如果不同单位在不同时间开始接受政策，不能将前面的两期公式直接套用于任意一个多期 TWFE 系数。

设 $G_{i}$ 为首次处理时期，$D_{i}$ 为开始处理后保持不变的强度。考察第 $g$ 期开始处理的单位在第 $t$ 期的结果时，定义：

$$
\Delta Y_{i,g,t}=Y_{it}-Y_{i,g-1}.
$$

当前样本只保留 $G_{i}=g$ 的处理单位，以及 $G_{i}>t$ 的尚未处理单位。当前比较中使用的处理强度为：

$$
D_{i}^{(g,t)}=D_{i}\mathbf{1}\{G_{i}=g\}.
$$

例如，考察 2020 年首次处理城市在 2022 年的表现，应以 2019 年为基期。2021 年已经开始接受政策的城市不能作为控制组；2023 年才开始接受政策的城市可以进入当前控制组，但其当前处理强度应记为零，不能填入它未来将接受的正强度。

对每个批次—时期样本分别计算水平对比、响应指数和混合权重，再按照明确规则汇总为事件时间结果。这里采用的是开始处理后强度固定的设计，与 Yu ([2026](https://doi.org/10.48550/arXiv.2609.09488), 第 2、6 节) 的基准框架一致。

如果强度逐年调整，特别是根据当期结果进行调整，就需要另行选择适合这种处理路径的方法。还应区分两种分解：Goodman-Bacon ([2021](https://doi.org/10.1016/j.jeconom.2021.03.014)) 分解的是错位实施 TWFE 系数包含的不同两组两期比较；本文分解的是一个给定两期样本中，组间差异与处理组内部强度关系在回归系数中的作用。

### 6.3 数据处理、协变量调整和统计推断需要分别检查

零强度必须确实代表当前未处理。额度缺失、没有记录和额度等于零不是同一种情况。原回归与各分项估计还应使用同一个有效样本，否则重构值与原系数不一致，可能只是因为不同计算步骤使用了不同观测。

处理组内部也必须存在强度变异。如果所有处理单位都接受同一正强度 $d_{0}$，原回归仍可估计，且 $\beta=\delta_{L}/d_{0}$；但响应指数的分母为零，因而**无法定义，而不是等于零**。如果全部单位都已接受正强度处理，则可以估计组内系数，却没有用于计算相对于零处理的水平对比的控制组。

加入协变量以后，不能直接沿用未经调整的分解权重。将强度对协变量回归所得的残差，不再保持「控制组为零、处理组为正」的结构。使用残差化强度估计的组内系数，描述的是协变量无法解释的强度差异与结果变化之间的关系，一般对应不同的目标参数 (Yu, [2026](https://doi.org/10.48550/arXiv.2609.09488), 第 7 节)。

点估计与统计推断也应区分。多个批次—时期样本可能共享单位和控制组，汇总权重本身也可能需要估计。不能将各项标准误当作相互独立后简单合并。下文的程序用于计算两期点估计和核对分解恒等式，不提供多期联合推断，也不使用四个城市进行有实证意义的显著性检验。

## 7. 有没有现成的软件包？

这里需要区分「估计连续处理 DID」与「直接计算本文的两部分分解」。前者已有公开软件，但不能因为软件处理的是连续变量，就认为它直接实现了本文的分解。

截至 2026 年 9 月 20 日，本次对论文页面、公开代码线索和相关软件文档的检索，尚未找到专门实现 Yu 两部分分解及其完整联合推断的官方公开软件包。这是本次检索的结果，不表示作者没有编写程序，也不排除后续发布软件。

与本文相关、但估计对象或研究设计不同的软件包括以下几种。

| 软件 | 语言 | 主要用途 | 与本文分解的关系 |
|---|---|---|---|
| `contdid` | R | 估计不同处理强度下的 ATT、平均因果响应及事件研究结果 | 官方文档未将其定义为本文的两部分系数分解程序 |
| `did_multiplegt_stat` | Stata、R | 在存在处理强度保持不变单位的设计中，估计静态处理效应和平均斜率等参数 | 比较的是处理发生变化与保持不变的单位，不等于本文的组内响应指数 |
| `did_multiplegt_dyn` | Stata、R | 针对允许处理强度多次变化的设计，估计动态效应和事件研究结果 | 处理路径和目标参数不同，不直接输出本文的分解 |

`contdid` 的 CRAN 文档明确区分 `target_parameter="level"` 与 `target_parameter="slope"`，后者对应 ACRT，即平均因果响应，而不是本文定义的处理组内 OLS 系数。本次核对的 CRAN 版本为 0.1.1。`did_multiplegt_stat` 的文档要求存在处理不变的单位，并对处理的滞后作用作出限制；`did_multiplegt_dyn` 则允许更一般的动态处理路径。具体使用时，需要分别阅读它们的识别条件和参数说明 (Callaway et al., [2026](https://cran.r-project.org/web/packages/contdid/refman/contdid.html); Credible-Answers, [n.d.-a](https://github.com/Credible-Answers/did_multiplegt_dyn), [n.d.-b](https://github.com/Credible-Answers/did_multiplegt_stat))。

对于本文的两期、等权、无协变量分解，不必等待专门命令。所需的均值、方差和回归系数，都可以用 Stata 自带命令计算。下面给出 Stata 示例，再保留 Python 实现供交叉核验。

## 8. Stata 实现：计算两个组成部分并重构原系数

### 8.1 数据准备与计算步骤

每个单位保留一行。`dose` 存放当前比较中的处理强度，`dy` 存放相同前后两期的结果变化量。本例沿用前文的四城市数据，而不是将不相关的示例数据人为解释为政策实验。

1. **Step 1**: 检查数据，区分零强度控制组和正强度处理组。
2. **Step 2**: 将 `dy` 对 `dose` 回归，保存原连续 DID 系数。
3. **Step 3**: 计算处理组的平均强度和经验方差，并构造组内中心化变量 `z`。
4. **Step 4**: 将 `dy` 对 `treated` 和 `z` 回归，分别读取水平对比和响应指数。
5. **Step 5**: 计算混合权重，使用分解公式重构原系数，再检查两种结果是否一致。

这段代码只使用 Stata 自带的 `regress`、`summarize`、`generate`、`scalar` 等命令，不需要安装第三方程序。`regress` 默认包含常数项，与前文的线性投影一致 (StataCorp, [n.d.](https://www.stata.com/manuals/rregress.pdf), regress)。

代码会清空内存中的数据，运行前应先保存正在使用的数据。

```stata
version 17.0
clear all
set more off

* 每个城市保留一行；dy 是相同前后两期的结果变化量。
input byte id double dose double dy
1 0  0
2 0  0
3 1 10
4 3 10
end

* Step 1: 检查输入，并识别处理组和控制组。
assert !missing(id, dose, dy)
assert dose >= 0
isid id
generate byte treated = (dose > 0)
quietly count if treated == 0
assert r(N) > 0
quietly count if treated == 1
assert r(N) >= 2

* Step 2: 估计原连续 DID 系数。
quietly regress dy dose
scalar beta = _b[dose]

* Step 3: 计算处理组内的均值与方差。
* r(Var) 的分母为 n1-1；分解公式需要分母为 n1 的经验方差。
* 此处不要使用 meanonly，否则 summarize 不计算方差。
quietly summarize dose if treated == 1
scalar n1   = r(N)
scalar mu_d = r(mean)
scalar v_d  = r(Var) * (scalar(n1) - 1) / scalar(n1)
assert scalar(v_d) > 0 & scalar(v_d) < .

quietly count if treated == 0
scalar rho = r(N) / _N

* Step 4: 在处理组内部中心化处理强度，再估计两个组成部分。
* treated 的系数等于水平对比，z 的系数等于响应指数。
generate double z = treated * (dose - scalar(mu_d))
quietly regress dy treated z
scalar delta_l = _b[treated]
scalar theta_r = _b[z]

* Step 5: 计算权重、两个加权项，以及重构系数。
scalar lambda_r = scalar(v_d) / ///
    (scalar(v_d) + scalar(rho) * scalar(mu_d)^2)
scalar level_part = (1 - scalar(lambda_r)) * ///
    scalar(delta_l) / scalar(mu_d)
scalar response_part = scalar(lambda_r) * scalar(theta_r)
scalar beta_rebuilt = scalar(level_part) + scalar(response_part)

* 检查直接估计与分解重构是否一致。
assert abs(scalar(beta) - scalar(beta_rebuilt)) < 1e-10

* 只展示点估计，不使用四个观测进行显著性检验。
display "原连续 DID 系数 = " %9.4f scalar(beta)
display "水平对比        = " %9.4f scalar(delta_l)
display "组内回归系数    = " %9.4f scalar(theta_r)
display "组内系数权重    = " %9.4f scalar(lambda_r)
display "分解重构系数    = " %9.4f scalar(beta_rebuilt)
```

### 8.2 结果应该怎样阅读？

对于四城市数据，本次 Stata 运行结果为：原连续 DID 系数 3.3333，水平对比 10，处理组内回归系数 0，响应指数权重 0.3333，分解重构系数 3.3333。配套的 `continuous_did_stata_demo.do` 还按相同步骤计算了四种情形，并为每种情形设置数值一致性断言。

> **运行环境与核验范围：** 示例已在 Windows、Stata 19.5 中实际运行，四种情形的断言均通过，结果与 Python 及精确数值基准一致。完整程序、最简程序和正文代码均已核验。本例只检验点估计和分解恒等式，不检验因果识别条件，也不进行统计推断。`version 17.0` 是代码的版本声明，本次未在 Stata 17 中实测。

| 情形 | 原系数 $\widehat{\beta}$ | 水平对比 $\widehat{\delta}_{L}$ | 组内系数 $\widehat{\theta}_{R}$ | 权重 $\widehat{\lambda}$ | 重构系数 |
|---|---:|---:|---:|---:|---:|
| 处理效应不随正强度变化 | 3.3333 | 10.0000 | 0.0000 | 0.3333 | 3.3333 |
| 处理效应与强度成正比 | 2.0000 | 4.0000 | 2.0000 | 0.3333 | 2.0000 |
| 组间差异与组内系数异号 | 2.0000 | 10.0000 | −4.0000 | 0.3333 | 2.0000 |
| 增加控制组 | 3.7500 | 10.0000 | 0.0000 | 0.2500 | 3.7500 |

先看第一行：水平对比为 10，组内系数为零，原连续系数为 3.3333。这分别对应第 2 节散点图中的两组平均高度差、蓝色虚线的斜率和橙色实线的斜率。程序计算的统计量与图中的几何关系是一致的。

第二行与第三行对应第 4.1 节条形图的上、下两组。原系数都为 2，但组内系数分别为 2 和 $-4$。将这些系数按表中的权重加权后，才得到图中的条形长度；不能将软件输出的未加权系数直接当作条形数值。

第一行与第四行则对应第 4.2 节曲线上的两个圆点。处理组数据没有改变，水平对比和组内系数也没有改变；控制组占比上升使响应指数的权重下降，原系数随之从 3.3333 变为 3.7500。每一行的最后一列均与原系数一致，用于检验代码是否正确实现了分解公式。

完整 do-file 运行后，会在 `results_stata` 文件夹下生成 Stata 日志和 `stata_results.csv`。如果断言失败，应检查样本、方差分母及中心化方式，不应删除断言后继续将结果作为正确输出。

### 8.3 两个容易忽略的实现细节

在 Stata 中，无权重 `summarize` 返回的 `r(Var)` 使用 $n_{1}-1$ 作为分母，而样本分解需要的是以处理组样本数 $n_{1}$ 为分母的经验方差。因此，代码使用：

$$
\widehat{V}_{D}=\frac{n_{1}-1}{n_{1}}s_{D}^{2}.
$$

不能将 `r(Var)` 不作调整就代入样本权重公式。另外，`summarize, meanonly` 不计算方差，所以提取 `r(Var)` 时不能使用该选项 (StataCorp, [n.d.](https://www.stata.com/manuals/rsummarize.pdf), summarize)。

另一个细节是中心化。`regress dy treated z` 中的 `treated` 系数等于水平对比，是因为 `z` 使用处理组平均强度进行中心化。如果改成 `regress dy treated dose`，`treated` 的系数就不再直接等于水平对比；还需加上平均处理强度乘以 `dose` 的系数。

这些等价关系只说明点估计的计算方式。以样本均值作为中心、按估计权重汇总结果，以及多个比较样本之间的相关性，都可能影响推断。不能仅给联合回归加上 `vce(robust)`，就宣称已完成原文的全部联合推断。

## 9. Python 实现与数值核验

### 9.1 计算函数

Python 实现与 Stata 使用相同的统计量。下面的函数还检查了缺失值、负强度、控制组缺失及组内强度方差为零等输入问题。它不会悄悄删除观测，而是要求使用者先确定统一的有效样本。

安装 `NumPy` 后即可运行：

```bash
python -m pip install numpy
```

`var(ddof=0)` 计算的正是分母为样本数的经验方差，与前面经调整的 Stata 方差对应 (NumPy Developers, [n.d.](https://numpy.org/doc/stable/reference/generated/numpy.var.html), numpy.var)。

```python
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
```

### 9.2 运行示例并核对结果

将下面的代码接在函数后，即可计算四城市例子中的各项结果。

```python
r = decompose_did(
    dose=[0, 0, 1, 3],
    dy=[0, 0, 10, 10],
)
for key in ("beta", "delta_L", "theta_R", "lambda_R", "beta_rebuilt"):
    print(f"{key:>12s} = {r[key]:.4f}")
```

本次 Python 实际运行结果为：

```text
        beta = 3.3333
     delta_L = 10.0000
     theta_R = 0.0000
    lambda_R = 0.3333
beta_rebuilt = 3.3333
```

`beta` 与 `beta_rebuilt` 一致，说明原系数能够通过分解公式重构。`theta_R` 为零，正好对应第 2 节散点图中的水平虚线；`beta` 为 3.3333，则对应同图中向上倾斜的实线。这些结果也与第 8.2 节 Stata 结果表的第一行一致。

配套脚本可以一次性运行四个例子和数值检查，并将结果保存到文件夹中：

```bash
python continuous_did_decomposition.py --self-test --output-dir results
```

程序使用独立 OLS 计算和中心化回归进行交叉核验 (NumPy Developers, [n.d.](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), numpy.linalg.lstsq)。 在 Python 3.11.14、NumPy 2.4.2 环境中，1,000 组随机样本、4 个教学情形和 10 类异常输入检查均通过；正文代码也已单独运行。随机样本中，原系数与重构系数的最大绝对差异小于 $2\times10^{-15}$。具体误差和日志见配套核验资料；这些检查不代替因果识别论证。

## 10. 在实证论文中应当报告什么？

主结果表可以并列报告水平对比和响应指数，附表再提供两组样本量、处理组平均强度、组内强度方差、混合权重及两个加权项。这样，读者既能了解两类关系，也能判断原连续系数主要由哪一部分构成。

若进行事件研究，应分别展示两类参数随事件时间的变化，并说明各期使用的处理批次与汇总权重是否一致。更换样本或增加协变量后，也应检查目标参数是否发生了变化，而不只是比较显著性。

本文的分解不能替代对处理强度与结果之间完整反应曲线的估计。需要研究非线性或不同强度区间的效果时，应选择相应方法，并重新核对识别条件。

简言之，先说明系数比较了哪些单位，再说明什么条件允许作因果解释。分解的用途，是让这两步判断都有明确的依据。

## 11. 参考文献

1. Callaway, B., Goodman-Bacon, A., & Sant’Anna, P. H. C. (**2026**). *contdid: Difference-in-differences with a continuous treatment* (Version 0.1.1) [R package]. [Link](https://cran.r-project.org/web/packages/contdid/refman/contdid.html).

2. Callaway, B., Goodman-Bacon, A., & Sant’Anna, P. H. C. (**in press**). Difference-in-differences with a continuous treatment. *American Economic Review*. [Link](https://doi.org/10.1257/aer.20240137), [PDF](https://arxiv.org/pdf/2107.02637v8), [Google](<https://scholar.google.com/scholar?q=Difference-in-Differences+with+a+Continuous+Treatment+Callaway+Goodman-Bacon+Sant%27Anna>). PDF 为 2025 年 12 月 31 日的 arXiv v8 工作论文，不是期刊最终排版版。

3. Credible-Answers. (**n.d.-a**). *did_multiplegt_dyn* [Stata and R software documentation]. [Link](https://github.com/Credible-Answers/did_multiplegt_dyn).

4. Credible-Answers. (**n.d.-b**). *did_multiplegt_stat* [Stata and R software documentation]. [Link](https://github.com/Credible-Answers/did_multiplegt_stat).

5. Goodman-Bacon, A. (**2021**). Difference-in-differences with variation in treatment timing. *Journal of Econometrics, 225*(2), 254–277. [Link](https://doi.org/10.1016/j.jeconom.2021.03.014), [PDF](https://doi.org/10.1016/j.jeconom.2021.03.014), [Google](<https://scholar.google.com/scholar?q=Difference-in-Differences+with+Variation+in+Treatment+Timing>).

6. NumPy Developers. (**n.d.**). *numpy.var*; *numpy.linalg.lstsq* [Software documentation]. [Variance](https://numpy.org/doc/stable/reference/generated/numpy.var.html), [Least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).

7. Roth, J. (**2022**). Pretest with caution: Event-study estimates after testing for parallel trends. *American Economic Review: Insights, 4*(3), 305–322. [Link](https://doi.org/10.1257/aeri.20210236), [PDF](https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf), [Google](<https://scholar.google.com/scholar?q=Pretest+with+Caution+Event-Study+Estimates+after+Testing+for+Parallel+Trends>).

8. StataCorp. (**n.d.**). *regress*; *summarize*. In *Stata Base Reference Manual*. [regress](https://www.stata.com/manuals/rregress.pdf), [summarize](https://www.stata.com/manuals/rsummarize.pdf).

9. Yu, F. (**2026**). *Two margins in difference-in-differences with a continuous treatment* (arXiv:2609.09488, v2, September 14, 2026). [Link](https://doi.org/10.48550/arXiv.2609.09488), [PDF](https://arxiv.org/pdf/2609.09488v2), [Google](<https://scholar.google.com/scholar?q=Two+Margins+in+Difference-in-Differences+with+a+Continuous+Treatment>).
