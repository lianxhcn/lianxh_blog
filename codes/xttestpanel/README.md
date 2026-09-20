# xttestpanel：面板回归诊断的复现材料

本目录配套连享会推文《Stata命令xttestpanel：面板数据模型跑完后，六项检验一起做》，作者为艾米丽 (连享会)。

- [下载全部附件 ZIP](https://raw.githubusercontent.com/lianxhcn/lianxh_blog/main/codes/xttestpanel/xttestpanel-materials.zip)
- [真实数据 dofile](scripts/xttestpanel_nlswork_post.do) 与 [原始运行日志](logs/xttestpanel_nlswork_post.log)：对应推文第 3 节。
- [模拟与教学 dofile](xttestpanel_lecture.do) 与 [原始运行日志](logs/xttestpanel_lecture.log)：其中模拟部分对应推文第 4 节。

## 运行方法

1. 下载并解压 ZIP。在 Stata 中用 `cd` 切换到包含本 README 和 `xttestpanel_lecture.do` 的目录，或通过菜单设置工作目录。不要直接在 ZIP 内运行。
2. 安装作者命令并核对版本：

```stata
ssc install xttestpanel, replace
which xttestpanel
```

3. 保存当前内存中的数据后，运行相应 dofile：

```stata
* 真实工资面板，包含全样本诊断和 func 子样本演示。
do "scripts/xttestpanel_nlswork_post.do"

* 人为设置问题的模拟及其他课堂演示。
do "xttestpanel_lecture.do"
```

脚本会清空当前会话数据，并覆盖 `logs/` 中的同名日志。需要保留随附件提供的基准日志时，请先解压一份工作副本。模拟讲义还会生成 `outputs/` 与 `figs/raw/` 下的文件。`webuse nlswork` 需要网络。

发布版仅调整命令路径检查：有项目内 `x/xttestpanel.ado` 时优先使用；否则使用已经安装的命令。模型、变量、随机种子、抽样方式和检验选项与基准脚本保持一致。没有捆绑或重新授权作者的 ado 文件，也没有捆绑 Stata 数据。

## 版本与结果口径

基准环境为 Windows、StataNow/MP 19.5；命令头为 `xttestpanel 1.0.0 09jun2026`。SSC 提供的版本可能更新，安装后应核对 `which xttestpanel` 的输出；本文不保证新版本与旧日志逐位相同。作者记录：[RePEc S459751](https://econpapers.repec.org/software/bocbocode/s459751.htm)。发布整理日期：2026-09-20。

- 真实数据日志的 FE 估计样本为 28,036 条观测、4,698 人。全样本 `func` 因超过 5,000 条观测返回缺失，其他五类诊断有输出。
- 函数形式演示另用随机抽取的 300 人、1,786 条观测，种子为 `20260919`，`reps(199)`；J = 6.9593，bootstrap p = 0.0050。
- 模拟采用 `N=50`、`T=20`、种子 `20260609`，`reps(99)` 是函数形式检验内部的 bootstrap 次数。这里只演示一次模拟，不是重复蒙特卡洛尺寸或功效评估。
- 教学讲义后半部分遇到网络下载失败时会回退到 Stata 自带的 `xtline1` 模拟数据。随附旧讲义日志确实包含这一回退例子，不能当成推文第 3 节的 nlswork 结果；真实结果应看专用脚本和日志。
- 表中的程序输出不等于方法有效性的全面验证。短、不平衡面板下的 CSD 差异、Hausman 的维持假设及 bootstrap 的适用条件，应结合推文讨论理解。

本目录只发布推文第 7.1 节所需附件，未包含尚未同步最新真实数据分析的 notebook。已有日志保持原样；其中本机路径仅记录原运行环境，不需要在读者电脑上创建相同路径。

## 文件校验

`MANIFEST.csv` 列出 README、两份 dofile 和两份日志的 SHA256。ZIP 包含上述五个文件、校验清单和 `.gitattributes`，保留 `scripts/` 与 `logs/` 目录结构。目录内禁用 Git 自动换行转换，以保证下载文件与清单中的字节校验一致；原日志的表格空格保持不变。
