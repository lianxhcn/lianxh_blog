# 连续 DID 系数分解：原理、适用条件与实现

本目录提供连享会方法教程及配套教学程序，作者为连小白 (连享会)。程序用于两期、等权、无额外协变量的连续 DID 系数分解；不是论文作者的官方软件包。

- [阅读 v08 正文](连续DID系数分解-原理适用条件与实现-连小白-v08.md)
- [下载 Stata 完整示例](continuous_did_stata_demo.do)
- [下载 Stata 最简示例](continuous_did_stata_minimal.do)
- [下载 Python 示例](continuous_did_decomposition.py)
- [下载自包含 HTML 预览](preview.html)；保存到本地后用浏览器打开。

## 运行

将三份程序下载到同一目录。在 Stata 中切换到该目录，执行：

```stata
do "continuous_did_stata_minimal.do"
do "continuous_did_stata_demo.do"
```

程序会清空当前会话的数据，运行前请先保存。完整版在当前目录的 `results_stata/` 下写入日志与 CSV；四条 case 完成输出及最终 STATA_CHECK 输出表示断言通过。

Python 需要 NumPy。在相应目录执行：

```bash
python continuous_did_decomposition.py --self-test --output-dir results
```

前序本地核验环境为 Windows、Stata 19.5、Python 3.11.14、NumPy 2.4.2。已核验四个教学情形、1,000 组随机样本和 10 类异常输入。本次 GitHub 发布保持程序原样，没有重复估计。`version 17.0` 是代码声明，未声称实测 Stata 17。

## 方法与范围

方法来源：Yu, F. (2026). *Two margins in difference-in-differences with a continuous treatment*. [论文](https://doi.org/10.48550/arXiv.2609.09488)。正文保留其 v2 对应的说明。

计算结果验证统计点估计与分解恒等式，不代表因果识别成立，不提供标准误或多期联合推断，不能用四城市数据进行实质性显著性检验。

正文三图使用正式 PicGo 图床地址；`figs/raw/` 保存同字节备份，链接清单见 `figs/uploaded-images.md`。程序与资源摘要见 `MANIFEST.json`。本目录沿用仓库已有 LICENSE，没有另行添加许可证。
