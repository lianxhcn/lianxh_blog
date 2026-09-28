# RePPI：人工智能生成的因变量怎样进入回归

连享会推文配套材料，署名：舒清扬。独立实现版本：0.2.0。验证日期：2026-09-27；发布整理日期：2026-09-28。

## 下载与阅读

- [下载代码 ZIP](downloads/reppi-ols-v0.2.0.zip?raw=true)
- [阅读推文](post.md)
- [浏览源代码及说明](code/README.md)
- [算法对照说明](code/ALGORITHM-NOTE.md)
- [复现日志](code/verification-logs/)

## 运行

解压 ZIP，进入 reppi-ols-v0.2.0 目录，运行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python example_simulation.py
python validation_monte_carlo.py
python validation_monte_carlo.py --reps 200 --n 300 --N 3000
```

验证环境为 Python 3.11.14；主要依赖版本见 code/requirements-tested.txt。
10 项测试通过。200 次主模拟覆盖率为 0.955、0.955、0.955；小样本压力模拟为 0.960、0.955、0.895。有限样本覆盖不足已在推文中明确保留。

## 方法与边界

本实现遵循论文 arXiv v2 的全样本比例、按组样本量加权汇总与代入式协方差估计。仅验证独立同分布样本中的无权重普通最小二乘回归，不包括聚类标准误、面板依赖、非随机标注权重或人工智能生成的解释变量。

- 论文：[Ji、Lei 和 Zrnic (2026)](https://doi.org/10.1093/biomet/asag053)
- 固定方法版本：[arXiv v2](https://arxiv.org/abs/2501.09731v2)
- 作者研究代码：[Wenlong2000/RePPI](https://github.com/Wenlong2000/RePPI)

源代码由教学项目独立编写；作者仓库仅用于只读核对，未复制进本目录。许可说明见 [NOTICE.md](NOTICE.md)。
