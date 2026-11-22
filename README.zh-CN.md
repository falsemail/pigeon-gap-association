# 群体飞行轨迹在观测中断前后的个体身份保持

本仓库对应费思邈、王悦、郝雪儿、李蕙存的论文 *Retaining individual identities across gaps in group flight trajectories*。读者可以从已保存的匹配结果重算外部评分，并检查开发集上只改变拒配门限后的结果。[English README](README.md)

在仓库根目录运行以下命令。需要 Python 3.8 或更新版本，无需安装第三方依赖、联网或使用 GPU。

```bash
python3 -I -S -B reproduce.py
```

Windows 可使用 `py -3 -I -S -B reproduce.py`。结果默认写入 `results/`，可用 `--output 路径` 修改输出目录。程序校验输入文件的 SHA-256、5,724 个条件、14 种方法逐条件的判断，以及输出与参考结果的一致性。比较浮点数的绝对容差为 `1e-12`。

| 输出文件 | 对应论文结果 |
|---|---|
| table3-retention.csv | 表 3，三种外部来源、两种时长下各方法的身份保持率 |
| table4-selection.csv | 表 4，20 秒中断时三种方法的保持率、精确率与计数误差 |
| external-results.csv | 外部终评的 84 行未舍入结果 |
| paired-intervals.csv | 按记录配对重抽样得到的 12 个描述性区间 |
| table-s9-calibration.csv | 补充表 S9，两种门限在相同 687 个开发条件上的评分，按来源等权 |
| 20s-attention-paired.csv | 事后对照，三个来源上混合模型减注意力匹配器的 20 秒保持率差异 |

外部原始数据的获取方式和 SHA-256 见[原始数据来源](data/source-provenance.json)。本仓库提供评分所需的局部参考标签、已保存的匹配决定和开发集的逐记录汇总。[模型目录](model/) 保留了论文的冻结权重及相应的模型源代码。服务器训练时使用的开发集数据适配程序未随包提供。从原始坐标重做训练和模型推理另需坐标预处理程序、各实验实现和软件环境。[数据字段和单位](docs/evidence.md) · [引用方式](CITATION.cff) · [使用说明](NOTICE.md)

[原始数据来源、划分与实验协议](docs/data-access.md)
