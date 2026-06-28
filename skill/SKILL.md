---
name: conversion-skills
description: 解「Trial-to-Paid Conversion」的一套可组合 skill —— 特征工程 / 模型训练 / 校准三件零件,组合成一条预测流水线。需要给试用用户打转化预测分时用我。
---

# conversion-skills(三件可组合的 skill)

> 这道题**一件 skill 解不了**:raw 单干 ≈ 0.55(瞎猜)。信号藏在特征的交互/非线性里,
> 必须把几件专门的技艺**组合**起来才过线。这正是 skill 市场的意义:每件技艺各有价钱,你买缺的那件。

这个 bundle 里有三件**各自可发布、可单独交换**的 skill,串成一条流水线:

| skill | 管的事 | 文件 | 大致价值(反事实增量) |
|---|---|---|---|
| `feature_engineer` | 如何处理数据(交互项/平方项) | `skill/feature_engineer.py` | **高**(主杠杆,~+0.3) |
| `train_model` | 用什么模型 / 怎么训练 | `skill/train_model.py` | 结构必需(没它产不出预测) |
| `calibrate` | 如何预测(阈值校准) | `skill/calibrate.py` | 低(便宜,~+0.0x) |

流水线:`原始特征 → feature_engineer → train_model → calibrate → 标签`(见 `pipeline.py`)。

## 怎么用(何时该叫我)
当你手上有一批试用用户的行为特征、要预测谁会付费时,调 `predict_conversion`(组合流水线);
或单独买 `engineer_features` 这件零件,拼到你自己的管线里。

## 输入 / 输出
- 输入:`test_x` = 二维数组,每行 6 个标准化行为特征
- 输出:与 `test_x` 等长的 0/1 列表

## 实现要点(指挥 agent 时的规格)
<三件 baseline 都很弱(身份特征 / 欠拟合模型 / 0.5 阈值)→ raw ~0.55。你要逐件改强,
 跑 `ml/ablation.py` 看每件抬了多少分(=它的价钱),组合到测试集 accuracy 过 0.80。>
