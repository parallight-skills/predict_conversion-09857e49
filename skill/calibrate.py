"""SKILL #3 —— 校准 / 预测(「如何 predict」)。
把模型输出的概率变成最终的 0/1 标签。

⚠️ 现在是固定 0.5 阈值。指挥 agent 改成**在训练集上挑一个让训练 accuracy 最高的阈值**
(threshold tuning)——类别不均衡时,这一步能再抠几分。这是价值最小、但也最便宜的一件 skill。"""


def to_labels(train_probs, train_y, test_probs):
    thr = 0.5  # 留给你:用 train_probs / train_y 调一个更好的阈值
    return [1 if p >= thr else 0 for p in test_probs]
