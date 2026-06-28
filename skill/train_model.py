"""SKILL #2 —— 模型与训练(「用什么模型 / 怎么训练」)。
在(已被 feature_engineer 处理过的)训练数据上拟合一个模型,给训练集和测试集都输出概率。

⚠️ 现在是一个**欠拟合**的 logistic(迭代太少、学习率太小、没正则)。
指挥 agent 把它训练到位(更多迭代 / 合适学习率 / L2 正则),或换个更合适的模型。"""
import numpy as np


def train_predict(train_X, train_y, test_X):
    Xtr = np.asarray(train_X, float)
    y = np.asarray(train_y, float)
    Xte = np.asarray(test_X, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    Xte = np.hstack([np.ones((len(Xte), 1)), Xte])
    w = np.zeros(Xtr.shape[1])
    lr = 0.1            # 学员指挥:0.02 -> 0.1
    for _ in range(300):  # 学员指挥:150 -> 300
        p = 1 / (1 + np.exp(-Xtr @ w))
        w -= lr * (Xtr.T @ (p - y)) / len(y)
    return {
        "train": [float(v) for v in 1 / (1 + np.exp(-Xtr @ w))],
        "test": [float(v) for v in 1 / (1 + np.exp(-Xte @ w))],
    }
