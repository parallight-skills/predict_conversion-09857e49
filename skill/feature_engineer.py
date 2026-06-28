"""SKILL #1 —— 特征工程(「如何处理数据」)。
把原始 6 个特征变换成「更可学」的特征。

⚠️ 现在是身份变换(原样返回、啥也没做)——这就是为什么 raw 流水线只有约 0.55(≈瞎猜):
这道题的信号几乎全藏在**特征的交互**里(x_i·x_j)和**非线性**里(x_i²),纯线性模型抓不到。

指挥你的 agent 把它改成真特征工程:加上两两交互项 x_i*x_j、平方项 x_i**2,
让藏起来的非线性信号变得「线性可分」。这是本题**价值最高的一件 skill**。
约束:只看特征本身做变换,绝不读 ml/_truth/。"""

from itertools import combinations_with_replacement


def engineer(X):
    # 学员指挥:原始 6 列平铺 + 所有 2 阶乘积(x_i·x_j，含平方) + 所有 3 阶乘积(x_i·x_j·x_k)。
    out = []
    for row in X:
        feats = list(row)
        n = len(row)
        for i, j in combinations_with_replacement(range(n), 2):
            feats.append(row[i] * row[j])
        for i, j, k in combinations_with_replacement(range(n), 3):
            feats.append(row[i] * row[j] * row[k])
        out.append(feats)
    return out
