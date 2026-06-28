# 设计你的 skill —— 先回答四个问题,再指挥 agent 动手

> 这道题一件 skill 解不了(raw ≈ 0.55)。你要把三件可组合的技艺逐件做强、组合过 0.80。
> 别急着让 agent 写代码,先想清楚这四问,它们就是你给 agent 的「规格」。

1. **为什么 raw 这么差**:先跑一次 baseline 流水线看分(≈0.55)。信号到底藏在哪?
   (提示:试试只把 `feature_engineer` 改成加交互项 x_i*x_j,再看分——这一件能抬多少?)
2. **每件 skill 值多少**:跑 `python -c "from ml.ablation import report; report()"`。
   feature_engineer、calibrate 各把验证集分数抬了多少?为什么三件价值天差地别?
   (这就是市场会给它们的「价钱」——你愿意花多少 credit 买哪件?)
3. **怎么组合过线**:三件都做强后,组合流水线对 test 的 accuracy 能不能过 0.80?
   哪一件是承重的、哪一件只是抠边角?
4. **怎么不靠偷看**:你怎么保证分数是真本事,而不是读了 `ml/_truth/`?
   (grader 读答案,你的解答**不读**;调阈值/选模型只能用训练集切验证集,绝不碰 test)
