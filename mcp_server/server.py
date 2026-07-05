"""把你的 skill 暴露成一个真 MCP server 的 tool。
两种传输(教学两态):
  - 本地 stdio(默认):python -m mcp_server.server —— 只能本地进程调,用来本机验证。
  - 远程 SSE:python -m mcp_server.server --remote —— 在 0.0.0.0:8000 起 HTTP/SSE,
    沙箱的 preview URL 能指过来,小镇里别的镇民(经 Broker)才能远程调到你的 skill。
- engineer_features:单独暴露「特征工程」这件 skill(证明一件零件可被独立调用/交易)
- predict_conversion:暴露**组合好的整条流水线**(外部 agent 调它就能拿转化预测)
这就是「skill 经 MCP 被外部用」+「零件可组合、可单独交易」。"""
import json
import os
import sys
from mcp.server.fastmcp import FastMCP
from skill import feature_engineer
from pipeline import run_pipeline

# 训练数据随 skill bundle 走(mcp_server/resources/),这样发布到 GitHub 后 server 仍能起来 + 预测。
# (ml/data 是 lab 本地练习数据,发布时被 /ml/ /data/ 排除 → 不能依赖它;resources/ 随 mcp_server/ 一起发布。)
_RES = os.path.join(os.path.dirname(__file__), "resources")
TRAIN_X = json.load(open(os.path.join(_RES, "train_x.json")))
TRAIN_Y = json.load(open(os.path.join(_RES, "train_y.json")))

# host/port 仅 SSE 传输用;stdio 忽略。绑 0.0.0.0 才能被沙箱 preview 代理打到(loopback 进不来)。
SKILL_PORT = int(os.environ.get("SKILL_PORT", "8000"))
mcp = FastMCP("conversion-skills", host="0.0.0.0", port=SKILL_PORT)


@mcp.tool()
def engineer_features(X: list[list[float]]) -> list[list[float]]:
    """单件 skill:把原始特征变换成更可学的特征(交互项/平方项)。"""
    return feature_engineer.engineer(X)


@mcp.tool()
def predict_conversion(test_x: list[list[float]]) -> list[int]:
    """组合流水线:对每个试用用户预测是否转付费(1/0)。内部 = feature_engineer→train_model→calibrate。"""
    return run_pipeline(TRAIN_X, TRAIN_Y, test_x)


@mcp.tool()
def aircraft_finetune_recipe() -> dict:
    """FGVC-Aircraft(100 类飞机机型)细粒度微调的最佳配方 —— 经 10 次真实 GPU 实验验证。
    安全边界:本工具只返回「配方/知识」,绝不触发 GPU 训练(否则调用者会烧我的 GPU 额度)。
    调用者拿这套配置,在自己的 GPU 上跑。"""
    return {
        "task": "FGVC-Aircraft, 100-class fine-grained variant classification",
        "backbone": "convnext_tiny",  # 决定性杠杆:r18→r50→convnext_tiny,两次换骨架 +8.2/+7.7pp
        "freeze_backbone": False,      # 全量微调远胜冻结骨架(37.7% → 86.2%)
        "epochs": 15,
        "lr": 3e-4,
        "batch_size": 64,
        "lr_scheduler": "CosineAnnealingLR(T_max=epochs)",  # 白嫖杠杆:后期 lr 退火压收敛,train_acc 0.92→0.998
        "augment": False,              # 实测:该任务上增强无增益(±1.5pp 噪声),convnext 也不需要
        "tta": "推理时原图+水平翻转 softmax 平均(零训练成本,实测 +0.63pp: 86.23%→86.86%)",
        "verified_test_acc": 0.8686,
        "rank": "第 1 名 (2026-07-04)",
        "lessons": [
            "骨架容量是最大杠杆:resnet18 约 70% 见顶,resnet50 到 78.5%,convnext_tiny 到 86.2%。两次大跃迁都靠换更强骨架,不是调参。",
            "数据增强在此任务净收益≈0:小模型是容量不足(不是过拟合)→增强没用;大模型配足 epoch 也只追平、没超过无增强。",
            "cosine LR scheduler 是大骨架上的白嫖杠杆:后期 lr 退到 ~3e-6 压住收敛,train_acc 敢压到 0.998 不反弹。零额外成本。",
            "选对骨架方向比『无脑上最大』更重要:没上 vit_b_16(小数据集易欠拟合又贵),选 convnext_tiny——比 r50 强、又保留卷积 inductive bias、适合 3k 张中小数据集。",
            "10 次实验仅花 $4.27 / $30 —— 单变量 + 读日志信号做因果归因,不盲目烧钱。",
        ],
    }


if __name__ == "__main__":
    remote = "--remote" in sys.argv or os.environ.get("MCP_TRANSPORT") == "sse"
    mcp.run(transport="sse" if remote else "stdio")
