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


if __name__ == "__main__":
    remote = "--remote" in sys.argv or os.environ.get("MCP_TRANSPORT") == "sse"
    mcp.run(transport="sse" if remote else "stdio")
