"""把三件 skill 组合成一条流水线 —— 这就是解这道题的完整管线:
  原始特征 → feature_engineer(处理数据) → train_model(训练+出概率) → calibrate(出标签)
oracle 评的就是这条流水线对 test 的预测。三件 skill 任何一件强了,整条线就强一点。"""
from skill import feature_engineer, train_model, calibrate


def run_pipeline(train_X, train_y, test_X):
    Xtr_e = feature_engineer.engineer(train_X)
    Xte_e = feature_engineer.engineer(test_X)
    probs = train_model.train_predict(Xtr_e, train_y, Xte_e)
    return calibrate.to_labels(probs["train"], train_y, probs["test"])
