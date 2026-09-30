"""第一章的点卡片实验及可重复的条件变更。只用 Python 标准库。"""

import argparse
import json
import math
from pathlib import Path


# 每条记录由 (两项测量, 人给的类别) 组成。数值是本章的教学数据。
records = [
    ((2.0, 2.0), 1),
    ((1.0, 3.0), 1),
    ((3.0, 1.0), 1),
    ((0.0, 0.0), -1),
    ((1.0, 1.0), -1),
    ((2.0, 0.0), -1),
    ((0.0, 2.0), -1),
]

parser = argparse.ArgumentParser(description="比较同一感知机在不同数据和步长下的更新")
parser.add_argument("--rate", type=float, default=1.0, help="每次更新的正步长")
parser.add_argument("--case", choices=("baseline", "omit-origin", "midpoint-conflict"),
                    default="baseline", help="要观察的卡片条件")
parser.add_argument("--max-epochs", type=int, default=200, help="最多检查多少整轮")
args = parser.parse_args()
if not math.isfinite(args.rate) or args.rate <= 0:
    parser.error("--rate 必须是有限的正数")
if args.max_epochs < 1:
    parser.error("--max-epochs 必须至少为 1")

if args.case == "omit-origin":
    records = [(features, label) for features, label in records if features != (0.0, 0.0)]
elif args.case == "midpoint-conflict":
    records = [(features, 1 if features == (1.0, 1.0) else label)
               for features, label in records]

learning_rate = args.rate
max_epochs = args.max_epochs


def lookup(examples, features):
    for saved_features, label in examples:
        if saved_features == features:
            return label
    return None


def score(weights, bias, features):
    return weights[0] * features[0] + weights[1] * features[1] + bias


def predict(weights, bias, features):
    return 1 if score(weights, bias, features) > 0 else -1


weights = [0.0, 0.0]
bias = 0.0
mistakes_by_epoch = []
first_updates = []

converged = False
for epoch in range(1, max_epochs + 1):
    updates = 0
    for features, label in records:
        old_score = score(weights, bias, features)
        # 等于 0 时也调整，让训练记录落到分界线的正确一侧。
        if label * old_score <= 0:
            old_weights = weights.copy()
            old_bias = bias
            weights[0] += learning_rate * label * features[0]
            weights[1] += learning_rate * label * features[1]
            bias += learning_rate * label
            updates += 1
            if len(first_updates) < 4:
                first_updates.append({
                    "epoch": epoch,
                    "features": list(features),
                    "label": label,
                    "old_score": old_score,
                    "old_weights": old_weights,
                    "old_bias": old_bias,
                    "new_weights": weights.copy(),
                    "new_bias": bias,
                    "new_score_on_same_record": score(weights, bias, features),
                })
    mistakes_by_epoch.append(updates)
    if updates == 0:
        converged = True
        break

unseen_points = [(3.0, 2.0), (0.0, 1.0)]
report = {
    "data_identity": "本章人工制作的二维教学卡片；并非历史实验或自然数据",
    "case": args.case,
    "learning_rate": learning_rate,
    "max_epochs": max_epochs,
    "converged_within_budget": converged,
    "epochs_until_no_update": epoch,
    "updates_by_epoch": mistakes_by_epoch,
    "first_updates": first_updates,
    "trained_weights": weights.copy(),
    "trained_bias": bias,
    "training_rows": [
        {"features": list(x), "label": y, "score": score(weights, bias, x),
         "prediction": predict(weights, bias, x)}
        for x, y in records
    ],
    "unseen_rows": [
        {"features": list(x), "lookup": lookup(records, x),
         "score": score(weights, bias, x), "prediction": predict(weights, bias, x)}
        for x in unseen_points
    ],
}

work_dir = Path(__file__).resolve().parents[1]
result_dir = work_dir / "results"
result_dir.mkdir(exist_ok=True)
canonical = args.case == "baseline" and learning_rate == 1.0 and max_epochs == 200
stem = "first_perceptron" if canonical else (
    "first_perceptron_probe_" + args.case.replace("-", "_")
    + "_rate_" + format(learning_rate, ".8g").replace(".", "p")
    + "_epochs_" + str(max_epochs)
)
model_path = result_dir / f"{stem}_model.json"
report_path = result_dir / f"{stem}_run.json"

report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

if converged:
    model_path.write_text(
        json.dumps({"weights": weights, "bias": bias}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    restored = json.loads(model_path.read_text(encoding="utf-8"))
    assert restored["weights"] == weights and restored["bias"] == bias
    assert all(row["prediction"] == row["label"] for row in report["training_rows"])

print("每轮更新次数:", mistakes_by_epoch)
print("学到的权重和偏置:", weights, bias)
print("未见过的卡片:", report["unseen_rows"])
print("运行记录:", report_path)
if converged:
    print("模型已保存并重新读取:", model_path)
else:
    print("达到轮数上限时仍有更新；这项观察本身不证明数据不可分。")
