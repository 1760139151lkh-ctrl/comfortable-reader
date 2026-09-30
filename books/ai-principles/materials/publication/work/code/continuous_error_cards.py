"""第二章：沿第一章的点卡片比较训练目标并改变目标编码。只用标准库。"""

import argparse
import json
import math
from pathlib import Path


base_records = [
    ((2.0, 2.0), 1),
    ((1.0, 3.0), 1),
    ((3.0, 1.0), 1),
    ((0.0, 0.0), -1),
    ((1.0, 1.0), -1),
    ((2.0, 0.0), -1),
    ((0.0, 2.0), -1),
]

work_dir = Path(__file__).resolve().parents[1]
result_dir = work_dir / "results"
old_report = json.loads(
    (result_dir / "first_perceptron_run.json").read_text(encoding="utf-8")
)
old_model = json.loads(
    (result_dir / "first_perceptron_model.json").read_text(encoding="utf-8")
)
assert [(tuple(row["features"]), row["label"]) for row in old_report["training_rows"]] == base_records
base_start = [*old_model["weights"], old_model["bias"]]
assert base_start == [1.0, 1.0, -3.0], "先恢复并运行第一章的原始实验"

parser = argparse.ArgumentParser(description="比较平方目标的训练及目标数字编码")
parser.add_argument("--target-scale", type=float, default=1.0,
                    help="将每条类别数字同时乘以同一个正数；变体结果另存")
args = parser.parse_args()
if not math.isfinite(args.target_scale) or args.target_scale <= 0:
    parser.error("--target-scale 必须是有限的正数")
records = [(features, target * args.target_scale) for features, target in base_records]
start = [value * args.target_scale for value in base_start]


def score(theta, features):
    x1, x2 = features
    return theta[0] * x1 + theta[1] * x2 + theta[2]


def classify(theta, features):
    return 1 if score(theta, features) > 0 else -1


def mean_square_loss(theta, examples):
    total = 0.0
    for features, target in examples:
        residual = score(theta, features) - target
        total += 0.5 * residual * residual
    return total / len(examples)


def mean_square_gradient(theta, examples):
    grad = [0.0, 0.0, 0.0]
    for features, target in examples:
        x1, x2 = features
        residual = score(theta, features) - target
        grad[0] += residual * x1
        grad[1] += residual * x2
        grad[2] += residual
    return [component / len(examples) for component in grad]


def single_sample_step(theta, features, target, learning_rate):
    residual = score(theta, features) - target
    x1, x2 = features
    return [
        theta[0] - learning_rate * residual * x1,
        theta[1] - learning_rate * residual * x2,
        theta[2] - learning_rate * residual,
    ]


first_features, first_target = records[0]
one_sample_trials = []
for rate in (0.1, 0.3):
    before = [0.0, 0.0, 0.0]
    after = single_sample_step(before, first_features, first_target, rate)
    one_sample_trials.append(
        {
            "learning_rate": rate,
            "before": before,
            "after": after,
            "score_before": score(before, first_features),
            "score_after": score(after, first_features),
            "loss_before": mean_square_loss(before, [records[0]]),
            "loss_after": mean_square_loss(after, [records[0]]),
        }
    )

theta = start.copy()
rate = 0.1
steps = 3000
checkpoints = []


def checkpoint(step):
    checkpoints.append(
        {
            "step": step,
            "theta": theta.copy(),
            "mean_square_loss": mean_square_loss(theta, records),
            "classification_errors": sum(
                classify(theta, x) != (1 if y > 0 else -1) for x, y in records
            ),
            "gradient": mean_square_gradient(theta, records),
        }
    )


checkpoint(0)
for step in range(1, steps + 1):
    gradient = mean_square_gradient(theta, records)
    theta = [theta[j] - rate * gradient[j] for j in range(3)]
    if step in (1, 10, 100, 500, 1000, steps):
        checkpoint(step)

exact_minimizer = [value * args.target_scale for value in (5 / 8, 5 / 8, -7 / 4)]
max_difference = max(abs(theta[j] - exact_minimizer[j]) for j in range(3))
assert max_difference < 1e-8
assert abs(mean_square_loss(start, records) - (2 / 7) * args.target_scale ** 2) < 1e-12
assert abs(mean_square_loss(exact_minimizer, records) - (3 / 28) * args.target_scale ** 2) < 1e-12
residuals_at_exact = [score(exact_minimizer, x) - y for x, y in records]
residual_feature_sums_at_exact = [
    sum(residual * x[0] for residual, (x, _) in zip(residuals_at_exact, records)),
    sum(residual * x[1] for residual, (x, _) in zip(residuals_at_exact, records)),
    sum(residuals_at_exact),
]
assert all(abs(value) < 1e-12 for value in residual_feature_sums_at_exact)

report = {
    "data_identity": "第一章同一批七张人工点卡片；没有新采样或自然数据",
    "target_scale": args.target_scale,
    "historical_scope": "本程序是现代课堂计算，不是 1960 年 Adaline 装置的复刻",
    "model_family": "z = w1*x1 + w2*x2 + b; z>0 为正类",
    "objective": "七条记录的 1/2*(z-y)^2 的平均",
    "start_loaded_from": "work/results/first_perceptron_model.json",
    "one_sample_trials": one_sample_trials,
    "batch_rate": rate,
    "batch_steps": steps,
    "checkpoints": checkpoints,
    "exact_minimizer_by_normal_equations": exact_minimizer,
    "residuals_at_exact": residuals_at_exact,
    "residual_feature_sums_at_exact": residual_feature_sums_at_exact,
    "max_difference_from_exact": max_difference,
    "final_predictions": [
        {
            "features": list(x),
            "target": y,
            "score": score(theta, x),
            "prediction": classify(theta, x),
        }
        for x, y in records
    ],
}

result_dir.mkdir(exist_ok=True)
stem = ("continuous_error_cards" if args.target_scale == 1.0 else
        "continuous_error_cards_scale_" + format(args.target_scale, ".8g").replace(".", "p"))
report_path = result_dir / f"{stem}_run.json"
model_path = result_dir / f"{stem}_model.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
model_path.write_text(
    json.dumps({"weights": theta[:2], "bias": theta[2]}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
restored = json.loads(model_path.read_text(encoding="utf-8"))
assert restored["weights"] == theta[:2] and restored["bias"] == theta[2]

print("单样本步长比较:", [
    (item["learning_rate"], item["score_after"], item["loss_after"])
    for item in one_sample_trials
])
print("批次起点损失:", checkpoints[0]["mean_square_loss"])
print("终点参数、损失、分类错数:", theta, checkpoints[-1]["mean_square_loss"],
      checkpoints[-1]["classification_errors"])
print("精确最小点最大差:", max_difference)
print("最小点残差与三列输入的内积:", residual_feature_sums_at_exact)
