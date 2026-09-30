"""第三章：用两处可调中间活动学习四点异或。只用 Python 标准库。"""

import json
import math
import random
from pathlib import Path


# 四个输入已在第一章用于证明“单条直线分不开”。
rows = [
    ((0.0, 0.0), -1.0),
    ((1.0, 0.0), 1.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 1.0), -1.0),
]


def initial_parameters(seed):
    rng = random.Random(seed)
    return {
        "a": [[rng.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(2)],
        "c": [rng.uniform(-0.5, 0.5) for _ in range(2)],
        "v": [rng.uniform(-0.5, 0.5) for _ in range(2)],
        "b": rng.uniform(-0.5, 0.5),
    }


def forward(params, features):
    x1, x2 = features
    z = [
        params["a"][j][0] * x1 + params["a"][j][1] * x2 + params["c"][j]
        for j in range(2)
    ]
    h = [math.tanh(value) for value in z]
    score = params["b"] + sum(params["v"][j] * h[j] for j in range(2))
    return z, h, score


def empty_gradient():
    return {
        "a": [[0.0, 0.0], [0.0, 0.0]],
        "c": [0.0, 0.0],
        "v": [0.0, 0.0],
        "b": 0.0,
    }


def loss_and_gradient(params, examples):
    gradient = empty_gradient()
    total_loss = 0.0
    count = len(examples)
    for features, target in examples:
        _, hidden, score = forward(params, features)
        residual = score - target
        total_loss += 0.5 * residual * residual / count
        gradient["b"] += residual / count

        for j in range(2):
            gradient["v"][j] += residual * hidden[j] / count
            influence = residual * params["v"][j] * (1.0 - hidden[j] ** 2)
            gradient["c"][j] += influence / count
            for k in range(2):
                gradient["a"][j][k] += influence * features[k] / count
    return total_loss, gradient


def updated(params, gradient, rate):
    # 返回另一组参数；本轮所有导数都来自同一组旧参数。
    return {
        "a": [
            [params["a"][j][k] - rate * gradient["a"][j][k] for k in range(2)]
            for j in range(2)
        ],
        "c": [params["c"][j] - rate * gradient["c"][j] for j in range(2)],
        "v": [params["v"][j] - rate * gradient["v"][j] for j in range(2)],
        "b": params["b"] - rate * gradient["b"],
    }


def predictions(params):
    result = []
    for features, target in rows:
        _, hidden, score = forward(params, features)
        result.append({
            "features": list(features),
            "target": target,
            "hidden": hidden,
            "score": score,
            "prediction": 1 if score > 0 else -1,
        })
    return result


def gradient_number(gradient, path):
    if path[0] == "b":
        return gradient["b"]
    if path[0] == "a":
        return gradient["a"][path[1]][path[2]]
    return gradient[path[0]][path[1]]


def changed_parameter(params, path, amount):
    result = json.loads(json.dumps(params))
    if path[0] == "b":
        result["b"] += amount
    elif path[0] == "a":
        result["a"][path[1]][path[2]] += amount
    else:
        result[path[0]][path[1]] += amount
    return result


def finite_difference_check(params):
    # 和手写导数独立地量“参数稍动一点，损失变多少”。
    _, gradient = loss_and_gradient(params, rows)
    paths = [("a", j, k) for j in range(2) for k in range(2)]
    paths += [("c", j) for j in range(2)]
    paths += [("v", j) for j in range(2)]
    paths += [("b",)]
    gap = 1e-6
    differences = []
    for path in paths:
        plus = changed_parameter(params, path, gap)
        minus = changed_parameter(params, path, -gap)
        plus_loss, _ = loss_and_gradient(plus, rows)
        minus_loss, _ = loss_and_gradient(minus, rows)
        measured = (plus_loss - minus_loss) / (2 * gap)
        differences.append(abs(measured - gradient_number(gradient, path)))
    return max(differences)


def gradient_l2(gradient):
    entries = [gradient["b"], *gradient["c"], *gradient["v"]]
    entries.extend(value for row in gradient["a"] for value in row)
    return math.sqrt(sum(value * value for value in entries))


def train(seed, steps, rate, initial=None):
    params = initial_parameters(seed) if initial is None else json.loads(json.dumps(initial))
    checkpoints = []

    def save_checkpoint(step):
        loss, gradient = loss_and_gradient(params, rows)
        checkpoints.append({
            "step": step,
            "loss": loss,
            "gradient": gradient,
            "parameters": params,
            "predictions": predictions(params),
        })

    save_checkpoint(0)
    for step in range(1, steps + 1):
        _, gradient = loss_and_gradient(params, rows)
        params = updated(params, gradient, rate)
        if step in (1, 10, 100, 300, 600, 1000, 3000):
            save_checkpoint(step)
    final_loss, final_gradient = loss_and_gradient(params, rows)
    return {
        "seed": seed,
        "steps": steps,
        "rate": rate,
        "checkpoints": checkpoints,
        "final_parameters": params,
        "final_loss": final_loss,
        "final_gradient_l2": gradient_l2(final_gradient),
        "final_predictions": predictions(params),
    }


hand_built = {
    "a": [[6.0, 6.0], [6.0, 6.0]],
    "c": [-3.0, -9.0],
    "v": [1.0, -1.0],
    "b": -1.0,
}
hand_predictions = predictions(hand_built)
assert all(item["prediction"] == item["target"] for item in hand_predictions)

zero = {
    "a": [[0.0, 0.0], [0.0, 0.0]],
    "c": [0.0, 0.0],
    "v": [0.0, 0.0],
    "b": 0.0,
}
zero_loss, zero_gradient = loss_and_gradient(zero, rows)
assert zero_loss == 0.5 and zero_gradient == empty_gradient()

gradient_check = finite_difference_check(initial_parameters(1))
assert gradient_check < 1e-8

successful_run = train(seed=1, steps=600, rate=0.2)
comparison_run = train(seed=3, steps=3000, rate=0.2)
tied_initial = {
    "a": [[0.2, 0.2], [0.2, 0.2]],
    "c": [0.1, 0.1],
    "v": [0.2, 0.2],
    "b": 0.0,
}
tied_run = train(seed=None, steps=600, rate=0.2, initial=tied_initial)
assert tied_run["final_parameters"]["a"][0] == tied_run["final_parameters"]["a"][1]
assert tied_run["final_parameters"]["c"][0] == tied_run["final_parameters"]["c"][1]
assert tied_run["final_parameters"]["v"][0] == tied_run["final_parameters"]["v"][1]
assert successful_run["final_loss"] < 1e-4
assert all(
    item["prediction"] == item["target"]
    for item in successful_run["final_predictions"]
)

report = {
    "data_identity": "四张人工指定类别的二值点卡片；全部点都参与训练",
    "historical_scope": "现代教学实现，不复现 1985/1986 论文的原任务、网络或结果",
    "model": "two tanh hidden units and one linear output, square loss",
    "hand_built_predictions": hand_predictions,
    "zero_initialization": {"loss": zero_loss, "gradient": zero_gradient},
    "finite_difference_max_abs_gap": gradient_check,
    "seed_1_run": successful_run,
    "seed_3_comparison": comparison_run,
    "tied_initial_and_run": {"initial": tied_initial, "run": tied_run},
}

result_dir = Path(__file__).resolve().parents[1] / "results"
result_dir.mkdir(exist_ok=True)
report_path = result_dir / "two_hidden_units_run.json"
model_path = result_dir / "two_hidden_units_model.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
model_path.write_text(
    json.dumps(successful_run["final_parameters"], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
restored = json.loads(model_path.read_text(encoding="utf-8"))
assert restored == successful_run["final_parameters"]

print("手造中间表示的四个分数:", [round(p["score"], 4) for p in hand_predictions])
print("零初值损失和最大导数:", zero_loss, max(
    abs(gradient_number(zero_gradient, path))
    for path in [("a", 0, 0), ("a", 0, 1), ("a", 1, 0), ("a", 1, 1),
                 ("c", 0), ("c", 1), ("v", 0), ("v", 1), ("b",)]
))
print("手写导数与数值差分最大差:", gradient_check)
print("种子 1，600 步:", successful_run["final_loss"],
      [round(p["score"], 4) for p in successful_run["final_predictions"]])
print("种子 3，3000 步:", comparison_run["final_loss"],
      [round(p["score"], 4) for p in comparison_run["final_predictions"]])
print("种子 3，终点梯度长度:", comparison_run["final_gradient_l2"])
print("相同非零初值，600 步:", tied_run["final_loss"],
      [round(p["score"], 4) for p in tied_run["final_predictions"]],
      "两单元参数保持相同:", tied_run["final_parameters"]["a"][0] == tied_run["final_parameters"]["a"][1])
