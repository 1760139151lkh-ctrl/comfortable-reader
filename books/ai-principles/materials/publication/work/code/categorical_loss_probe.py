"""第十三章：手算三类归一化与梯度，再核对 PyTorch 交叉熵。"""

import json
import math
from pathlib import Path

import torch


scores = (2.0, 1.0, 0.0)
target = 0


def probabilities(values):
    # 同减最大值只为避免大指数溢出，不改变概率。
    high = max(values)
    weights = [math.exp(value - high) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


manual_prob = probabilities(scores)
manual_loss = -math.log(manual_prob[target])
manual_gradient = [p - int(index == target) for index, p in enumerate(manual_prob)]

logits = torch.tensor([scores], dtype=torch.float64, requires_grad=True)
label = torch.tensor([target], dtype=torch.long)
loss = torch.nn.functional.cross_entropy(logits, label)
loss.backward()
torch_gradient = logits.grad[0].tolist()

shifted_scores = tuple(value + 1000.0 for value in scores)
shifted_prob = probabilities(shifted_scores)
shifted_logits = torch.tensor([shifted_scores], dtype=torch.float64)
shifted_loss = torch.nn.functional.cross_entropy(shifted_logits, label).item()

assert max(abs(a - b) for a, b in zip(manual_prob, shifted_prob)) < 1e-12
assert abs(manual_loss - loss.item()) < 1e-12
assert max(abs(a - b) for a, b in zip(manual_gradient, torch_gradient)) < 1e-12
assert abs(manual_loss - shifted_loss) < 1e-12

result = {
    "scores": scores,
    "correct_class_index": target,
    "probabilities": manual_prob,
    "manual_negative_log_probability": manual_loss,
    "manual_gradient_p_minus_one_hot": manual_gradient,
    "torch_cross_entropy": loss.item(),
    "torch_gradient": torch_gradient,
    "common_shift": 1000.0,
    "shifted_probabilities": shifted_prob,
    "shifted_torch_loss": shifted_loss,
    "scope": "One author-made three-class numeric check; does not train Fashion-MNIST or prove calibration.",
}
output = Path(__file__).resolve().parents[1] / "results/categorical_loss_probe.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("三类概率:", [round(p, 6) for p in manual_prob])
print("损失:", round(manual_loss, 6))
print("三个分数的导数:", [round(g, 6) for g in manual_gradient])
print("共同加 1000 后，概率与损失不变；PyTorch 与手算一致")
