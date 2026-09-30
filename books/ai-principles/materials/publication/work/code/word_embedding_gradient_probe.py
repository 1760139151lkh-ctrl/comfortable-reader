"""用同一个词占据两个上下文位置，手算并核对嵌入行收到的梯度。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


OUT = Path(__file__).resolve().parents[1] / "results" / "word_embedding_gradient_probe.json"


def main() -> None:
    embeddings = np.array([[0.0, 0.0], [0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])
    hidden_weights = np.array([[0.2, -0.1, 0.4, 0.3], [-0.2, 0.5, 0.1, -0.4]])
    hidden_bias = np.array([0.05, -0.03])
    output_weights = np.array([[0.1, 0.2], [-0.4, 0.3], [0.5, -0.2], [0.2, 0.4]])
    output_bias = np.array([0.0, 0.1, -0.05, 0.03])
    context = [1, 1]
    target = 2

    joined = np.concatenate([embeddings[index] for index in context])
    preactivation = hidden_weights @ joined + hidden_bias
    hidden = np.tanh(preactivation)
    logits = output_weights @ hidden + output_bias
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    loss = -np.log(probabilities[target])

    # 第十三章的 softmax 导数先到输出分数，随后沿两层链式法则回到两个查表位置。
    score_gradient = probabilities.copy()
    score_gradient[target] -= 1.0
    hidden_gradient = output_weights.T @ score_gradient
    preactivation_gradient = (1.0 - hidden**2) * hidden_gradient
    joined_gradient = hidden_weights.T @ preactivation_gradient
    first_use = joined_gradient[:2]
    second_use = joined_gradient[2:]
    shared_row_gradient = first_use + second_use

    e = torch.tensor(embeddings, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(hidden_weights, dtype=torch.float64)
    b = torch.tensor(hidden_bias, dtype=torch.float64)
    u = torch.tensor(output_weights, dtype=torch.float64)
    a = torch.tensor(output_bias, dtype=torch.float64)
    x = torch.tensor(context, dtype=torch.long)
    torch_logits = u @ torch.tanh(w @ e[x].flatten() + b) + a
    torch_loss = F.cross_entropy(torch_logits.unsqueeze(0), torch.tensor([target]))
    torch_loss.backward()
    mismatch = float(np.max(np.abs(shared_row_gradient - e.grad[1].detach().numpy())))
    if mismatch > 1e-12 or not np.allclose(e.grad[0].detach().numpy(), 0) or not np.allclose(e.grad[3].detach().numpy(), 0):
        raise AssertionError("手算梯度与共享嵌入行的自动求导不一致")

    revised = embeddings.copy()
    revised[1] -= 0.1 * shared_row_gradient
    new_hidden = np.tanh(hidden_weights @ np.concatenate([revised[index] for index in context]) + hidden_bias)
    new_scores = output_weights @ new_hidden + output_bias
    new_shifted = new_scores - new_scores.max()
    new_probabilities = np.exp(new_shifted) / np.exp(new_shifted).sum()
    new_loss = -np.log(new_probabilities[target])

    result = {
        "context_ids": context,
        "target_id": target,
        "meaning": "同一编号1在两个输入位置被查两次，只更新一次共享的矩阵行；编号2虽是目标，并未作为上下文嵌入行被查。",
        "probabilities": probabilities.tolist(),
        "loss_before": float(loss),
        "output_score_gradient": score_gradient.tolist(),
        "first_use_embedding_gradient": first_use.tolist(),
        "second_use_embedding_gradient": second_use.tolist(),
        "shared_embedding_row_gradient": shared_row_gradient.tolist(),
        "torch_shared_embedding_row_gradient": e.grad[1].detach().numpy().tolist(),
        "largest_manual_autograd_difference": mismatch,
        "absent_input_row_gradient_norm": float(e.grad[3].norm().item()),
        "loss_after_only_embedding_row_sgd_step": float(new_loss),
        "learning_rate": 0.1,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
