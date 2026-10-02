"""第四章：九参数网络的一步计算职责交接。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


WORK = Path(__file__).resolve().parents[1]
RATE = 0.2


def flattened(parts):
    return [
        *parts["a"][0],
        *parts["a"][1],
        *parts["c"],
        *parts["v"],
        parts["b"],
    ]


def tensor_values(a, c, v, b):
    return {
        "a": a.detach().cpu().tolist(),
        "c": c.detach().cpu().tolist(),
        "v": v.detach().cpu().tolist(),
        "b": b.detach().cpu().item(),
    }


def max_gap(left, right):
    return max(abs(x - y) for x, y in zip(flattened(left), flattened(right)))


def reference():
    source = json.loads(
        (WORK / "results/two_hidden_units_run.json").read_text(encoding="utf-8")
    )
    checkpoints = source["seed_1_run"]["checkpoints"]
    start = checkpoints[0]
    after = checkpoints[1]
    rows = start["predictions"]
    x = np.array([row["features"] for row in rows], dtype=np.float64)
    y = np.array([row["target"] for row in rows], dtype=np.float64)
    assert x.tolist() == [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert y.tolist() == [-1.0, 1.0, 1.0, -1.0]
    return start, after, x, y


def numpy_manual(start, x, y):
    theta = start["parameters"]
    a = np.array(theta["a"], dtype=np.float64)
    c = np.array(theta["c"], dtype=np.float64)
    v = np.array(theta["v"], dtype=np.float64)
    b = np.float64(theta["b"])
    z = x @ a.T + c
    h = np.tanh(z)
    scores = h @ v + b
    residual = scores - y
    loss = 0.5 * np.mean(residual**2)
    influence = residual[:, None] * v[None, :] * (1 - h**2)
    gradient = {
        "a": (influence.T @ x / len(x)).tolist(),
        "c": influence.mean(axis=0).tolist(),
        "v": (h.T @ residual / len(x)).tolist(),
        "b": residual.mean().item(),
    }
    after = {
        "a": (a - RATE * np.array(gradient["a"])).tolist(),
        "c": (c - RATE * np.array(gradient["c"])).tolist(),
        "v": (v - RATE * np.array(gradient["v"])).tolist(),
        "b": (b - RATE * gradient["b"]).item(),
    }
    return {
        "shapes": {
            "x": list(x.shape), "y": list(y.shape),
            "z": list(z.shape), "h": list(h.shape), "scores": list(scores.shape),
        },
        "scores": scores.tolist(),
        "loss": loss.item(),
        "gradient": gradient,
        "after": after,
    }


def torch_manual(start, x, y):
    theta = start["parameters"]
    a = torch.tensor(theta["a"], dtype=torch.float64)
    c = torch.tensor(theta["c"], dtype=torch.float64)
    v = torch.tensor(theta["v"], dtype=torch.float64)
    b = torch.tensor(theta["b"], dtype=torch.float64)
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    assert not a.requires_grad
    z = tx @ a.T + c
    h = torch.tanh(z)
    scores = h @ v + b
    residual = scores - ty
    loss = 0.5 * residual.square().mean()
    influence = residual[:, None] * v[None, :] * (1 - h.square())
    ga = influence.T @ tx / len(tx)
    gc = influence.mean(dim=0)
    gv = h.T @ residual / len(tx)
    gb = residual.mean()
    gradient = tensor_values(ga, gc, gv, gb)
    after = tensor_values(a - RATE * ga, c - RATE * gc, v - RATE * gv, b - RATE * gb)
    return {
        "scores": scores.tolist(),
        "loss": loss.item(),
        "gradient": gradient,
        "after": after,
        "loss_requires_grad": loss.requires_grad,
    }


def torch_autograd(start, x, y):
    theta = start["parameters"]
    a = torch.tensor(theta["a"], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(theta["c"], dtype=torch.float64, requires_grad=True)
    v = torch.tensor(theta["v"], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(theta["b"], dtype=torch.float64, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)

    def loss_now():
        h = torch.tanh(tx @ a.T + c)
        scores = h @ v + b
        return 0.5 * (scores - ty).square().mean()

    loss = loss_now()
    graph_entry = type(loss.grad_fn).__name__
    loss.backward()
    gradient = tensor_values(a.grad, c.grad, v.grad, b.grad)
    with torch.no_grad():
        a -= RATE * a.grad
        c -= RATE * c.grad
        v -= RATE * v.grad
        b -= RATE * b.grad
    after = tensor_values(a, c, v, b)

    # 新前向、新反向，但不清梯度：叶参数收到的是两次之和。
    for parameter in (a, c, v, b):
        parameter.grad = None
    fresh_loss = loss_now()
    fresh_loss.backward()
    one_gradient = tensor_values(a.grad, c.grad, v.grad, b.grad)
    second_fresh_loss = loss_now()
    second_fresh_loss.backward()
    two_gradients = tensor_values(a.grad, c.grad, v.grad, b.grad)
    double_gap = max(
        abs(two - 2 * one)
        for one, two in zip(flattened(one_gradient), flattened(two_gradients))
    )
    return {
        "loss": loss.item(),
        "graph_entry": graph_entry,
        "gradient": gradient,
        "after": after,
        "two_backward_vs_twice_one_max_gap": double_gap,
    }


class TinyMLP(nn.Module):
    def __init__(self, initial, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial["a"], dtype=dtype, device=device))
        self.c = nn.Parameter(torch.tensor(initial["c"], dtype=dtype, device=device))
        self.v = nn.Parameter(torch.tensor(initial["v"], dtype=dtype, device=device))
        self.b = nn.Parameter(torch.tensor(initial["b"], dtype=dtype, device=device))

    def forward(self, x):
        hidden = torch.tanh(x @ self.a.T + self.c)
        return hidden @ self.v + self.b


def module_step(start, x, y, dtype, device):
    model = TinyMLP(start["parameters"], dtype=dtype, device=device)
    tx = torch.tensor(x, dtype=dtype, device=device)
    ty = torch.tensor(y, dtype=dtype, device=device)
    assert tx.shape == (4, 2) and ty.shape == (4,)
    names = [name for name, _ in model.named_parameters()]
    assert names == ["a", "c", "v", "b"]
    optimizer = torch.optim.SGD(model.parameters(), lr=RATE, momentum=0.0, weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = model(tx)
    assert scores.shape == ty.shape
    loss = 0.5 * (scores - ty).square().mean()
    loss.backward()
    gradient = tensor_values(model.a.grad, model.c.grad, model.v.grad, model.b.grad)
    before_step = tensor_values(model.a, model.c, model.v, model.b)
    optimizer.step()
    after = tensor_values(model.a, model.c, model.v, model.b)

    model.eval()
    eval_scores = model(tx)
    eval_still_tracks_grad = eval_scores.requires_grad
    with torch.no_grad():
        no_grad_scores = model(tx)
    eval_no_grad_equal = torch.equal(eval_scores, no_grad_scores)

    return {
        "device": str(device), "dtype": str(dtype),
        "parameter_names": names,
        "loss": loss.item(), "scores": scores.detach().cpu().tolist(),
        "gradient": gradient, "before_step": before_step, "after": after,
        "eval_still_tracks_grad": eval_still_tracks_grad,
        "no_grad_still_tracks_grad": no_grad_scores.requires_grad,
        "eval_no_grad_scores_equal": eval_no_grad_equal,
    }


def main():
    start, expected_after, x, y = reference()
    numpy_result = numpy_manual(start, x, y)
    manual_result = torch_manual(start, x, y)
    autograd_result = torch_autograd(start, x, y)
    module_cpu64 = module_step(start, x, y, torch.float64, "cpu")
    base_gradient = start["gradient"]
    base_after = expected_after["parameters"]
    comparisons = {
        "numpy_vs_python_gradient": max_gap(numpy_result["gradient"], base_gradient),
        "tensor_manual_vs_python_gradient": max_gap(manual_result["gradient"], base_gradient),
        "autograd_vs_python_gradient": max_gap(autograd_result["gradient"], base_gradient),
        "module_vs_python_gradient": max_gap(module_cpu64["gradient"], base_gradient),
        "numpy_vs_python_after_step": max_gap(numpy_result["after"], base_after),
        "tensor_manual_vs_python_after_step": max_gap(manual_result["after"], base_after),
        "autograd_vs_python_after_step": max_gap(autograd_result["after"], base_after),
        "module_vs_python_after_step": max_gap(module_cpu64["after"], base_after),
    }
    assert max(comparisons.values()) < 1e-12
    assert abs(numpy_result["loss"] - start["loss"]) < 1e-12
    assert not manual_result["loss_requires_grad"]
    assert autograd_result["two_backward_vs_twice_one_max_gap"] < 1e-12

    probe = nn.Module()
    probe.cached_tensor = torch.tensor(1.0, requires_grad=True)
    probe.registered_weight = nn.Parameter(torch.tensor(1.0))
    registration_names = [name for name, _ in probe.named_parameters()]
    assert registration_names == ["registered_weight"]

    scores = torch.tensor(module_cpu64["scores"], dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    wrong_shape = list((scores[:, None] - target).shape)
    wrong_loss = 0.5 * (scores[:, None] - target).square().mean().item()
    assert wrong_shape == [4, 4]
    assert abs(wrong_loss - module_cpu64["loss"]) > 1e-8

    variants = {
        "cpu_float32": module_step(start, x, y, torch.float32, "cpu"),
        "cuda_float64": None,
        "cuda_float32": None,
    }
    if torch.cuda.is_available():
        variants["cuda_float64"] = module_step(start, x, y, torch.float64, "cuda")
        variants["cuda_float32"] = module_step(start, x, y, torch.float32, "cuda")
        torch.cuda.synchronize()
    for value in variants.values():
        if value is not None:
            value["gradient_gap_from_cpu64"] = max_gap(
                value["gradient"], module_cpu64["gradient"]
            )
            value["after_gap_from_cpu64"] = max_gap(value["after"], module_cpu64["after"])

    report = {
        "identity": "第三章同一四张人工异或卡片、九项种子 1 初值、平均半平方损失、一步步长 0.2",
        "python_version": __import__("sys").version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_gib": (
            torch.cuda.get_device_properties(0).total_memory / 2**30
            if torch.cuda.is_available() else None
        ),
        "python_reference_loss": start["loss"],
        "numpy_manual": numpy_result,
        "tensor_manual": manual_result,
        "autograd": autograd_result,
        "module_optimizer_cpu64": module_cpu64,
        "max_gaps_from_python": comparisons,
        "plain_tensor_vs_parameter_names": registration_names,
        "broadcast_wrong_target": {"residual_shape": wrong_shape, "loss": wrong_loss},
        "device_dtype_variants": variants,
    }
    output = WORK / "results/torch_handoff_one_step.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("起点损失:", start["loss"])
    print("四种梯度与旧 Python 的最大差:", max(
        comparisons[key] for key in comparisons if "gradient" in key
    ))
    print("四种一步参数与旧 Python 的最大差:", max(
        comparisons[key] for key in comparisons if "after_step" in key
    ))
    print("错误广播形状:", wrong_shape, "错误损失:", wrong_loss)
    print("CUDA 可用:", report["cuda_available"], report["cuda_device_name"])


if __name__ == "__main__":
    main()
