"""Small, explicit MAML versus pooled pretraining on analytic sine tasks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.func import functional_call

INNER_LR = 0.01
OUTER_LR = 0.001
SEED = 4605


def model_factory() -> nn.Module:
    return nn.Sequential(nn.Linear(1, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, 1))


def draw_task(rng: np.random.Generator, query_count: int = 20) -> dict:
    amp = float(rng.uniform(0.1, 5.0))
    phase = float(rng.uniform(0, np.pi))
    support_x = rng.uniform(-5, 5, size=(10, 1)).astype(np.float32)
    query_x = rng.uniform(-5, 5, size=(query_count, 1)).astype(np.float32)
    return {"amplitude": amp, "phase": phase, "support_x": support_x, "query_x": query_x}


def tensors(task: dict, device: str, grid_query: bool = False):
    sx = torch.tensor(task["support_x"], device=device)
    qx = torch.linspace(-5, 5, 100, device=device).unsqueeze(1) if grid_query else torch.tensor(task["query_x"], device=device)
    sy = task["amplitude"] * torch.sin(sx + task["phase"])
    qy = task["amplitude"] * torch.sin(qx + task["phase"])
    return sx, sy, qx, qy


def one_task_meta_loss(model: nn.Module, task: dict, device: str) -> torch.Tensor:
    sx, sy, qx, qy = tensors(task, device)
    parameters = OrderedDict(model.named_parameters())
    support_loss = torch.mean((functional_call(model, parameters, (sx,)) - sy).square())
    derivatives = torch.autograd.grad(support_loss, tuple(parameters.values()), create_graph=True)
    adapted = OrderedDict((name, value - INNER_LR * grad)
                          for (name, value), grad in zip(parameters.items(), derivatives))
    return torch.mean((functional_call(model, adapted, (qx,)) - qy).square())


def train_arm(kind: str, initial: dict, device: str, steps: int) -> tuple[nn.Module, list, float]:
    model = model_factory().to(device)
    model.load_state_dict(copy.deepcopy(initial))
    rng = np.random.default_rng(SEED + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=OUTER_LR)
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        tasks = [draw_task(rng) for _ in range(4)]
        optimizer.zero_grad(set_to_none=True)
        if kind == "maml":
            loss = sum(one_task_meta_loss(model, task, device) for task in tasks) / 4
        else:
            terms = []
            for task in tasks:
                _, _, qx, qy = tensors(task, device)
                terms.append(torch.mean((model(qx) - qy).square()))
            loss = sum(terms) / 4
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {kind} outer loss at {step}")
        loss.backward()
        optimizer.step()
        if step in (1, 100, 200, 400, 600, 800, steps):
            history.append({"step": step, "batch_objective": float(loss.detach())})
            print(kind, step, round(float(loss.detach()), 4), flush=True)
    return model, history, time.perf_counter() - start


def adapt_predictions(model: nn.Module, task: dict, device: str) -> tuple[dict, dict]:
    sx, sy, qx, qy = tensors(task, device, grid_query=True)
    parameters = OrderedDict((name, value.detach().clone().requires_grad_(True))
                             for name, value in model.named_parameters())
    losses = {}
    predictions = {}
    for step in range(6):
        with torch.no_grad():
            pred = functional_call(model, parameters, (qx,)).detach()
            losses[str(step)] = float(torch.mean((pred - qy).square()))
            if step in (0, 1, 5):
                predictions[str(step)] = pred.cpu().numpy().ravel().tolist()
        if step < 5:
            support_loss = torch.mean((functional_call(model, parameters, (sx,)) - sy).square())
            gradients = torch.autograd.grad(support_loss, tuple(parameters.values()))
            parameters = OrderedDict((name, (value - INNER_LR * grad).detach().requires_grad_(True))
                                     for (name, value), grad in zip(parameters.items(), gradients))
    return losses, predictions


def assess(models: dict, seed: int, device: str) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    tasks = [draw_task(rng) for _ in range(64)]
    scores = {}
    first = {}
    for name, model in models.items():
        rows = []
        for index, task in enumerate(tasks):
            losses, curves = adapt_predictions(model, task, device)
            rows.append(losses)
            if index == 0:
                first[name] = curves
        scores[name] = {
            key: {"mean_mse": float(np.mean([row[key] for row in rows])),
                  "median_mse": float(np.median([row[key] for row in rows])),
                  "q90_mse": float(np.quantile([row[key] for row in rows], .9))}
            for key in ("0", "1", "5")
        }
    first["truth"] = (tasks[0]["amplitude"] * np.sin(np.linspace(-5, 5, 100) + tasks[0]["phase"])).tolist()
    first["amplitude"] = tasks[0]["amplitude"]
    first["phase"] = tasks[0]["phase"]
    first["support_x"] = tasks[0]["support_x"].ravel().tolist()
    first["support_y"] = (tasks[0]["amplitude"] * np.sin(tasks[0]["support_x"].ravel() + tasks[0]["phase"])).tolist()
    return scores, first


def main(out_dir: Path, steps: int) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    random_model = model_factory().to(device)
    initial = copy.deepcopy(random_model.state_dict())
    maml, maml_history, maml_seconds = train_arm("maml", initial, device, steps)
    pooled, pooled_history, pooled_seconds = train_arm("pooled", initial, device, steps)
    models = {"random": random_model, "pooled": pooled, "maml": maml}
    validation, _ = assess(models, 4606, device)
    test, first = assess(models, 4607, device)
    png = out_dir / "first_heldout_sine_task.png"
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    xx = np.linspace(-5, 5, 100)
    ax.plot(xx, first["truth"], color="black", linewidth=2, label="analytic task")
    for name, color in (("random", "#999999"), ("pooled", "#c36d2b"), ("maml", "#226c96")):
        ax.plot(xx, first[name]["1"], color=color, linewidth=1.5, label=f"{name}: after one update")
    ax.scatter(first["support_x"], first["support_y"], color="black", s=25, marker="x", label="10 support examples")
    ax.set(xlabel="input x", ylabel="output", title="First preselected held-out sine task")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(png, dpi=170)
    plt.close(fig)
    ckpt = out_dir / "models.pt"
    torch.save({"random": {k: v.cpu() for k, v in random_model.state_dict().items()},
                "pooled": {k: v.cpu() for k, v in pooled.state_dict().items()},
                "maml": {k: v.cpu() for k, v in maml.state_dict().items()}, "steps": steps}, ckpt)
    report = {
        "identity": "Author-generated sine task family; no real measurements or language model meta-training",
        "source_paper": "https://arxiv.org/html/1703.03400",
        "task_family": "A*sin(x+phase), A uniform [0.1,5], phase uniform [0,pi], support/query x uniform [-5,5]",
        "support_n": 10, "train_query_n": 20, "eval_query_grid_n": 100,
        "outer_steps": steps, "tasks_per_outer_step": 4, "inner_lr": INNER_LR, "outer_lr": OUTER_LR,
        "device": device, "maml_seconds": maml_seconds, "pooled_seconds": pooled_seconds,
        "maml_history": maml_history, "pooled_history": pooled_history,
        "validation_task_seed": 4606, "test_task_seed": 4607,
        "validation_64_tasks": validation, "test_64_tasks": test,
        "first_test_task_curves": first,
        "model_checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "figure_sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "test": test, "figure_sha256": report["figure_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    main(args.out_dir, args.steps)
