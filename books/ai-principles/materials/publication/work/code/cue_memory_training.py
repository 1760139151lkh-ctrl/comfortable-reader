"""平衡开头线索／独立干扰任务：末段窗口、普通 RNN、GRU、LSTM。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / "results" / "cue_memory_training.json"
FIGURE = WORK / "figures" / "cue_memory_training.png"
ARMS = ("last8", "rnn", "gru", "lstm")
SEED = 20260924
TRAIN_N = 16_384
VALID_N = 4_096
TEST_N = 4_096
TRAIN_LENGTH = 24
TEST_LENGTHS = (24, 80)
BATCH = 256
EPOCHS = 20
LEARNING_RATE = 0.003
HIDDEN = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_data(count: int, length: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cue = rng.integers(0, 2, size=count, dtype=np.int8)
    later = rng.integers(0, 2, size=(count, length - 1), dtype=np.int8)
    bits = np.column_stack((cue, later)).astype(np.float32)
    value = 2 * bits - 1
    marker = np.zeros_like(value)
    marker[:, 0] = 1
    x = np.stack((value, marker), axis=2)
    return x, cue.astype(np.int64)


class LastEight(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8 * 2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -8:, :].flatten(1))


class RecurrentClassifier(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.recurrent = cell(input_size=2, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        return self.head(sequence[:, -1, :])


def make_model(arm: str, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    return (LastEight() if arm == "last8" else RecurrentClassifier(arm)).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    loss_sum, correct = 0.0, 0
    for start in range(0, len(y), BATCH):
        bx, by = x[start:start + BATCH], y[start:start + BATCH]
        logits = model(bx)
        loss_sum += float(F.cross_entropy(logits, by, reduction="sum").item())
        correct += int((logits.argmax(dim=1) == by).sum().item())
    return {"loss": loss_sum / len(y), "accuracy": correct / len(y), "correct": correct, "total": len(y)}


def analytic_gradient_probe() -> dict:
    results = {}
    for a in (0.8, 1.0, 1.2):
        values = {}
        for length in (8, 32, 64):
            initial = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            hidden = initial
            for _ in range(length):
                hidden = torch.tanh(a * hidden)
            derivative = torch.autograd.grad(hidden, initial)[0].item()
            expected = a**length
            if not math.isclose(derivative, expected, rel_tol=1e-11, abs_tol=1e-15):
                raise AssertionError("全零轨迹上的时间导数与 a^T 不符")
            values[str(length)] = {"autograd": derivative, "closed_form": expected}
        results[str(a)] = values
    # 非零轨迹中的 tanh' 因子会改变结论；保存一个直接可验的例子。
    initial = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    hidden = initial
    for _ in range(32):
        hidden = torch.tanh(1.2 * hidden)
    nonzero = torch.autograd.grad(hidden, initial)[0].item()
    return {"zero_path": results, "a_1_2_nonzero_h0_0_7_length32_gradient": nonzero}


def train_one(arm: str, data: dict, orders: list[np.ndarray], device: torch.device) -> dict:
    model = make_model(arm, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_train, y_train = data["train"]
    x_valid, y_valid = data["valid"]
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch, order in enumerate(orders, start=1):
        model.train()
        train_loss, correct = 0.0, 0
        for start in range(0, len(order), BATCH):
            ids = torch.as_tensor(order[start:start + BATCH], device=device)
            bx, by = x_train.index_select(0, ids), y_train.index_select(0, ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().item()) * len(ids)
            correct += int((logits.detach().argmax(dim=1) == by).sum().item())
        validation = evaluate(model, x_valid, y_valid)
        entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(y_train),
            "train_accuracy": correct / len(y_train),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_correct": validation["correct"],
        }
        history.append(entry)
        print(f"{arm} epoch {epoch}: train {entry['train_accuracy']:.4f}, "
              f"valid {entry['validation_accuracy']:.4f}, loss {entry['validation_loss']:.4f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if best_state is None:
        raise AssertionError("没有最佳模型")
    path = WORK / "results" / f"cue_{arm}_best.pt"
    torch.save({"arm": arm, "epoch": best_epoch, "seed": SEED, "state_dict": best_state}, path)
    reloaded = make_model(arm, device)
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    model.load_state_dict(best_state)
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        if not torch.equal(model(x_valid[:1]), reloaded(x_valid[:1])):
            raise AssertionError("保存恢复在验证串上的 logits 不符")
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "chosen_epoch": best_epoch,
        "chosen_validation_loss": best_loss,
        "train_seconds": seconds,
        "cuda_peak_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "checkpoint": str(path.relative_to(WORK)).replace("\\", "/"),
        "checkpoint_sha256": sha256(path),
        "_model": reloaded,
    }


def draw(probe: dict, outcomes: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), layout="constrained")
    for a in ("0.8", "1.0", "1.2"):
        x = np.array([8, 32, 64])
        y = np.array([abs(probe["zero_path"][a][str(n)]["autograd"]) for n in x])
        axes[0].plot(x, y, marker="o", label=f"a={a}")
    axes[0].set(xlabel="时间步数", ylabel="|∂h_T/∂h_0|（全零轨迹）", yscale="log", title="标量递推")
    axes[0].legend(fontsize=8)
    for arm in ARMS:
        curve = [entry["validation_accuracy"] for entry in outcomes[arm]["history"]]
        axes[1].plot(range(1, EPOCHS + 1), curve, label=("末八步" if arm == "last8" else arm.upper()))
    axes[1].set(xlabel="训练轮次", ylabel="验证准确率", ylim=(0.4, 1.02), title="首步给线索，末尾才作答")
    axes[1].legend(fontsize=8)
    positions = np.arange(len(ARMS))
    for offset, length in ((-0.18, 24), (0.18, 80)):
        values = [outcomes[arm]["held_out"][str(length)]["accuracy"] for arm in ARMS]
        axes[2].bar(positions + offset, values, width=0.35, label=f"长度 {length}")
    axes[2].axhline(0.5, color="black", linestyle=":", lw=0.8)
    axes[2].set(xticks=positions, xticklabels=["末八步", "RNN", "GRU", "LSTM"], ylabel="新串准确率", ylim=(0, 1.03), title="选定权重在新长度上")
    axes[2].legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    probe = analytic_gradient_probe()
    train_x, train_y = make_data(TRAIN_N, TRAIN_LENGTH, SEED + 101)
    valid_x, valid_y = make_data(VALID_N, TRAIN_LENGTH, SEED + 202)
    if not np.array_equal(np.bincount(train_y, minlength=2).sum(), TRAIN_N):
        raise AssertionError("训练标签异常")
    data = {
        "train": (torch.from_numpy(train_x).to(device), torch.from_numpy(train_y).to(device)),
        "valid": (torch.from_numpy(valid_x).to(device), torch.from_numpy(valid_y).to(device)),
    }
    order_rng = np.random.default_rng(SEED + 303)
    orders = [order_rng.permutation(TRAIN_N) for _ in range(EPOCHS)]
    outcomes = {arm: train_one(arm, data, orders, device) for arm in ARMS}
    tests = {}
    for length in TEST_LENGTHS:
        x, y = make_data(TEST_N, length, SEED + 404 + length)
        tests[str(length)] = (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
    for arm in ARMS:
        selected = outcomes[arm].pop("_model")
        outcomes[arm]["held_out"] = {
            length: evaluate(selected, *tensors) for length, tensors in tests.items()
        }
        print(f"{arm} chosen epoch {outcomes[arm]['chosen_epoch']}: "
              f"new length24 {outcomes[arm]['held_out']['24']['accuracy']:.4f}, "
              f"new length80 {outcomes[arm]['held_out']['80']['accuracy']:.4f}", flush=True)

    RESULT.parent.mkdir(exist_ok=True)
    FIGURE.parent.mkdir(exist_ok=True)
    record = {
        "purpose": "controlled first-bit cue with independent later distractors; does not represent natural language or activity sensing",
        "train_count": TRAIN_N,
        "validation_count": VALID_N,
        "new_test_count_per_length": TEST_N,
        "train_length": TRAIN_LENGTH,
        "new_test_lengths": TEST_LENGTHS,
        "last_window": 8,
        "analytic_last8_optimal_accuracy_for_length_gt8": 0.5,
        "seed": SEED,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "selection": "lowest validation cross-entropy, earlier epoch on tie",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gradient_probe": probe,
        "arms": outcomes,
    }
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(probe, outcomes)
    print(f"saved {RESULT} and {FIGURE}", flush=True)


if __name__ == "__main__":
    main()
