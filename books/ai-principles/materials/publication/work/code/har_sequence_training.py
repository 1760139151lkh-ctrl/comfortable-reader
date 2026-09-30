"""UCI HAR 六路 128 步惯性时序：末端 16 步、RNN、GRU、LSTM。"""

from __future__ import annotations

import csv
import hashlib
import json
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
DATA = WORK / "data" / "uci_har"
RESULT = WORK / "results" / "har_sequence_training.json"
PREDICTIONS = WORK / "results" / "har_test_predictions.csv"
FIGURE = WORK / "figures" / "har_sequence_training.png"
ARMS = ("last16", "rnn", "gru", "lstm")
SEED = 20260924
EPOCHS = 12
BATCH = 128
LEARNING_RATE = 0.001
HIDDEN = 32


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LastSixteen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(16 * 6, 64), nn.ReLU(), nn.Linear(64, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -16:, :])


class SequenceClassifier(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.recurrent = cell(input_size=6, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Linear(HIDDEN, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        return self.head(output[:, -1, :])


def make_model(arm: str, device: torch.device) -> nn.Module:
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    return (LastSixteen() if arm == "last16" else SequenceClassifier(arm)).to(device)


def load_data(device: torch.device) -> dict:
    manifest_file = DATA / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if sha256(DATA / "UCI_HAR_Dataset.zip") != manifest["outer_zip_sha256"]:
        raise ValueError("UCI 外包身份不符")
    prepared = DATA / "har_sequences.npz"
    if sha256(prepared) != manifest["prepared_file_sha256"]:
        raise ValueError("时序／受试者划分身份不符")
    with np.load(prepared) as raw:
        arrays = {key: raw[key].copy() for key in raw.files}
    train_subjects = set(arrays["train_subject"][arrays["train_indices"]].tolist())
    valid_subjects = set(arrays["train_subject"][arrays["validation_indices"]].tolist())
    test_subjects = set(arrays["test_subject"].tolist())
    if train_subjects & valid_subjects or train_subjects & test_subjects or valid_subjects & test_subjects:
        raise ValueError("受试者身份跨训练／验证／测试重叠")
    if arrays["train_x"].shape != (7352, 128, 6) or arrays["test_x"].shape != (2947, 128, 6):
        raise ValueError("准备好的真实时间窗形状不符")
    mean = arrays["mean"].reshape(1, 1, 6)
    std = arrays["std"].reshape(1, 1, 6)
    # 只使用训练受试者计算过的均值／标准差；标签绝不参与缩放。
    arrays["train_x"] = (arrays["train_x"] - mean) / std
    arrays["test_x"] = (arrays["test_x"] - mean) / std
    if not np.isfinite(arrays["train_x"]).all() or not np.isfinite(arrays["test_x"]).all():
        raise ValueError("标准化后的时序有非有限数")
    return {
        "manifest": manifest,
        "x_all": torch.from_numpy(arrays["train_x"]).to(device=device, dtype=torch.float32),
        "y_all": torch.from_numpy(arrays["train_y"].astype(np.int64)).to(device),
        "x_test": torch.from_numpy(arrays["test_x"]).to(device=device, dtype=torch.float32),
        "y_test": torch.from_numpy(arrays["test_y"].astype(np.int64)).to(device),
        "test_subject": arrays["test_subject"],
        "train_indices": arrays["train_indices"],
        "validation_indices": arrays["validation_indices"],
        "example_signal": arrays["train_x"][int(arrays["train_indices"][0])],
        "example_label": int(arrays["train_y"][int(arrays["train_indices"][0])]),
        "example_subject": int(arrays["train_subject"][int(arrays["train_indices"][0])]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, indices: np.ndarray | None = None) -> dict:
    model.eval()
    if indices is None:
        indices = np.arange(len(y))
    loss_sum, correct, predictions = 0.0, 0, []
    for start in range(0, len(indices), BATCH):
        ids = torch.as_tensor(indices[start:start + BATCH], device=x.device, dtype=torch.long)
        logits = model(x.index_select(0, ids))
        target = y.index_select(0, ids)
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum").item())
        guessed = logits.argmax(dim=1)
        correct += int((guessed == target).sum().item())
        predictions.extend(guessed.cpu().tolist())
    return {
        "loss": loss_sum / len(indices),
        "accuracy": correct / len(indices),
        "correct": correct,
        "total": len(indices),
        "predictions": predictions,
    }


def train_one(arm: str, data: dict, orders: list[np.ndarray], device: torch.device) -> dict:
    model = make_model(arm, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x, y = data["x_all"], data["y_all"]
    train_indices = data["train_indices"]
    validation_indices = data["validation_indices"]
    history = []
    best_loss, best_epoch, best_state = float("inf"), None, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch, order in enumerate(orders, start=1):
        model.train()
        loss_sum, correct = 0.0, 0
        ordered_indices = train_indices[order]
        for start in range(0, len(ordered_indices), BATCH):
            ids = torch.as_tensor(ordered_indices[start:start + BATCH], device=device, dtype=torch.long)
            bx, by = x.index_select(0, ids), y.index_select(0, ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(ids)
            correct += int((logits.detach().argmax(dim=1) == by).sum().item())
        validation = evaluate(model, x, y, validation_indices)
        entry = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_indices),
            "train_accuracy": correct / len(train_indices),
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
        raise AssertionError("没有经过验证选择的模型")
    checkpoint = WORK / "results" / f"har_{arm}_best.pt"
    torch.save(
        {
            "arm": arm,
            "chosen_epoch": best_epoch,
            "seed": SEED,
            "data_sha256": data["manifest"]["prepared_file_sha256"],
            "state_dict": best_state,
        },
        checkpoint,
    )
    reloaded = make_model(arm, device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
    model.load_state_dict(best_state)
    model.eval()
    reloaded.eval()
    first = int(validation_indices[0])
    with torch.no_grad():
        if not torch.equal(model(x[first:first + 1]), reloaded(x[first:first + 1])):
            raise AssertionError("权重保存恢复后验证样本预测不一致")
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "chosen_epoch": best_epoch,
        "chosen_validation_loss": best_loss,
        "train_seconds": seconds,
        "cuda_peak_mib_including_resident_data": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "checkpoint": str(checkpoint.relative_to(WORK)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "_model": reloaded,
    }


def draw(data: dict, outcomes: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    example = data["example_signal"]
    time_axis = np.arange(128) / 50
    for channel in range(3):
        axes[0].plot(time_axis, example[:, channel], lw=0.8, label=data["manifest"]["channels"][channel])
    axes[0].set(xlabel="2.56 秒窗口内的时间", ylabel="训练期标准化的加速度",
                title=f"受试者 {data['example_subject']}，活动类别 {data['example_label']}")
    axes[0].legend(fontsize=6)
    for arm in ARMS:
        curve = [entry["validation_accuracy"] for entry in outcomes[arm]["history"]]
        axes[1].plot(range(1, EPOCHS + 1), curve, marker="o", markersize=2, label=("末 16 步" if arm == "last16" else arm.upper()))
    axes[1].set(xlabel="训练轮次", ylabel="验证准确率", xticks=range(1, EPOCHS + 1), title="完整受试者留出的验证")
    axes[1].legend(fontsize=8)
    values = [outcomes[arm]["official_test"]["accuracy"] for arm in ARMS]
    axes[2].bar(["末 16 步", "RNN", "GRU", "LSTM"], values, color=["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1"])
    axes[2].set(ylabel="官方测试准确率", ylim=(0, 1), title="官方测试受试者")
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data = load_data(device)
    rng = np.random.default_rng(SEED + 77)
    orders = [rng.permutation(len(data["train_indices"])) for _ in range(EPOCHS)]
    outcomes = {arm: train_one(arm, data, orders, device) for arm in ARMS}

    # 全部四组已选完；此时才评价从未参与选择的官方测试受试者。
    true_labels = data["y_test"].cpu().numpy()
    predictions = {}
    for arm in ARMS:
        selected = outcomes[arm].pop("_model")
        test = evaluate(selected, data["x_test"], data["y_test"])
        prediction = np.asarray(test.pop("predictions"), dtype=np.int64)
        confusion = np.zeros((6, 6), dtype=np.int64)
        np.add.at(confusion, (true_labels, prediction), 1)
        test["confusion_true_rows_predicted_columns"] = confusion.tolist()
        test["per_class_accuracy"] = (np.diag(confusion) / confusion.sum(axis=1)).tolist()
        outcomes[arm]["official_test"] = test
        predictions[arm] = prediction
        print(f"{arm} selected epoch {outcomes[arm]['chosen_epoch']}: "
              f"official test {test['accuracy']:.4f}", flush=True)

    RESULT.parent.mkdir(exist_ok=True)
    FIGURE.parent.mkdir(exist_ok=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["official_test_window_zero_based", "subject_id", "true_label_zero_based", *ARMS])
        for row, label in enumerate(true_labels):
            writer.writerow([row, int(data["test_subject"][row]), int(label), *[int(predictions[arm][row]) for arm in ARMS]])
    record = {
        "purpose": "real pre-windowed smartphone inertial signals: activity class, not free-running continuous prediction",
        "data_manifest": str((DATA / "manifest.json").relative_to(WORK)).replace("\\", "/"),
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "training_rows": len(data["train_indices"]),
        "validation_rows": len(data["validation_indices"]),
        "official_test_rows": len(true_labels),
        "train_subject_ids": data["manifest"]["train_subject_ids"],
        "validation_subject_ids": data["manifest"]["validation_subject_ids"],
        "test_subject_ids": data["manifest"]["official_test_subject_ids"],
        "channel_order": data["manifest"]["channels"],
        "normalization": "per-channel mean/std from training subjects only",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "selection": "lowest validation cross-entropy, earlier epoch on tie",
        "hidden_width_for_recurrent": HIDDEN,
        "last_window_steps": 16,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "arms": outcomes,
        "test_predictions_csv": str(PREDICTIONS.relative_to(WORK)).replace("\\", "/"),
        "test_predictions_csv_sha256": sha256(PREDICTIONS),
        "limits": "one seed, 12 epochs, original 128-step preprocessed 50%-overlap windows, subject-level split; cannot establish arbitrary long memory",
    }
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(data, outcomes)
    print(f"saved {RESULT}, {PREDICTIONS}, {FIGURE}", flush=True)


if __name__ == "__main__":
    main()
