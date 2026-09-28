"""第四章：两条小批次训练与一次真正从新进程恢复的比较。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_handoff_one_step import TinyMLP, reference


WORK = Path(__file__).resolve().parents[1]
OUT = WORK / "results"
CHECKPOINT = OUT / "tiny_mlp_epoch50.pt"
RATE = 0.05
MOMENTUM = 0.9
SPLIT_EPOCH = 50
TOTAL_EPOCHS = 200


def context():
    start, _, x_numpy, y_numpy = reference()
    x = torch.tensor(x_numpy, dtype=torch.float64)
    y = torch.tensor(y_numpy, dtype=torch.float64)
    ids = torch.arange(len(y))
    dataset = TensorDataset(x, y, ids)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    return start["parameters"], x, y, loader


def model_and_optimizer(initial):
    model = TinyMLP(initial, dtype=torch.float64, device="cpu")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=RATE, momentum=MOMENTUM, weight_decay=0.0
    )
    return model, optimizer


def parameter_values(model):
    return {
        name: value.detach().cpu().tolist()
        for name, value in model.named_parameters()
    }


def momentum_size(optimizer):
    buffers = [
        item["momentum_buffer"]
        for item in optimizer.state.values()
        if "momentum_buffer" in item
    ]
    return sum(buffer.square().sum().item() for buffer in buffers) ** 0.5


def train(model, optimizer, loader, first_epoch, last_epoch):
    batches = []
    model.train()
    for epoch in range(first_epoch, last_epoch + 1):
        for batch_number, (xb, yb, row_ids) in enumerate(loader, start=1):
            assert xb.shape == (2, 2) and yb.shape == (2,)
            optimizer.zero_grad(set_to_none=True)
            scores = model(xb)
            assert scores.shape == yb.shape
            loss = 0.5 * (scores - yb).square().mean()
            batches.append({
                "epoch": epoch,
                "batch": batch_number,
                "row_ids": row_ids.tolist(),
                "loss_before_step": loss.item(),
            })
            loss.backward()
            optimizer.step()
    return batches


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        scores = model(x)
        loss = 0.5 * (scores - y).square().mean()
    return {
        "scores": scores.tolist(),
        "mean_square_loss": loss.item(),
        "classification_errors": sum(
            (1 if score > 0 else -1) != target
            for score, target in zip(scores.tolist(), y.tolist())
        ),
    }


def save_json(name, data):
    path = OUT / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_branch(mode):
    initial, x, y, loader = context()
    model, optimizer = model_and_optimizer(initial)
    loaded_momentum = 0.0
    if mode in ("resume", "weights_only"):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        assert saved["epoch"] == SPLIT_EPOCH
        model.load_state_dict(saved["model_state_dict"])
        if mode == "resume":
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        loaded_momentum = momentum_size(optimizer)
        first = SPLIT_EPOCH + 1
        last = TOTAL_EPOCHS
    else:
        first = 1
        last = TOTAL_EPOCHS if mode == "full" else SPLIT_EPOCH
    batches = train(model, optimizer, loader, first, last)
    if mode == "part":
        torch.save({
            "epoch": SPLIT_EPOCH,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_identity": "the four XOR rows, fixed order, batch_size=2",
        }, CHECKPOINT)
    result = {
        "mode": mode,
        "first_epoch": first,
        "last_epoch": last,
        "batch_count": len(batches),
        "batches": batches,
        "loaded_momentum_norm": loaded_momentum,
        "final_momentum_norm": momentum_size(optimizer),
        "parameters": parameter_values(model),
        "training_set_evaluation": evaluate(model, x, y),
    }
    save_json(f"tiny_mlp_{mode}.json", result)
    print(mode, "batches", len(batches), "train_loss",
          result["training_set_evaluation"]["mean_square_loss"],
          "mistakes", result["training_set_evaluation"]["classification_errors"])


def max_parameter_gap(left, right):
    gaps = []
    for name in ("a", "c", "v", "b"):
        l = torch.tensor(left[name], dtype=torch.float64)
        r = torch.tensor(right[name], dtype=torch.float64)
        gaps.append((l - r).abs().max().item())
    return max(gaps)


def first_step_momentum_probe():
    """在第 51 轮第一批精确核对丢失动量造成的参数距离。"""
    initial, _, _, loader = context()
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    complete_model, complete_optim = model_and_optimizer(initial)
    weights_model, weights_optim = model_and_optimizer(initial)
    complete_model.load_state_dict(saved["model_state_dict"])
    weights_model.load_state_dict(saved["model_state_dict"])
    complete_optim.load_state_dict(saved["optimizer_state_dict"])
    old_momentum_norm = momentum_size(complete_optim)
    xb, yb, _ = next(iter(loader))
    for model, optimizer in ((complete_model, complete_optim),
                             (weights_model, weights_optim)):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * (model(xb) - yb).square().mean()
        loss.backward()
    for complete, weights in zip(complete_model.parameters(),
                                 weights_model.parameters()):
        assert torch.equal(complete.grad, weights.grad)
    complete_optim.step()
    weights_optim.step()
    squared_gap = sum(
        (complete - weights).detach().square().sum().item()
        for complete, weights in zip(complete_model.parameters(),
                                     weights_model.parameters())
    )
    actual_gap = squared_gap ** 0.5
    predicted_gap = RATE * MOMENTUM * old_momentum_norm
    assert abs(actual_gap - predicted_gap) < 1e-12
    return actual_gap, predicted_gap


def compare():
    def read(mode):
        return json.loads((OUT / f"tiny_mlp_{mode}.json").read_text(encoding="utf-8"))

    full, part, resumed, weights_only = (
        read("full"), read("part"), read("resume"), read("weights_only")
    )
    stitched = part["batches"] + resumed["batches"]
    assert len(full["batches"]) == TOTAL_EPOCHS * 2
    assert full["batches"] == stitched
    exact_parameter_gap = max_parameter_gap(full["parameters"], resumed["parameters"])
    assert exact_parameter_gap == 0.0
    assert resumed["loaded_momentum_norm"] == part["final_momentum_norm"]
    assert weights_only["loaded_momentum_norm"] == 0.0
    first_loss_gap = abs(
        full["batches"][SPLIT_EPOCH * 2]["loss_before_step"]
        - weights_only["batches"][0]["loss_before_step"]
    )
    next_loss_gap = abs(
        full["batches"][SPLIT_EPOCH * 2 + 1]["loss_before_step"]
        - weights_only["batches"][1]["loss_before_step"]
    )
    assert first_loss_gap == 0.0 and next_loss_gap > 0.0
    first_step_gap, predicted_gap = first_step_momentum_probe()
    result = {
        "identity": "第三章同一九参数 XOR 网络；CPU float64；batch_size=2；顺序固定；SGD lr=0.05 momentum=0.9",
        "full_batches": len(full["batches"]),
        "first_process_batches": len(part["batches"]),
        "resumed_process_batches": len(resumed["batches"]),
        "all_batch_records_exactly_match": full["batches"] == stitched,
        "final_parameter_gap_full_vs_restored": exact_parameter_gap,
        "momentum_norm_at_checkpoint": part["final_momentum_norm"],
        "momentum_norm_after_loading_both_states": resumed["loaded_momentum_norm"],
        "momentum_norm_after_loading_weights_only": weights_only["loaded_momentum_norm"],
        "weights_only_first_resumed_batch_loss_gap": first_loss_gap,
        "weights_only_second_resumed_batch_loss_gap": next_loss_gap,
        "first_step_parameter_l2_gap_from_missing_momentum": first_step_gap,
        "first_step_predicted_l2_gap": predicted_gap,
        "weights_only_final_parameter_gap": max_parameter_gap(
            full["parameters"], weights_only["parameters"]
        ),
        "full_training_set_evaluation": full["training_set_evaluation"],
        "checkpoint_path": "work/results/tiny_mlp_epoch50.pt",
        "scope": "Only a whole-epoch checkpoint in this CPU process/data order; no shuffle, workers, random layers or mid-batch recovery.",
    }
    save_json("torch_training_resume.json", result)
    print("连续与完整恢复：400 批记录和最终参数逐项相同")
    print("只读权重：首个恢复批次损失差", first_loss_gap,
          "第二批次损失差", next_loss_gap)
    print("只读权重最终参数最大差", result["weights_only_final_parameter_gap"])
    print("首个恢复批次：动量缺失造成的参数距离与公式值", first_step_gap, predicted_gap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "full", "part", "resume",
                                           "weights_only", "compare"], default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.mode == "all":
        for mode in ("full", "part", "resume", "weights_only", "compare"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--mode", mode], check=True)
    elif args.mode == "compare":
        compare()
    else:
        run_branch(args.mode)


if __name__ == "__main__":
    main()
