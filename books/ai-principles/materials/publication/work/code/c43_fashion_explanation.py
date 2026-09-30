"""Explain and stress one already-trained Fashion CNN on source-identified images.

This is post-publication analysis of C13's model and previously opened official
test set. A heatmap or an activation swap is never a human-concept proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fashion_spatial_training import CNN
from prepare_fashion_mnist import DATA, load_images, load_labels


WORK = Path(__file__).resolve().parents[1]
RUNS = WORK / "runs"
CHECKPOINT = WORK / "results/fashion_cnn_original_best.pt"
CHECKPOINT_SHA = "cf21ae82864d5f4b5ba9fbbb76da34a3a6908b9509f19aafb5b74a032cfc5c73"
SPLIT_SHA = "ac870595933df8a373507f0728aa2d275645e05ac53a052556995c49fe43ea9b"
CLASS_NAMES = ["T恤/上衣", "裤子", "套衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
TEMPERATURES = np.exp(np.linspace(math.log(0.25), math.log(4.0), 101))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        if sha(DATA / name) != item["sha256"]:
            raise RuntimeError(f"official data SHA changed: {name}")
    split = DATA / "chapter_split.npz"
    if sha(split) != SPLIT_SHA or manifest["split_file_sha256"] != SPLIT_SHA:
        raise RuntimeError("C13 split changed")
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("C13 checkpoint changed")
    with np.load(split) as prepared:
        val_ids = prepared["validation_indices"].astype(np.int64)
    if val_ids.size != 10000 or np.unique(val_ids).size != 10000:
        raise RuntimeError("unexpected C13 validation indices")
    train_pixels = load_images(DATA / "train-images-idx3-ubyte.gz", 60000)
    train_labels = load_labels(DATA / "train-labels-idx1-ubyte.gz", 60000)
    test_pixels = load_images(DATA / "t10k-images-idx3-ubyte.gz", 10000)
    test_labels = load_labels(DATA / "t10k-labels-idx1-ubyte.gz", 10000)
    val_x = torch.from_numpy(train_pixels[val_ids].copy()).float().unsqueeze(1) / 255
    val_y = torch.from_numpy(train_labels[val_ids].astype(np.int64).copy())
    test_x = torch.from_numpy(test_pixels.copy()).float().unsqueeze(1) / 255
    test_y = torch.from_numpy(test_labels.astype(np.int64).copy())
    model = CNN()
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved["arm"] != "cnn_original" or saved["data_split_sha256"] != SPLIT_SHA:
        raise RuntimeError("C13 arm or data identity mismatch")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, val_x, val_y, test_x, test_y, val_ids, int(saved["chosen_epoch"])


@torch.no_grad()
def logits_of(model: nn.Module, xs: torch.Tensor, batch: int = 512) -> torch.Tensor:
    return torch.cat([model(xs[i:i + batch]) for i in range(0, len(xs), batch)], 0)


def metrics(logits: torch.Tensor, labels: torch.Tensor, temperature: float):
    p = torch.softmax(logits / temperature, dim=1).numpy()
    y = labels.numpy()
    predicted = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = predicted == y
    chosen = np.maximum(p[np.arange(len(y)), y], 1e-30)
    one_hot = np.eye(10, dtype=float)[y]
    ece = 0.0
    bins = []
    for number in range(10):
        members = np.minimum((confidence * 10).astype(int), 9) == number
        count = int(members.sum())
        if count:
            accuracy = float(correct[members].mean())
            conf = float(confidence[members].mean())
            ece += count / len(y) * abs(accuracy - conf)
        else:
            accuracy = conf = None
        bins.append({"range": [number / 10, (number + 1) / 10], "n": count,
                     "accuracy": accuracy, "mean_confidence": conf})
    return {"n": len(y), "correct": int(correct.sum()),
            "accuracy": float(correct.mean()), "nll_nats": float(-np.log(chosen).mean()),
            "brier_sum_classes": float(np.square(p - one_hot).sum(axis=1).mean()),
            "ece_10_bins": float(ece), "mean_confidence": float(confidence.mean()),
            "wrong_mean_confidence": float(confidence[~correct].mean()) if (~correct).any() else None,
            "bins": bins}


def choose_case(model, val_x, val_y, val_ids, val_logits):
    subset = 2048
    scores = torch.softmax(val_logits[:subset], dim=1)
    prob, pred = scores.max(dim=1)
    wrong = [i for i in range(subset) if int(pred[i]) != int(val_y[i])]
    if not wrong:
        raise RuntimeError("no error in predeclared validation subset")
    target_i = min(wrong, key=lambda i: (-float(prob[i]), int(val_ids[i])))
    truth = int(val_y[target_i])
    source_candidates = [i for i in range(subset) if i != target_i and
                         int(val_y[i]) == truth and int(pred[i]) == truth]
    if not source_candidates:
        raise RuntimeError("no correct source of target true class in predeclared subset")
    target_pixels = val_x[target_i].numpy()
    source_i = min(source_candidates, key=lambda i: (
        float(np.square(val_x[i].numpy() - target_pixels).sum()), int(val_ids[i])))
    x = val_x[target_i:target_i + 1].clone().requires_grad_(True)
    original_logits = model(x)[0]
    wrong_class = int(pred[target_i])
    margin = original_logits[wrong_class] - original_logits[truth]
    gradient = torch.autograd.grad(margin, x)[0].detach()[0, 0]
    original_margin = float(margin.item())
    target = val_x[target_i:target_i + 1]
    source = val_x[source_i:source_i + 1]
    masks = target.repeat(49, 1, 1, 1)
    for cell in range(49):
        row, col = divmod(cell, 7)
        masks[cell, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = 0
    with torch.no_grad():
        masked = model(masks)
        occlusion_delta = ((masked[:, wrong_class] - masked[:, truth]) - original_margin).view(7, 7)
        target_h = model.features(target)
        source_h = model.features(source)
        if abs(float((model.classifier(target_h)[0, wrong_class] -
                      model.classifier(target_h)[0, truth]).item()) - original_margin) > 1e-5:
            raise RuntimeError("feature/classifier split not equal to direct model")
        patched = target_h.repeat(49, 1, 1, 1)
        zeroed = target_h.repeat(49, 1, 1, 1)
        for cell in range(49):
            row, col = divmod(cell, 7)
            patched[cell, :, row, col] = source_h[0, :, row, col]
            zeroed[cell, :, row, col] = 0
        patched_logits = model.classifier(patched)
        zeroed_logits = model.classifier(zeroed)
        patch_delta = ((patched_logits[:, wrong_class] - patched_logits[:, truth]) - original_margin).view(7, 7)
        zero_delta = ((zeroed_logits[:, wrong_class] - zeroed_logits[:, truth]) - original_margin).view(7, 7)
    min_patch = int(torch.argmin(patch_delta).item())
    min_occ = int(torch.argmin(occlusion_delta).item())
    best_patch_logits = patched_logits[min_patch]
    best_occ_logits = masked[min_occ]
    grad_abs = gradient.abs().numpy()
    grad_blocks = grad_abs.reshape(7, 4, 7, 4).sum(axis=(1, 3))
    occlusion_np = occlusion_delta.numpy()
    return {
        "selection": "first 2048 C13 validation indices: highest wrong-class softmax among errors; source nearest L2 correctly classified true-class image",
        "validation_position": target_i, "original_training_row": int(val_ids[target_i]),
        "source_validation_position": source_i, "source_original_training_row": int(val_ids[source_i]),
        "true_class": truth, "true_class_name": CLASS_NAMES[truth],
        "predicted_class": wrong_class, "predicted_class_name": CLASS_NAMES[wrong_class],
        "original_predicted_probability": float(prob[target_i].item()),
        "original_wrong_minus_true_logit": original_margin,
        "source_pixel_l2": float(np.sqrt(np.square(source.numpy() - target.numpy()).sum())),
        "occlusion_4x4_wrong_minus_true_delta": occlusion_np.tolist(),
        "occlusion_best_cell": list(divmod(min_occ, 7)),
        "occlusion_best_delta": float(occlusion_delta.flatten()[min_occ].item()),
        "occlusion_best_prediction": int(best_occ_logits.argmax().item()),
        "feature_patch_wrong_minus_true_delta": patch_delta.numpy().tolist(),
        "feature_patch_best_cell": list(divmod(min_patch, 7)),
        "feature_patch_best_delta": float(patch_delta.flatten()[min_patch].item()),
        "feature_patch_best_prediction": int(best_patch_logits.argmax().item()),
        "feature_zero_wrong_minus_true_delta": zero_delta.numpy().tolist(),
        "gradient_abs_sum_by_4x4": grad_blocks.tolist(),
        "gradient_vs_occlusion_decrease_49_cell_pearson": float(np.corrcoef(
            grad_blocks.ravel(), (-occlusion_np).ravel())[0, 1]),
        "target_pixels": target[0, 0].numpy(),
        "source_pixels": source[0, 0].numpy(),
        "gradient_abs_pixels": grad_abs,
    }


def shift_two(xs):
    result = torch.zeros_like(xs)
    result[:, :, 2:, 2:] = xs[:, :, :-2, :-2]
    return result


def adversarial(model, xs, labels, epsilon):
    output = []
    for start in range(0, len(xs), 128):
        x = xs[start:start + 128].clone().requires_grad_(True)
        y = labels[start:start + 128]
        loss = F.cross_entropy(model(x), y)
        g = torch.autograd.grad(loss, x)[0]
        output.append((x.detach() + epsilon * g.sign()).clamp(0, 1))
    return torch.cat(output)


def draw_case(out, c):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(c["target_pixels"], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"错分的验证图：真{c['true_class_name']}，报{c['predicted_class_name']}")
    axes[0, 1].imshow(c["source_pixels"], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"另一张判对的{c['true_class_name']}：仅作替换源")
    gradient_image = axes[0, 2].imshow(c["gradient_abs_pixels"], cmap="magma")
    axes[0, 2].set_title("错类-真类分数的单点梯度绝对值")
    fig.colorbar(gradient_image, ax=axes[0, 2], fraction=0.046, pad=0.04)
    values = [np.asarray(c[key]) for key in (
        "occlusion_4x4_wrong_minus_true_delta", "feature_patch_wrong_minus_true_delta",
        "feature_zero_wrong_minus_true_delta")]
    bound = max(float(np.max(np.abs(v))) for v in values)
    axes[1, 0].imshow(values[0], cmap="coolwarm", vmin=-bound, vmax=bound)
    axes[1, 0].set_title("输入逐块置零后，错-真分数变化")
    axes[1, 1].imshow(values[1], cmap="coolwarm", vmin=-bound, vmax=bound)
    axes[1, 1].set_title("换中层同格32通道后，错-真分数变化")
    bottom_image = axes[1, 2].imshow(values[2], cmap="coolwarm", vmin=-bound, vmax=bound)
    axes[1, 2].set_title("中层同格清零后，错-真分数变化")
    fig.colorbar(bottom_image, ax=axes[1, :].tolist(), fraction=0.023, pad=0.02,
                 label="蓝：削弱错类优势；红：增强错类优势（logit）")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(wspace=0.18, hspace=0.18, right=0.89)
    fig.savefig(out / "one_error_three_questions.png", dpi=170)
    plt.close(fig)


def draw_reliability(out, original, scaled, shifted):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, title, reports in (
        (axes[0], "原测试图", [("原分数", original["before"]), ("验证选温度", original["after"])]),
        (axes[1], "右下2像素零填充", [("原分数", shifted["before"]), ("同一温度", shifted["after"])])
    ):
        ax.plot([0, 1], [0, 1], "--", color="0.5")
        for label, report in reports:
            xs = [b["mean_confidence"] for b in report["bins"] if b["n"]]
            ys = [b["accuracy"] for b in report["bins"] if b["n"]]
            ax.plot(xs, ys, marker="o", label=f"{label}；ECE={report['ece_10_bins']:.3f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="分箱平均最高概率", ylabel="分箱实际正确率", title=title)
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability_original_and_shift.png", dpi=160)
    plt.close(fig)


def draw_adversarial(out, model, original, changed):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    images = [("原官方测试第0图", original),
              ("ε=0.03，白箱一步", changed[0.03]),
              ("ε=0.10，白箱一步", changed[0.10])]
    with torch.no_grad():
        for ax, (title, item) in zip(axes[:3], images):
            guessed = int(model(item).argmax(dim=1).item())
            ax.imshow(item[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{title}\n报：{CLASS_NAMES[guessed]}")
    difference = (changed[0.10] - original).abs()[0, 0].numpy()
    panel = axes[3].imshow(difference, cmap="magma", vmin=0, vmax=0.10)
    axes[3].set_title("ε=0.10时逐像素绝对改动")
    fig.colorbar(panel, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "white_box_fixed_first_test_image.png", dpi=170)
    plt.close(fig)


def main(out_dir: Path):
    out = out_dir.resolve()
    if not out.is_relative_to(RUNS.resolve()) or out.exists():
        raise ValueError("choose a new directory inside work/runs")
    torch.set_num_threads(4)
    model, vx, vy, tx, ty, val_ids, epoch = load()
    out.mkdir(parents=True)
    val_logits = logits_of(model, vx)
    case = choose_case(model, vx, vy, val_ids, val_logits)
    draw_case(out, case)
    best_t = min(TEMPERATURES, key=lambda t: metrics(val_logits, vy, float(t))["nll_nats"])
    test_logits = logits_of(model, tx)
    shift_logits = logits_of(model, shift_two(tx))
    original = {"before": metrics(test_logits, ty, 1.0),
                "after": metrics(test_logits, ty, float(best_t))}
    shifted = {"before": metrics(shift_logits, ty, 1.0),
               "after": metrics(shift_logits, ty, float(best_t))}
    if original["before"]["correct"] != 8779 or shifted["before"]["correct"] != 5815:
        raise RuntimeError("C13 test reconstruction differs from original report")
    draw_reliability(out, original, None, shifted)
    stress = {"base_first1024": metrics(test_logits[:1024], ty[:1024], float(best_t))}
    first_changed = {}
    for epsilon in (0.03, 0.10):
        attacked = adversarial(model, tx[:1024], ty[:1024], epsilon)
        stress[f"fgsm_epsilon_{epsilon:.2f}"] = metrics(logits_of(model, attacked), ty[:1024], float(best_t))
        first_changed[epsilon] = attacked[:1]
    draw_adversarial(out, model, tx[:1], first_changed)
    case_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in case.items()}
    report = {
        "source_model": str(CHECKPOINT.relative_to(WORK)),
        "source_model_sha256": CHECKPOINT_SHA,
        "script_sha256": sha(Path(__file__)),
        "c13_split_sha256": SPLIT_SHA,
        "selected_epoch_from_c13": epoch,
        "evaluation_identity": "C13 validation used to select original CNN; official test already opened in C13, reused here as post-publication diagnostic, not blind test",
        "case": case_serializable,
        "temperature_selection": {"candidates": 101, "range": [0.25, 4.0],
                                  "criterion": "minimum C13 validation NLL", "selected": float(best_t),
                                  "validation_before": metrics(val_logits, vy, 1.0),
                                  "validation_after": metrics(val_logits, vy, float(best_t))},
        "reused_official_test": {"original": original, "two_pixel_right_down_zero_fill": shifted},
        "white_box_stress": stress,
        "figures": ["one_error_three_questions.png", "reliability_original_and_shift.png",
                    "white_box_fixed_first_test_image.png"],
        "limits": "one selected validation error; input blanking and feature mixing may leave image/activation manifold; ECE finite bins; white-box true-label FGSM is not natural shift; no causal claim about human garment concepts, no new blind test"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case": {k: report["case"][k] for k in (
        "original_training_row", "true_class_name", "predicted_class_name",
        "original_predicted_probability", "feature_patch_best_delta")},
        "temperature": float(best_t),
        "original": {k: original["after"][k] for k in ("accuracy", "nll_nats", "ece_10_bins")},
        "shifted": {k: shifted["after"][k] for k in ("accuracy", "nll_nats", "ece_10_bins")},
        "fgsm_accuracy": [stress[f"fgsm_epsilon_{e:.2f}"]["accuracy"] for e in (0.03, 0.1)]},
        ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.out_dir)
