"""随机初始化词向量 + 两词前馈语言模型，和同词表三元计数法比较。"""

from __future__ import annotations

import copy
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
DATA = WORK / "data" / "wikitext2_raw"
RESULT = WORK / "results" / "word_neural_lm.json"
CHECKPOINT = WORK / "results" / "word_neural_lm_best.pt"
FIGURE = WORK / "figures" / "word_language_models.png"
SEED = 20260924
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 512
EVAL_BATCH = 1024
EPOCHS = 5
LEARNING_RATE = 0.001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TwoWordLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.hidden = nn.Linear(2 * EMBED_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(context)
        joined = vectors.flatten(1)
        activity = torch.tanh(self.hidden(joined))
        return self.output(activity)


def target_positions(offsets: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.arange(int(start) + 2, int(end), dtype=np.int64)
         for start, end in zip(offsets[:-1], offsets[1:])]
    )


def create_examples(tokens: np.ndarray, offsets: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = target_positions(offsets)
    context = np.column_stack((tokens[positions - 2], tokens[positions - 1])).astype(np.int64)
    target = tokens[positions].astype(np.int64)
    if len(context) != len(target):
        raise AssertionError("上下文与目标行数不同")
    return torch.from_numpy(context).to(device), torch.from_numpy(target).to(device)


@torch.no_grad()
def evaluate(model: nn.Module, context: torch.Tensor, target: torch.Tensor, unknown_id: int) -> dict:
    model.eval()
    total_loss, total_correct = 0.0, 0
    unknown_loss, known_loss = 0.0, 0.0
    unknown_count = 0
    for start in range(0, len(target), EVAL_BATCH):
        x = context[start:start + EVAL_BATCH]
        y = target[start:start + EVAL_BATCH]
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        total_loss += float(losses.sum().item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        mask = y == unknown_id
        unknown_loss += float(losses[mask].sum().item())
        known_loss += float(losses[~mask].sum().item())
        unknown_count += int(mask.sum().item())
    average = total_loss / len(target)
    return {
        "targets": len(target),
        "mean_nll_nats": average,
        "perplexity_on_mapped_vocabulary": math.exp(average),
        "top1_accuracy_on_mapped_vocabulary": total_correct / len(target),
        "unknown_target_fraction": unknown_count / len(target),
        "unknown_target_mean_nll": unknown_loss / unknown_count if unknown_count else None,
        "known_target_mean_nll": known_loss / (len(target) - unknown_count),
    }


def draw(history: list[dict], ngram: dict, test: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), layout="constrained")
    epochs = [entry["epoch"] for entry in history]
    axes[0].plot(epochs, [entry["validation"]["mean_nll_nats"] for entry in history], marker="o", label="神经模型，验证集")
    axes[0].axhline(ngram["validation_by_discount"][str(ngram["selected_discount"])]["mean_nll_nats"],
                    color="#e15759", linestyle="--", label="三元计数，验证集")
    axes[0].set(xlabel="训练轮次", ylabel="平均负对数概率（nat）", xticks=epochs,
                title="同一份映射后的词目标")
    axes[0].legend(fontsize=8)
    names = ["整体词频", "三元计数", "神经模型"]
    values = [ngram["unigram_test"]["mean_nll_nats"],
              ngram["test_selected_discount"]["mean_nll_nats"], test["mean_nll_nats"]]
    axes[1].bar(names, values, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[1].set(ylabel="平均负对数概率（nat）", title="留出测试文章")
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if sha256(DATA / "word_documents.npz") != manifest["arrays_sha256"]:
        raise ValueError("文章词序列身份不符")
    if sha256(DATA / "word_vocab.json") != manifest["vocab_sha256"]:
        raise ValueError("词表身份不符")
    vocab = json.loads((DATA / "word_vocab.json").read_text(encoding="utf-8"))
    with np.load(DATA / "word_documents.npz") as ready:
        arrays = {key: ready[key].copy() for key in ready.files}
    sets = {
        split: create_examples(arrays[split + "_tokens"], arrays[split + "_offsets"], device)
        for split in ("train", "validation", "test")
    }
    for split, (context, target) in sets.items():
        if len(target) != manifest["split_stats"][split]["next_token_examples_including_eos"]:
            raise ValueError(f"{split} 目标数量不符")
        if target.min() < 0 or target.max() >= len(vocab):
            raise ValueError(f"{split} 目标越出训练词表")
    ngram = json.loads((WORK / "results" / "word_ngram.json").read_text(encoding="utf-8"))
    if ngram["vocab_sha256"] != manifest["vocab_sha256"]:
        raise ValueError("计数对照用的不是同一词表")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    model = TwoWordLM(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(SEED + 10)
    history = []
    best_loss, best_epoch, best_model, best_optimizer = float("inf"), None, None, None
    first_batch_trace = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_context, train_target = sets["train"]
    validation_context, validation_target = sets["validation"]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_target))
        loss_sum, correct = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            ids = torch.as_tensor(order[start:start + BATCH_SIZE], device=device, dtype=torch.long)
            x = train_context.index_select(0, ids)
            y = train_target.index_select(0, ids)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if first_batch_trace is None:
                word_id = int(x[0, 0].item())
                unused = next(index for index in range(len(vocab))
                              if not torch.any(x == index).item())
                first_batch_trace = {
                    "source_context_ids": x[0].detach().cpu().tolist(),
                    "source_context_tokens": [vocab[int(i)] for i in x[0].detach().cpu().tolist()],
                    "source_target_id": int(y[0].item()),
                    "source_target_token": vocab[int(y[0].item())],
                    "first_context_embedding_grad_norm": float(model.embedding.weight.grad[word_id].norm().item()),
                    "absent_from_batch_embedding_grad_norm": float(model.embedding.weight.grad[unused].norm().item()),
                    "absent_from_batch_token": vocab[unused],
                    "first_batch_loss": float(loss.detach().item()),
                }
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(ids)
            correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        validation = evaluate(model, validation_context, validation_target, vocab.index("<unk>"))
        entry = {
            "epoch": epoch,
            "online_train_loss": loss_sum / len(train_target),
            "online_train_accuracy": correct / len(train_target),
            "validation": validation,
        }
        history.append(entry)
        print(f"epoch {epoch}: train online NLL {entry['online_train_loss']:.4f}, "
              f"validation NLL {validation['mean_nll_nats']:.4f}, "
              f"PPL {validation['perplexity_on_mapped_vocabulary']:.2f}", flush=True)
        if validation["mean_nll_nats"] < best_loss:
            best_loss, best_epoch = validation["mean_nll_nats"], epoch
            best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_optimizer = copy.deepcopy(optimizer.state_dict())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    if best_model is None or best_optimizer is None:
        raise AssertionError("没有验证选中的模型")
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": best_model,
            "optimizer_state_dict": best_optimizer,
            "chosen_epoch": best_epoch,
            "seed": SEED,
            "vocab_sha256": manifest["vocab_sha256"],
            "document_arrays_sha256": manifest["arrays_sha256"],
            "architecture": {"context": 2, "embedding": EMBED_DIM, "hidden": HIDDEN_DIM, "vocab": len(vocab)},
        },
        CHECKPOINT,
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    selected = TwoWordLM(len(vocab)).to(device)
    selected.load_state_dict(checkpoint["state_dict"])
    selected.eval()
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        if not torch.equal(model(validation_context[:1]), selected(validation_context[:1])):
            raise AssertionError("保存恢复后的验证输入 logits 不同")

    # 每次运行在验证选轮后才评价 test；本轮修正计数公式前曾打开过同一测试集。
    test_context, test_target = sets["test"]
    test = evaluate(selected, test_context, test_target, vocab.index("<unk>"))
    context_ids = torch.tensor([[vocab.index("the"), vocab.index("game")]], device=device)
    with torch.no_grad():
        probs = F.softmax(selected(context_ids), dim=-1)[0]
        values, indices = torch.topk(probs, 12)
    top_next = [{"token": vocab[int(i)], "probability": float(p)}
                for p, i in zip(values.cpu(), indices.cpu())]
    record = {
        "data_manifest": str((DATA / "manifest.json").relative_to(WORK)).replace("\\", "/"),
        "data_manifest_sha256": sha256(DATA / "manifest.json"),
        "vocab_sha256": manifest["vocab_sha256"],
        "same_mapped_target_definition": ngram["same_mapped_target_definition"],
        "vocab_size": len(vocab),
        "training_targets": len(train_target),
        "validation_targets": len(validation_target),
        "test_targets": len(test_target),
        "architecture": {"context_words": 2, "embedding_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                         "nonlinearity": "tanh", "output": "full-vocabulary logits/softmax",
                         "trainable_parameters": sum(parameter.numel() for parameter in model.parameters())},
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "selection": "lowest validation NLL; earlier epoch on tie",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "history": history,
        "chosen_epoch": best_epoch,
        "chosen_validation_nll": best_loss,
        "test": test,
        "first_batch_gradient_trace": first_batch_trace,
        "the_game_top_next_words": top_next,
        "checkpoint": str(CHECKPOINT.relative_to(WORK)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "training_seconds_excluding_data_loading": training_seconds,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "note": "one randomly initialized feedforward word LM, not 2003 paper replication or future random Transformer; unknown raw word targets map to UNK",
    }
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw(history, ngram, test)
    print(f"chosen epoch {best_epoch}; test NLL {test['mean_nll_nats']:.4f}, "
          f"PPL {test['perplexity_on_mapped_vocabulary']:.2f}; saved {RESULT}", flush=True)


if __name__ == "__main__":
    main()
