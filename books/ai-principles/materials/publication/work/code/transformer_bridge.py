"""C17: 从真实双语句对核对 Transformer 的读取、遮罩与一次短训练。

运行: python work/code/transformer_bridge.py probe
      python work/code/transformer_bridge.py train
程序使用 C15/C16 已固定的本地数据，不下载模型或语料。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORK = Path(__file__).resolve().parents[1]
PAIR_DIR = WORK / "data" / "tatoeba_fra_eng"
TEXT_DIR = WORK / "data" / "wikitext2_raw"
PROBE_FILE = WORK / "results" / "transformer_bridge_probe.json"
TRAIN_FILE = WORK / "results" / "transformer_bridge_train.json"
CHECKPOINT = WORK / "results" / "transformer_bridge_best.pt"
PAD, UNK, BOS, EOS = 0, 1, 2, 3
SEED = 20260924


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positional_encoding(length: int, width: int, device: torch.device) -> torch.Tensor:
    """原文的正弦/余弦绝对位置，偶数宽度；不是可学习参数。"""
    if width % 2:
        raise ValueError("position width must be even")
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, width, 2, device=device, dtype=torch.float32) / width
    )
    angles = positions * frequencies[None, :]
    result = torch.empty(length, width, device=device)
    result[:, 0::2] = torch.sin(angles)
    result[:, 1::2] = torch.cos(angles)
    return result


class MultiHeadRead(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        return x.reshape(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, allowed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query: [batch, target positions, width]; memory: [batch, readable positions, width]
        q, k, v = self._split(self.q(query)), self._split(self.k(memory)), self._split(self.v(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        if allowed.dtype != torch.bool or allowed.shape != (len(query), 1, len(query[0]), len(memory[0])):
            raise ValueError("allowed must be boolean [batch,1,query,key]")
        if not bool(allowed.any(dim=-1).all()):
            raise ValueError("every query row must retain at least one readable key")
        weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        values = weights @ v
        combined = values.transpose(1, 2).contiguous().reshape(len(query), len(query[0]), -1)
        return self.out(combined), weights


class EncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, inner: int) -> None:
        super().__init__()
        self.read = MultiHeadRead(width, heads)
        self.norm_read = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, inner), nn.ReLU(), nn.Linear(inner, width))
        self.norm_ffn = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        read, weights = self.read(x, x, allowed)
        x = self.norm_read(x + read)  # 原版为子层之后 Add & Norm (post-LN)
        return self.norm_ffn(x + self.ffn(x)), weights


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, inner: int) -> None:
        super().__init__()
        self.self_read = MultiHeadRead(width, heads)
        self.cross_read = MultiHeadRead(width, heads)
        self.norm_self = nn.LayerNorm(width)
        self.norm_cross = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, inner), nn.ReLU(), nn.Linear(inner, width))
        self.norm_ffn = nn.LayerNorm(width)

    def forward(
        self, x: torch.Tensor, source: torch.Tensor, causal: torch.Tensor, source_allowed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        read, self_weights = self.self_read(x, x, causal)
        x = self.norm_self(x + read)
        read, cross_weights = self.cross_read(x, source, source_allowed)
        x = self.norm_cross(x + read)
        return self.norm_ffn(x + self.ffn(x)), self_weights, cross_weights


def source_mask(source: torch.Tensor, queries: int) -> torch.Tensor:
    return (source != PAD)[:, None, None, :].expand(-1, 1, queries, -1)


def causal_mask(target_input: torch.Tensor) -> torch.Tensor:
    length = target_input.shape[1]
    triangle = torch.ones(length, length, device=target_input.device, dtype=torch.bool).tril()
    return triangle[None, None, :, :] & (target_input != PAD)[:, None, None, :]


class TranslationTransformer(nn.Module):
    """2017 原版的三类读取与 post-LN 顺序，缩小宽度和层数供本机教学。"""

    def __init__(self, source_vocab: int, target_vocab: int, width=96, heads=4, inner=192, layers=2):
        super().__init__()
        self.width = width
        self.source_embed = nn.Embedding(source_vocab, width, padding_idx=PAD)
        self.target_embed = nn.Embedding(target_vocab, width, padding_idx=PAD)
        self.encoder = nn.ModuleList([EncoderBlock(width, heads, inner) for _ in range(layers)])
        self.decoder = nn.ModuleList([DecoderBlock(width, heads, inner) for _ in range(layers)])
        self.output = nn.Linear(width, target_vocab)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor, capture=False):
        src = self.source_embed(source) * math.sqrt(self.width)
        src = src + positional_encoding(source.shape[1], self.width, source.device)[None, :, :]
        encoder_allowed = source_mask(source, source.shape[1])
        encoder_weights = None
        for block in self.encoder:
            src, encoder_weights = block(src, encoder_allowed)

        tgt = self.target_embed(target_input) * math.sqrt(self.width)
        tgt = tgt + positional_encoding(target_input.shape[1], self.width, target_input.device)[None, :, :]
        allowed_self = causal_mask(target_input)
        allowed_cross = source_mask(source, target_input.shape[1])
        self_weights = cross_weights = None
        for block in self.decoder:
            tgt, self_weights, cross_weights = block(tgt, src, allowed_self, allowed_cross)
        diagnostics = (encoder_weights, self_weights, cross_weights) if capture else None
        return self.output(tgt), diagnostics


class CausalTextTransformer(nn.Module):
    """只有遮住未来的自身读取；没有源句编码器和跨句读取。"""

    def __init__(self, vocab: int, width=64, heads=4, inner=128, layers=2):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([EncoderBlock(width, heads, inner) for _ in range(layers)])
        self.output = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens) * math.sqrt(self.width)
        x = x + positional_encoding(tokens.shape[1], self.width, tokens.device)[None, :, :]
        # WikiText C15 词表没有 PAD；把合法词和 UNK=0 都纳入可读前缀。
        length = tokens.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()[None, None].expand(len(tokens), -1, -1, -1)
        for block in self.blocks:
            x, _ = block(x, allowed)
        return self.output(x)


def load_pair_split(manifest: dict, split: str) -> dict[str, np.ndarray]:
    spec = manifest["files"][split]
    path = WORK / spec["arrays"]
    if sha256(path) != spec["arrays_sha256"]:
        raise ValueError(f"C16 {split} arrays changed")
    with np.load(path) as item:
        return {key: item[key].copy() for key in item.files}


def pair_data() -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    manifest = json.loads((PAIR_DIR / "manifest.json").read_text(encoding="utf-8"))
    if sha256(PAIR_DIR / "fra-eng.zip") != manifest["source_identity"]["zip_sha256"]:
        raise ValueError("C16 source ZIP changed")
    for name, field in (("english_vocab.json", "source_vocab_sha256"), ("french_vocab.json", "target_vocab_sha256")):
        if sha256(PAIR_DIR / name) != manifest[field]:
            raise ValueError(f"C16 {name} changed")
    return manifest, {split: load_pair_split(manifest, split) for split in ("train", "validation")}


def batch_tensors(part: dict, indices: np.ndarray, device: torch.device):
    source_length = int(part["source_len"][indices].max())
    target_length = int(part["target_len"][indices].max())
    source = torch.as_tensor(part["source_ids"][indices, :source_length], device=device, dtype=torch.long)
    target_input = torch.as_tensor(part["target_input"][indices, :target_length], device=device, dtype=torch.long)
    target = torch.as_tensor(part["target_output"][indices, :target_length], device=device, dtype=torch.long)
    return source, target_input, target


def summed_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    count = int((target != PAD).sum().item())
    return F.cross_entropy(logits.transpose(1, 2), target, ignore_index=PAD, reduction="sum"), count


def probe() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, parts = pair_data()
    model = TranslationTransformer(manifest["source_vocab_size"], manifest["target_vocab_size"]).to(device).eval()
    trace_index = None
    with (PAIR_DIR / "validation_attribution.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if json.loads(line)["source_row_zero_based"] == 26766:
                trace_index = index
                break
    if trace_index is None:
        raise AssertionError("C16 red-pants trace row missing")
    row = np.array([0, trace_index], dtype=np.int64)
    source, target_input, target = batch_tensors(parts["validation"], row, device)
    with torch.no_grad():
        logits, (encoder_w, self_w, cross_w) = model(source, target_input, capture=True)
        source_edit = source.clone()
        source_edit[0, 0] = (source_edit[0, 0] + 1) % manifest["source_vocab_size"]
        changed_source, _ = model(source_edit, target_input)
        target_edit = target_input.clone()
        last_real = int(parts["validation"]["target_len"][0]) - 1
        target_edit[0, last_real] = (target_edit[0, last_real] + 1) % manifest["target_vocab_size"]
        changed_future, _ = model(source, target_edit)
        earlier_difference = float((logits[0, :last_real] - changed_future[0, :last_real]).abs().max())
        source_difference = float((logits[0] - changed_source[0]).abs().max())
        self_forbidden = float(self_w.masked_select(~causal_mask(target_input)).abs().max())
        cross_forbidden = float(cross_w.masked_select(~source_mask(source, target_input.shape[1])).abs().max()) if bool((source == PAD).any()) else 0.0
        row_error = float((self_w.sum(dim=-1) - 1).abs().max())

        # 无位置信号时，整句同步调换顺序只会把输出同步调换；加入位置后一般不再如此。
        plain = model.encoder[0].read
        positions = torch.randn(1, 5, model.width, device=device)
        permutation = torch.tensor([2, 4, 0, 1, 3], device=device)
        full = torch.ones(1, 1, 5, 5, dtype=torch.bool, device=device)
        plain_output, _ = plain(positions, positions, full)
        permuted_output, _ = plain(positions[:, permutation], positions[:, permutation], full)
        no_position_error = float((permuted_output - plain_output[:, permutation]).abs().max())
        with_position = positions + positional_encoding(5, model.width, device)[None]
        with_position_permuted = positions[:, permutation] + positional_encoding(5, model.width, device)[None]
        positioned_output, _ = plain(with_position, with_position, full)
        positioned_permuted_output, _ = plain(with_position_permuted, with_position_permuted, full)
        position_breaks_equivariance = float((positioned_permuted_output - positioned_output[:, permutation]).abs().max())

    # C15 已固定的真实文章的一段，验证 decoder-only 采用同样的“当前前缀 -> 下一词”。
    text_manifest = json.loads((TEXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if sha256(TEXT_DIR / "word_documents.npz") != text_manifest["arrays_sha256"]:
        raise ValueError("C15 word arrays changed")
    with np.load(TEXT_DIR / "word_documents.npz") as documents:
        article = documents["train_tokens"][:19].astype(np.int64)
    text_input = torch.as_tensor(article[:-1], device=device)[None, :]
    text_target = torch.as_tensor(article[1:], device=device)[None, :]
    language_model = CausalTextTransformer(text_manifest["vocab_size"]).to(device)
    optimizer = torch.optim.Adam(language_model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    text_logits = language_model(text_input)
    loss_before = F.cross_entropy(text_logits.transpose(1, 2), text_target)
    loss_before.backward()
    gradient_norm = float(language_model.blocks[0].read.q.weight.grad.norm().item())
    q_before = language_model.blocks[0].read.q.weight.detach().clone()
    optimizer.step()
    parameter_delta = float((language_model.blocks[0].read.q.weight.detach() - q_before).abs().max())
    language_model.eval()
    with torch.no_grad():
        text_logits = language_model(text_input)
        later = text_input.clone()
        later[0, 12] = (later[0, 12] + 1) % text_manifest["vocab_size"]
        later_logits = language_model(later)
        text_future_difference = float((text_logits[0, :12] - later_logits[0, :12]).abs().max())

    q = torch.randn(8192, 64, device=device)
    k = torch.randn(8192, 64, device=device)
    dot = (q * k).sum(dim=-1)
    report = {
        "task": "architecture and visibility probe; no claim of translation quality",
        "device": str(device), "torch": torch.__version__,
        "source_manifest_sha256": sha256(PAIR_DIR / "manifest.json"),
        "text_manifest_sha256": sha256(TEXT_DIR / "manifest.json"),
        "translation_shapes": {"source": list(source.shape), "target_input": list(target_input.shape), "logits": list(logits.shape), "encoder_weights": list(encoder_w.shape), "decoder_self_weights": list(self_w.shape), "cross_weights": list(cross_w.shape)},
        "decoder_future_change_earlier_logits_max_abs": earlier_difference,
        "source_change_logits_max_abs": source_difference,
        "decoder_masked_weight_max_abs": self_forbidden,
        "cross_padding_weight_max_abs": cross_forbidden,
        "decoder_weight_row_sum_max_error": row_error,
        "no_position_permutation_equivariance_max_error": no_position_error,
        "position_signal_breaks_permutation_equivariance_max_abs": position_breaks_equivariance,
        "dot_score_variance_unscaled": float(dot.var().item()),
        "dot_score_variance_scaled": float((dot / math.sqrt(64)).var().item()),
        "real_text_next_word_loss_before_step": float(loss_before.item()),
        "real_text_query_projection_gradient_norm": gradient_norm,
        "real_text_query_projection_parameter_max_change_after_step": parameter_delta,
        "real_text_future_change_earlier_logits_max_abs": text_future_difference,
        "real_text_first_19_ids": article.tolist(),
    }
    if earlier_difference > 1e-5 or text_future_difference > 1e-5 or self_forbidden != 0 or cross_forbidden != 0 or row_error > 1e-5 or no_position_error > 1e-5:
        raise AssertionError("mask or row normalization probe failed")
    if source_difference <= 0 or gradient_norm <= 0 or parameter_delta <= 0 or position_breaks_equivariance <= 0:
        raise AssertionError("source or gradient path did not reach the output")
    PROBE_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


@torch.no_grad()
def evaluate(model: TranslationTransformer, part: dict, rows: np.ndarray, device: torch.device) -> dict:
    model.eval()
    total_loss, total_targets = 0.0, 0
    for start in range(0, len(rows), 128):
        source, target_input, target = batch_tensors(part, rows[start:start + 128], device)
        logits, _ = model(source, target_input)
        batch_loss, count = summed_loss(logits, target)
        total_loss += float(batch_loss.item())
        total_targets += count
    return {"target_count_including_eos": total_targets, "mean_teacher_forced_nll_nats": total_loss / total_targets}


@torch.no_grad()
def greedy(model: TranslationTransformer, source: torch.Tensor, max_steps=20) -> list[int]:
    model.eval()
    prefix = torch.full((1, 1), BOS, device=source.device, dtype=torch.long)
    output = []
    for _ in range(max_steps):
        scores, _ = model(source, prefix)
        token = int(scores[0, -1].argmax())
        output.append(token)
        if token == EOS:
            break
        prefix = torch.cat([prefix, torch.tensor([[token]], device=source.device)], dim=1)
    return output


def train(train_limit: int, val_limit: int, epochs: int) -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    manifest, parts = pair_data()
    train_rows = rng.permutation(len(parts["train"]["source_ids"]))[:train_limit]
    val_rows = rng.permutation(len(parts["validation"]["source_ids"]))[:val_limit]
    model = TranslationTransformer(manifest["source_vocab_size"], manifest["target_vocab_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    history = []
    best, best_epoch = math.inf, 0
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        rows = rng.permutation(train_rows)
        train_loss, train_targets = 0.0, 0
        for offset in range(0, len(rows), 128):
            source, target_input, target = batch_tensors(parts["train"], rows[offset:offset + 128], device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(source, target_input)
            loss_sum, count = summed_loss(logits, target)
            (loss_sum / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss_sum.item())
            train_targets += count
        validation = evaluate(model, parts["validation"], val_rows, device)
        record = {"epoch": epoch, "train_teacher_nll": train_loss / train_targets, "train_target_count": train_targets, "validation": validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation["mean_teacher_forced_nll_nats"] < best:
            best, best_epoch = validation["mean_teacher_forced_nll_nats"], epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "seed": SEED}, CHECKPOINT)

    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    parts["test"] = load_pair_split(manifest, "test")
    test_rows = np.arange(len(parts["test"]["source_ids"]), dtype=np.int64)
    test_measure = evaluate(model, parts["test"], test_rows, device)
    trace_row = None
    with (PAIR_DIR / "validation_attribution.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            if item["source_row_zero_based"] == 26766:
                trace_row = (index, item)
                break
    if trace_row is None:
        raise AssertionError("preselected C16 validation row missing")
    index, metadata = trace_row
    source, target_input, target = batch_tensors(parts["validation"], np.array([index]), device)
    with torch.no_grad():
        teacher_logits, _ = model(source, target_input)
        reference_probability = teacher_logits.softmax(dim=-1).gather(-1, target[..., None]).squeeze(-1)
        decoded = greedy(model, source)
    target_vocab = json.loads((PAIR_DIR / "french_vocab.json").read_text(encoding="utf-8"))
    old_examples = json.loads((WORK / "results" / "tatoeba_translation_examples.json").read_text(encoding="utf-8"))
    if old_examples["manifest_sha256"] != sha256(PAIR_DIR / "manifest.json"):
        raise ValueError("C16 preselected test examples use a different sentence manifest")
    chosen_groups = {entry["source_group_sha256"]: entry for entry in old_examples["test_hash_selected_examples"]}
    chosen_rows = {}
    with (PAIR_DIR / "test_attribution.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            item = json.loads(line)
            signature = item["source_group_sha256"]
            if signature in chosen_groups and signature not in chosen_rows:
                chosen_rows[signature] = (index, item)
    if len(chosen_rows) != len(chosen_groups):
        raise ValueError("fixed C16 test sample groups are missing")
    examples = []
    for signature, old in chosen_groups.items():
        sample_index, item = chosen_rows[signature]
        sample_source, _, _ = batch_tensors(parts["test"], np.array([sample_index]), device)
        examples.append({"length_bin": old["length_bin"], "source_group_sha256": signature,
                         "english_original": item["english_original"], "french_first_reference": item["french_original"],
                         "attribution": item["attribution"],
                         "greedy_output": [target_vocab[i] for i in greedy(model, sample_source)]})
    report = {
        "task": "short real-pair architecture training, not WMT14 reproduction or main language-model pretraining",
        "command": f"python work/code/transformer_bridge.py train --train-limit {train_limit} --val-limit {val_limit} --epochs {epochs}",
        "data_manifest_sha256": sha256(PAIR_DIR / "manifest.json"),
        "train_pair_rows": len(train_rows), "validation_pair_rows": len(val_rows), "test_pair_rows": len(test_rows),
        "test_opened_only_after_all_epochs_and_validation_checkpoint_selection": True,
        "train_subset_selection": "seeded permutation of existing training split; no source group crosses train/validation",
        "seed": SEED, "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "model": {"width": 96, "heads": 4, "head_width": 24, "encoder_layers": 2, "decoder_layers": 2, "feed_forward_inner": 192, "post_layer_norm": True, "position": "sinusoidal", "separate_source_target_vocab": True, "trainable_parameters": sum(p.numel() for p in model.parameters())},
        "history": history, "selected_epoch": best_epoch, "test": test_measure,
        "fixed_test_examples": examples, "checkpoint": str(CHECKPOINT.relative_to(WORK)),
        "checkpoint_sha256": sha256(CHECKPOINT), "elapsed_seconds": time.perf_counter() - start_time,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "preselected_validation_trace": {
            "source_row_zero_based": 26766, "english_original": metadata["english_original"], "french_original": metadata["french_original"],
            "attribution": metadata["attribution"],
            "teacher_target": [target_vocab[int(i)] for i in target[0].tolist() if i != PAD],
            "teacher_target_probabilities": [float(v) for v in reference_probability[0].tolist()],
            "greedy_output": [target_vocab[i] for i in decoded],
        },
    }
    TRAIN_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_epoch": best_epoch, "best_validation_nll": best, "trace": report["preselected_validation_trace"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["probe", "train"])
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--val-limit", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.action == "probe":
        probe()
    else:
        train(args.train_limit, args.val_limit, args.epochs)


if __name__ == "__main__":
    main()
