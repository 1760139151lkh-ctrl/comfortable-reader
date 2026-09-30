"""C18: 在真实 WikiText 训练块上核对可见性、有效目标与一次反向传播。

运行：python work/code/causal_batch_probe.py
这不是持续预训练；参数只从随机初值算一次梯度，不执行 optimizer.step()。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformer_bridge import EncoderBlock, positional_encoding


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "wikitext2_causal"
RESULT = WORK / "results" / "causal_batch_probe.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OneLayerCausalReader(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.width = 32
        self.embed = nn.Embedding(vocab_size, self.width)
        self.block = EncoderBlock(self.width, heads=4, inner=64)
        self.output = nn.Linear(self.width, vocab_size)

    def forward(self, ids: torch.Tensor, real: torch.Tensor):
        length = ids.shape[1]
        triangle = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
        allowed = triangle[None, None, :, :] & real[:, None, None, :]
        x = self.embed(ids) * math.sqrt(self.width)
        x = x + positional_encoding(length, self.width, ids.device)[None]
        x, weights = self.block(x, allowed)
        return self.output(x), weights, allowed


def main() -> None:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = WORK / manifest["splits"]["train"]["arrays_path"]
    if sha256(array_path) != manifest["splits"]["train"]["arrays_sha256"]:
        raise ValueError("C18 training blocks changed")
    if sha256(DATA / "tokenizer.json") != manifest["tokenizer"]["model_sha256"]:
        raise ValueError("C18 tokenizer changed")
    with np.load(array_path) as arrays:
        article_indices = arrays["article_index"]
        first_article_blocks = np.flatnonzero(article_indices == 0)
        selected = [int(first_article_blocks[0]), int(first_article_blocks[-1])]
        inputs = torch.from_numpy(arrays["input_ids"][selected].astype(np.int64))
        labels = torch.from_numpy(arrays["labels"][selected].astype(np.int64))
        real = torch.from_numpy(arrays["attention_mask"][selected].copy())
        offsets = arrays["first_token_offset"][selected].tolist()
    if selected[0] == selected[1] or bool(real[1].all()):
        raise AssertionError("need one full and one padded block from the same real article")
    if not torch.equal(real, labels != -100):
        raise AssertionError("当前这份右邻预测资料应让真实输入与有效目标位置一一对应")
    if int(labels[1, int(real[1].sum()) - 1]) != manifest["tokenizer"]["special_ids"]["<|eos|>"]:
        raise AssertionError("last real target should end this article")
    torch.manual_seed(20260924)
    model = OneLayerCausalReader(manifest["tokenizer"]["vocab_size"])
    logits, weights, allowed = model(inputs, real)
    valid_count = int((labels != -100).sum())
    sum_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    mean_real = sum_loss / valid_count
    mean_if_pad_counted_in_denominator = sum_loss / labels.numel()
    mean_real.backward()
    forbidden_max = float(weights.detach().masked_select(~allowed.expand_as(weights)).abs().max())
    if forbidden_max != 0:
        raise AssertionError("attention placed mass on future or padded keys")
    if not torch.isfinite(model.block.read.q.weight.grad).all():
        raise AssertionError("a core parameter did not receive finite gradients")
    # 输入仍是真实前文，只撤去一项训练目标：可读位置数与损失分母便不再相同。
    masked_labels = labels.clone()
    masked_labels[0, 0] = -100
    with torch.no_grad():
        masked_sum = F.cross_entropy(
            logits.detach().reshape(-1, logits.shape[-1]), masked_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
    masked_target_count = int((masked_labels != -100).sum())
    if masked_target_count != valid_count - 1 or int(real.sum()) != valid_count:
        raise AssertionError("额外标签遮罩不得改变真实输入列")
    report = {
        "experiment": "one real full block and one padded final block of the first training article",
        "input_manifest_sha256": sha256(manifest_path),
        "selected_block_indices": selected,
        "article_index": 0,
        "first_token_offsets": offsets,
        "context_positions_each": int(inputs.shape[1]),
        "real_targets_each": real.sum(dim=1).tolist(),
        "pad_slots_each": (~real).sum(dim=1).tolist(),
        "first_inputs": inputs[0, :8].tolist(),
        "first_labels": labels[0, :8].tolist(),
        "last_target_id": int(labels[1, int(real[1].sum()) - 1]),
        "forbidden_attention_weight_max": forbidden_max,
        "random_initialization_mean_nll_per_real_target": float(mean_real.detach()),
        "wrong_mean_if_pad_positions_counted": float(mean_if_pad_counted_in_denominator.detach()),
        "valid_target_count": valid_count,
        "input_and_label_masks_are_separate_roles": {
            "real_readable_input_positions_after_masking_one_target": int(real.sum()),
            "loss_targets_after_masking_one_target": masked_target_count,
            "correct_mean_over_remaining_targets": float(masked_sum / masked_target_count),
            "wrong_mean_over_readable_inputs": float(masked_sum / int(real.sum())),
            "note": "首块第一项输入BOS仍可读，但其右邻目标暂不计损失；遮标签不等于删输入。",
        },
        "query_projection_gradient_norm": float(model.block.read.q.weight.grad.norm().detach()),
        "optimizer_step_executed": False,
        "torch": torch.__version__,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
