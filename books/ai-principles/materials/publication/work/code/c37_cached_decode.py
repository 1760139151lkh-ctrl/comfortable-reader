"""Exact-window KV cache for this book's own C19 post-LN absolute-position LM."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from train_causal_wikitext import BOS, EOS, PAD, CausalLM, TrainConfig
from transformer_bridge import positional_encoding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wikitext2_causal"
SERVING = ROOT / "data" / "c37_serving" / "c19_model_only.pt"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_serving(device: torch.device):
    state = torch.load(SERVING, map_location="cpu", weights_only=True)
    cfg = TrainConfig()
    if state["config"] != asdict(cfg) or state["c18_manifest_sha256"] != sha(DATA / "manifest.json") or state["tokenizer_sha256"] != sha(DATA / "tokenizer.json"):
        raise RuntimeError("serving model and local C18 tokenizer/data identity disagree")
    model = CausalLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    return model, tokenizer, state


def _split(read, tensor):
    return read._split(tensor)


@torch.inference_mode()
def prefill(model: CausalLM, ids: torch.Tensor):
    """Run all prompt positions, retaining each layer's pre-attention K and V."""
    cfg = model.cfg
    if ids.ndim != 2 or not (1 <= ids.shape[1] <= cfg.context) or bool((ids == PAD).any()):
        raise ValueError("prefill requires [batch,1..context] real token IDs, no PAD")
    batch, length = ids.shape
    x = model.embed(ids) * math.sqrt(cfg.width)
    x = x + positional_encoding(length, cfg.width, ids.device)[None]
    mask = torch.ones((length, length), device=ids.device, dtype=torch.bool).tril()[None, None]
    cache = []
    for block in model.blocks:
        read = block.read
        q = _split(read, read.q(x))
        k = _split(read, read.k(x))
        v = _split(read, read.v(x))
        score = (q @ k.transpose(-2, -1)) / math.sqrt(read.head_width)
        weights = score.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        values = weights @ v
        combined = values.transpose(1, 2).contiguous().reshape(batch, length, cfg.width)
        attended = read.out(combined)
        x = block.norm_read(x + attended)
        x = block.norm_ffn(x + block.ffn(x))
        cache.append((k, v))
    logits = model.output(x[:, -1]).float()
    return logits, cache


@torch.inference_mode()
def decode_one(model: CausalLM, ids: torch.Tensor, cache):
    """Append one selected token while no position has been evicted."""
    cfg = model.cfg
    if ids.ndim != 1:
        raise ValueError("ids must be [batch] for one new position")
    if len(cache) != len(model.blocks) or any(pair[0].shape[0] != len(ids) for pair in cache):
        raise ValueError("cache and batch mismatch")
    position = cache[0][0].shape[2]
    if not (1 <= position < cfg.context):
        raise ValueError("cache full; rebuild on last 128 tokens instead of evicting K/V")
    x = model.embed(ids[:, None]) * math.sqrt(cfg.width)
    x = x + positional_encoding(position + 1, cfg.width, ids.device)[None, position:position + 1]
    new_cache = []
    for block, (old_k, old_v) in zip(model.blocks, cache):
        read = block.read
        q = _split(read, read.q(x))
        k_new = _split(read, read.k(x))
        v_new = _split(read, read.v(x))
        k = torch.cat((old_k, k_new), dim=2)
        v = torch.cat((old_v, v_new), dim=2)
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(read.head_width)).softmax(dim=-1)
        values = weights @ v
        combined = values.transpose(1, 2).contiguous().reshape(len(ids), 1, cfg.width)
        x = block.norm_read(x + read.out(combined))
        x = block.norm_ffn(x + block.ffn(x))
        new_cache.append((k, v))
    return model.output(x[:, -1]).float(), new_cache


@torch.inference_mode()
def full_logits(model: CausalLM, ids: torch.Tensor):
    if ids.ndim != 2 or bool((ids == PAD).any()):
        raise ValueError("full_logits requires a PAD-free batch")
    window = ids[:, -model.cfg.context:]
    return model(window, torch.ones_like(window, dtype=torch.bool))[:, -1].float()


def select_token(logits, method="greedy", generator=None, temperature=0.8):
    scores = logits.clone()
    scores[:, [PAD, BOS]] = float("-inf")
    if method == "greedy":
        return scores.argmax(dim=-1)
    if method == "sample_top40":
        if not 0.1 <= temperature <= 5.0:
            raise ValueError("temperature must be within 0.1..5.0")
        top_value, top_id = scores.topk(40, dim=-1)
        choices = torch.multinomial(F.softmax(top_value / temperature, dim=-1), 1, generator=generator)
        return top_id.gather(1, choices).squeeze(1)
    raise ValueError(method)


@torch.inference_mode()
def generate(model: CausalLM, tokens: list[int], max_new: int, engine: str, method="greedy", seed=20260925, temperature=0.8):
    """One request; cached engine rebuilds if C19's sliding 128 window rebases."""
    if not tokens or max_new < 1:
        raise ValueError("a nonempty prompt and positive max_new are needed")
    device = next(model.parameters()).device
    produced = list(tokens)
    token_ids = []
    generator = torch.Generator(device=device).manual_seed(seed)
    window = torch.tensor(produced[-model.cfg.context:], dtype=torch.long, device=device)[None]
    if engine == "cached":
        logits, cache = prefill(model, window)
    elif engine == "full":
        logits, cache = full_logits(model, window), None
    else:
        raise ValueError(engine)
    rebuilds = 0
    for step in range(max_new):
        chosen = select_token(logits, method=method, generator=generator, temperature=temperature)
        next_id = int(chosen[0])
        token_ids.append(next_id)
        produced.append(next_id)
        if next_id == EOS or step + 1 == max_new:
            break
        window = torch.tensor(produced[-model.cfg.context:], dtype=torch.long, device=device)[None]
        if engine == "cached":
            if cache[0][0].shape[2] == model.cfg.context:
                # This particular C19 model rebases absolute positions when
                # it slides its context; dropping only oldest K/V is wrong.
                logits, cache = prefill(model, window)
                rebuilds += 1
            else:
                logits, cache = decode_one(model, chosen, cache)
        else:
            logits = full_logits(model, window)
    return {"new_token_ids": token_ids, "new_token_count": len(token_ids), "ended_with_eos": bool(token_ids and token_ids[-1] == EOS), "context_rebuilds_after_window_overflow": rebuilds, "engine": engine, "selection": method}


def cache_bytes(cache):
    return sum((k.numel() * k.element_size() + v.numel() * v.element_size()) for k, v in cache)
