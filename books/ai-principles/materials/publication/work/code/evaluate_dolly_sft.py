"""C25: open the held-out Dolly response set once after validation selection.

All three adaptation arms were fixed before this command. This script does
not train or select a checkpoint from the test outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from prepare_dolly_sft import TOKENIZER_FILE
from train_dolly_sft import (
    ROOT,
    attach_lora,
    choose_device,
    evaluate_sft,
    fixed_validation_prompts,
    generate_fixed,
    load_base,
    load_sft_manifest,
    load_sft_split,
    write_json,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(run_dir: Path, expected_mode: str, data_sha: str, base_sha: str) -> tuple[dict, dict]:
    summary_path = run_dir / "train.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_path = ROOT / summary["checkpoint_path"]
    if (
        summary["mode"] != expected_mode
        or summary["sft_manifest_sha256"] != data_sha
        or summary["base"]["c19_checkpoint_sha256"] != base_sha
        or summary["test_opened"] is not False
        or sha256(checkpoint_path) != summary["checkpoint_sha256"]
    ):
        raise RuntimeError(f"{expected_mode} run no longer matches its declared training identity")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint["mode"] != expected_mode
        or checkpoint["base_checkpoint_sha256"] != base_sha
        or checkpoint["sft_manifest_sha256"] != data_sha
        or checkpoint["selected_epoch"] != summary["best_epoch"]
    ):
        raise RuntimeError(f"{expected_mode} checkpoint metadata mismatch")
    return summary, checkpoint


def selected_model(mode: str, checkpoint: dict, device: torch.device):
    model, cfg, _ = load_base()
    if mode == "lora":
        routes = attach_lora(model)
        if routes != checkpoint["adapter_routes"]:
            raise RuntimeError("LoRA route changed")
        expected = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if expected != set(checkpoint["model"]):
            raise RuntimeError("LoRA adapter names changed")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    value = checkpoint["model"][name]
                    if parameter.shape != value.shape:
                        raise RuntimeError("LoRA adapter shape changed")
                    parameter.copy_(value)
    else:
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, default=Path("work/runs/c25_full_trial"))
    parser.add_argument("--lora", type=Path, default=Path("work/runs/c25_lora_trial"))
    parser.add_argument("--mix", type=Path, default=Path("work/runs/c25_full_mix_trial"))
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c25_dolly_sft_test.json")
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"test already opened; preserve its result: {out}")
    manifest, data_sha = load_sft_manifest()
    base_model, cfg, base_identity = load_base()
    run_dirs = {
        "full": args.full,
        "lora": args.lora,
        "full-mix": args.mix,
    }
    run_dirs = {
        mode: path if path.is_absolute() else ROOT / path
        for mode, path in run_dirs.items()
    }
    runs = {
        mode: load_run(path, mode, data_sha, base_identity["c19_checkpoint_sha256"])
        for mode, path in run_dirs.items()
    }

    # Only now open response labels of the held-out test arrays.
    test = load_sft_split(manifest, "test")
    fixed_prompts = fixed_validation_prompts(test)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    device = choose_device()
    models = {"base": (base_model.to(device), cfg)}
    for mode, (_, checkpoint) in runs.items():
        models[mode] = selected_model(mode, checkpoint, device)
    results = {}
    for mode, (model, config) in models.items():
        model.eval()
        results[mode] = {
            "test": evaluate_sft(model, test, device, with_categories=True),
            "free_generations": generate_fixed(model, tokenizer, fixed_prompts, device, config),
            "validation_selected_epoch": (
                runs[mode][0]["best_epoch"] if mode != "base" else None
            ),
            "validation_response_nll_before_test": (
                runs[mode][0]["best_validation_response_nll"]
                if mode != "base"
                else runs["full"][0]["baseline_validation"]["response_nll"]
            ),
            "old_wikitext_validation_nll_before_test": (
                runs[mode][0]["best_wiki_validation_nll"]
                if mode != "base"
                else runs["full"][0]["baseline_wiki_validation"]["nll"]
            ),
        }
        print(
            mode,
            "heldout response NLL",
            round(results[mode]["test"]["response_nll"], 6),
            "old WikiText val",
            round(results[mode]["old_wikitext_validation_nll_before_test"], 6),
            flush=True,
        )
    output = {
        "scope": "one-time held-out Dolly sft_v2 response-only test after three arms independently selected epochs by validation; free generations use four test prompts fixed by source order/category before viewing answers; no test-based model change",
        "data_manifest_sha256": data_sha,
        "test_arrays_sha256": manifest["splits"]["test"]["arrays_sha256"],
        "base_checkpoint_sha256": base_identity["c19_checkpoint_sha256"],
        "device": str(device),
        "torch_version": torch.__version__,
        "model_selection": "each arm selected its own best epoch by validation response NLL; no arm or setting chosen by this test file",
        "run_summaries": {
            mode: {
                "path": str((path / "train.json").relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path / "train.json"),
                "checkpoint_sha256": runs[mode][0]["checkpoint_sha256"],
                "trainable_parameters": runs[mode][0]["parameters"]["trainable_parameters"],
            }
            for mode, path in run_dirs.items()
        },
        "test_prompt_selection": "first test record for each distinct category in source order, stop after four categories; independent of response content",
        "test_prompt_source_rows": [x["source_row"] for x in fixed_prompts],
        "results": results,
    }
    write_json(out, output)


if __name__ == "__main__":
    main()
