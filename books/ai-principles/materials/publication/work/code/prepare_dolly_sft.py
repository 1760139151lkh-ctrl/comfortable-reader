"""C25: prepare real Dolly instruction/response records for the existing C19 LM.

Uses the fixed C18 tokenizer; no tokenizer training or pretrained model download.
Prompt and answer are encoded separately to make the response-only label
boundary exact, then concatenated as one causal sequence.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "work/data/dolly15k/databricks-dolly-15k.jsonl"
TOKENIZER_FILE = ROOT / "work/data/wikitext2_causal/tokenizer.json"
SOURCE_URL = (
    "https://huggingface.co/datasets/databricks/"
    "databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
)
BOS, EOS, PAD, CONTEXT = 1, 2, 0, 128
INSPECTED_SOURCE_ROWS = (0, 1, 2)
CATEGORIES = (
    "brainstorming",
    "classification",
    "closed_qa",
    "creative_writing",
    "general_qa",
    "information_extraction",
    "open_qa",
    "summarization",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def prompt_text(row: dict) -> str:
    instruction = row["instruction"].strip()
    context = row["context"].strip()
    return (
        "Instruction:\n" + instruction + "\n"
        + ("Context:\n" + context + "\n" if context else "")
        + "Response:\n"
    )


def group_key(row: dict) -> str:
    combined = normalized(row["instruction"]) + "\n" + normalized(row["context"])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def split_for_group(group: str) -> str:
    bucket = int(group[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def encode_row(tokenizer: Tokenizer, row: dict, row_number: int) -> dict | None:
    if not row["instruction"].strip() or not row["response"].strip():
        return None
    prompt = prompt_text(row)
    response = row["response"].strip()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    response_ids = tokenizer.encode(response, add_special_tokens=False).ids
    if not response_ids or 1 + len(prompt_ids) + len(response_ids) > CONTEXT:
        return None
    stream = [BOS] + prompt_ids + response_ids + [EOS]
    inputs = np.full(CONTEXT, PAD, dtype=np.int32)
    labels = np.full(CONTEXT, -100, dtype=np.int32)
    real = np.zeros(CONTEXT, dtype=np.bool_)
    length = len(stream) - 1
    inputs[:length] = stream[:-1]
    labels[:length] = stream[1:]
    labels[: len(prompt_ids)] = -100
    real[:length] = True
    if labels[len(prompt_ids)] != response_ids[0]:
        raise AssertionError("first scored target must be first response token")
    if labels[length - 1] != EOS or int((labels != -100).sum()) != len(response_ids) + 1:
        raise AssertionError("EOS/response-only labels mismatch")
    if not bool(real[0]) or np.any(inputs[~real] != PAD):
        raise AssertionError("real input/pad mismatch")
    return {
        "input_ids": inputs,
        "labels": labels,
        "attention_mask": real,
        "source_row": row_number,
        "category_id": CATEGORIES.index(row["category"]),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=Path("work/data/dolly15k/sft_v2")
    )
    args = parser.parse_args()
    dest = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"preserve old prepared data: {dest}")
    if not RAW.is_file() or not TOKENIZER_FILE.is_file():
        raise FileNotFoundError("raw Dolly JSONL and fixed C18 tokenizer must exist")
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    if [tokenizer.token_to_id(x) for x in ("<|pad|>", "<|bos|>", "<|eos|>")] != [
        PAD, BOS, EOS
    ]:
        raise RuntimeError("C18 special ID mapping changed")
    records = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    if len(records) != 15_011:
        raise RuntimeError("Dolly source row count changed; inspect new source before use")
    inspected_triples = {
        hashlib.sha256(
            "\n".join(
                normalized(records[i][name])
                for name in ("instruction", "context", "response")
            ).encode("utf-8")
        ).hexdigest()
        for i in INSPECTED_SOURCE_ROWS
    }

    chosen: dict[str, list[dict]] = {name: [] for name in ("train", "validation", "test")}
    source_categories = collections.Counter()
    discarded = collections.Counter()
    seen_triples: set[str] = set()
    split_group_sets: dict[str, set[str]] = {name: set() for name in chosen}
    for row_number, row in enumerate(records):
        if set(row) != {"instruction", "context", "response", "category"}:
            raise ValueError(f"unexpected Dolly fields at source row {row_number}")
        if row["category"] not in CATEGORIES:
            raise ValueError(f"unexpected category at row {row_number}")
        source_categories[row["category"]] += 1
        triple = "\n".join(
            normalized(row[name]) for name in ("instruction", "context", "response")
        )
        triple_sha = hashlib.sha256(triple.encode("utf-8")).hexdigest()
        if triple_sha in inspected_triples:
            discarded["author_inspected_source_or_duplicate"] += 1
            continue
        if triple_sha in seen_triples:
            discarded["exact_normalized_triple_duplicate"] += 1
            continue
        seen_triples.add(triple_sha)
        encoded = encode_row(tokenizer, row, row_number)
        if encoded is None:
            discarded["empty_or_over_128_positions"] += 1
            continue
        group = group_key(row)
        split = split_for_group(group)
        split_group_sets[split].add(group)
        chosen[split].append(encoded)
    names = list(chosen)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if split_group_sets[names[i]] & split_group_sets[names[j]]:
                raise AssertionError("prompt/context group crosses data splits")

    split_specs = {}
    for name, items in chosen.items():
        if not items:
            raise RuntimeError(f"empty {name} after filtering")
        arrays = {
            key: np.stack([row[key] for row in items])
            for key in ("input_ids", "labels", "attention_mask")
        }
        for key in ("source_row", "category_id", "prompt_tokens", "response_tokens"):
            arrays[key] = np.asarray([row[key] for row in items], dtype=np.int32)
        npz = dest / f"{name}.npz"
        np.savez_compressed(npz, **arrays)
        category_counts = collections.Counter(
            CATEGORIES[item["category_id"]] for item in items
        )
        split_specs[name] = {
            "arrays_path": str(npz.relative_to(ROOT)).replace("\\", "/"),
            "arrays_sha256": sha256(npz),
            "records": len(items),
            "unique_prompt_context_groups": len(split_group_sets[name]),
            "categories": dict(sorted(category_counts.items())),
            "response_target_tokens_including_eos": int(
                (arrays["labels"] != -100).sum()
            ),
            "real_input_positions": int(arrays["attention_mask"].sum()),
            "prompt_visible_but_not_scored_positions": int(
                (arrays["attention_mask"] & (arrays["labels"] == -100)).sum()
            ),
            "min_source_row": int(arrays["source_row"].min()),
            "max_source_row": int(arrays["source_row"].max()),
        }
    illustrative_index = chosen["train"][0]["source_row"]
    illustrative = records[illustrative_index]
    illustrative_encoded = encode_row(tokenizer, illustrative, illustrative_index)
    manifest = {
        "scope": "Dolly 15k human-written public instruction records filtered to fit the unchanged C19 128-position byte-BPE model; no inherited instruction model weights",
        "source_url": SOURCE_URL,
        "raw_path": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "raw_sha256": sha256(RAW),
        "raw_records": len(records),
        "license_from_source_card": "CC BY-SA 3.0; source includes employee-written examples and some Wikipedia reference material",
        "tokenizer_path": str(TOKENIZER_FILE.relative_to(ROOT)).replace("\\", "/"),
        "tokenizer_sha256": sha256(TOKENIZER_FILE),
        "context_positions": CONTEXT,
        "special_ids": {"pad": PAD, "bos": BOS, "eos": EOS},
        "prompt_template": "Instruction:\\n{instruction}\\n[Context:\\n{context}\\n]Response:\\n",
        "prompt_and_response_encoded_separately": True,
        "sequence": "BOS + prompt IDs + response IDs + EOS; input=sequence[:-1], next-token targets=sequence[1:]",
        "attention_vs_loss": "all real prompt and response input IDs visible causally; only response IDs and EOS are scored, prompt/PAD targets=-100",
        "filter": "strip outer whitespace; discard empty instruction/response, exact normalized triple repeats, and examples whose full input exceeds 128 positions; no truncation of an answer or context",
        "author_inspected_source_rows_excluded": list(INSPECTED_SOURCE_ROWS),
        "normalized_prompt_group": "sha256(casefold and collapse whitespace of instruction + newline + context); same prompt/context stays in one split",
        "split_rule": "int(first 16 hex chars of prompt group SHA,16) mod 100: 0..79 train,80..89 validation,90..99 test",
        "source_categories": dict(sorted(source_categories.items())),
        "discarded": dict(discarded),
        "splits": split_specs,
        "illustrative_training_source_row": {
            "source_row": illustrative_index,
            "category": illustrative["category"],
            "instruction": illustrative["instruction"],
            "response": illustrative["response"],
            "assigned_split": split_for_group(group_key(illustrative)),
            "prompt_tokens": illustrative_encoded["prompt_tokens"] if illustrative_encoded else None,
            "response_tokens": illustrative_encoded["response_tokens"] if illustrative_encoded else None,
            "response_boundary_input_position": (
                illustrative_encoded["prompt_tokens"] if illustrative_encoded else None
            ),
        },
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("source", len(records), "discarded", dict(discarded))
    for name, spec in split_specs.items():
        print(name, spec["records"], spec["response_target_tokens_including_eos"], spec["categories"])
    print("manifest", manifest_path)


if __name__ == "__main__":
    main()
