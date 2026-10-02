"""C38: explicit local tool trajectories from the same C18 tokenizer and real article index.

The article titles are real WikiText train-article titles. Multiplication and
copy/unknown-key questions are author-generated teaching tasks. The test JSONL
is written at preparation but must not be opened by the trainer or chooser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/wikitext2_causal/article_source_index.json"
TOKENIZER = ROOT / "data/wikitext2_causal/tokenizer.json"
PARENT = ROOT / "runs/c25_full_mix_trial/best.pt"
DEFAULT_OUT = ROOT / "data/c38_tool_trajectories_v1"
SEED = 20260925
TOOLS = "Tools: mul(a,b) computes product; title(id) reads WikiText article title.\n"
WORDS = "red blue green amber silver violet cedar maple river stone cloud rain ember coral lemon olive raven swan cedarwood meadow".split()
DIRECT_TEMPLATES = (
    "Repeat the word {word}.",
    "Copy the word {word}.",
    "Say exactly {word}.",
    "Return {word}.",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def segments(row: dict) -> tuple[str, str, str, str]:
    prefix = TOOLS + "Q: " + row["question"] + "\nA: "
    first = row["action"] + "\n"
    observation = "" if row["observation"] is None else "T: " + row["observation"] + "\nA: "
    answer = "" if row["observation"] is None else row["answer"] + "\n"
    return prefix, first, observation, answer


def tokenized(tok: Tokenizer, row: dict) -> dict:
    prefix, first, observation, answer = segments(row)
    chunks = [encode(tok, text) for text in (prefix, first, observation, answer)]
    bos = tok.token_to_id("<|bos|>")
    pad = tok.token_to_id("<|pad|>")
    if (pad, bos, tok.token_to_id("<|eos|>")) != (0, 1, 2):
        raise RuntimeError("C18 special-token IDs changed")
    seq = [bos] + sum(chunks, [])
    if len(seq) > 129:
        raise ValueError(f"trace exceeds 128 model inputs: {row['kind']} {row['question']} {len(seq)}")
    target_source = [0] * (1 + len(chunks[0])) + [1] * len(chunks[1]) + [0] * len(chunks[2]) + [1] * len(chunks[3])
    final_source = [0] * (1 + len(chunks[0]) + len(chunks[1]) + len(chunks[2])) + [1] * len(chunks[3])
    assert len(target_source) == len(seq)
    assert len(final_source) == len(seq)
    return {
        "input_ids": seq[:-1] + [pad] * (128 - len(seq) + 1),
        "labels": [token if scored else -100 for token, scored in zip(seq[1:], target_source[1:])] + [-100] * (128 - len(seq) + 1),
        "attention_mask": [1] * (len(seq) - 1) + [0] * (128 - len(seq) + 1),
        "assistant_final_target_mask": final_source[1:] + [0] * (128 - len(seq) + 1),
        "prefix_token_count": len(chunks[0]) + 1,
        "assistant_first_targets": len(chunks[1]),
        "observation_visible_unscored_tokens": len(chunks[2]),
        "assistant_final_targets": len(chunks[3]),
    }


def item(kind: str, question: str, action: str, observation: str | None, answer: str, **identity) -> dict:
    return {"kind": kind, "question": question, "action": action,
            "observation": observation, "answer": answer, **identity}


def split_rows(rows: list[dict], train: int, validation: int, test: int, rng: random.Random):
    rng.shuffle(rows)
    if len(rows) < train + validation + test:
        raise RuntimeError("not enough distinct examples")
    return rows[:train], rows[train:train + validation], rows[train + validation:train + validation + test]


def prepare(out: Path) -> None:
    if out.exists():
        raise RuntimeError("output directory exists; use a new path rather than overwrite the split")
    tok = Tokenizer.from_file(str(TOKENIZER))
    rng = random.Random(SEED)
    original = json.loads(INDEX.read_text(encoding="utf-8"))["train"][:600]
    titles = {int(entry["index"]): entry["title"] for entry in original
              if re.fullmatch(r"[A-Za-z ]{4,25}", entry["title"])
              and len(entry["title"].split()) <= 3}
    if len(titles) != 270:
        raise RuntimeError(f"title filter drifted: {len(titles)}")

    multiplication = [item("multiply", f"Multiply {a} by {b}.", f"call mul({a},{b})", str(a * b), str(a * b), a=a, b=b)
                      for a in range(10, 100) for b in range(10, 100)]
    title_rows = [item("title", f"What is the title of WikiText article {index}?", f"call title({index})", title, title, article_id=index)
                  for index, title in titles.items()]
    direct = [item("direct", template.format(word=word), word, None, word, word=word, template_id=i)
              for word in WORDS for i, template in enumerate(DIRECT_TEMPLATES)]
    missing = [item("missing", f"What is the title of WikiText article {index}?", f"call title({index})", "NOT_FOUND", "not found", article_id=index)
               for index in range(600, 720)]

    splits = {name: [] for name in ("train", "validation", "test")}
    for rows, counts in ((multiplication, (600, 100, 100)),
                         (title_rows, (190, 40, 40)),
                         (direct, (56, 12, 12)),
                         (missing, (70, 15, 15))):
        for name, subset in zip(splits, split_rows(rows, *counts, rng)):
            splits[name].extend(subset)
    for name in splits:
        rng.shuffle(splits[name])
    assert all(len({(row["kind"], row["question"]) for row in part}) == len(part) for part in splits.values())
    assert len(set.intersection(*(set((row["kind"], row["question"]) for row in part) for part in splits.values()))) == 0

    stats = {}
    for name, rows in splits.items():
        lengths = [tokenized(tok, row) for row in rows]
        stats[name] = {
            "rows": len(rows), "kinds": {kind: sum(row["kind"] == kind for row in rows) for kind in ("multiply", "title", "direct", "missing")},
            "max_input_positions": max(sum(data["attention_mask"]) for data in lengths),
            "assistant_first_targets": sum(data["assistant_first_targets"] for data in lengths),
            "observation_visible_unscored_tokens": sum(data["observation_visible_unscored_tokens"] for data in lengths),
            "assistant_final_targets": sum(data["assistant_final_targets"] for data in lengths),
        }

    out.mkdir(parents=True)
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "title_catalog.json").write_text(json.dumps(titles, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "origin": "C18 WikiText-2 raw-v1 TRAIN article titles only; arithmetic/copy/missing rows author constructed",
        "parent_c25_full_mix_checkpoint_sha256": sha(PARENT),
        "article_index_sha256": sha(INDEX), "tokenizer_sha256": sha(TOKENIZER),
        "seed": SEED, "context_positions": 128,
        "title_selection": "first 600 C18 train articles; ASCII letters/spaces only, title length 4..25, at most three words; 270 entries",
        "split_identity": "disjoint exact multiplication pairs, title IDs, missing IDs and direct (word, template) combinations; factors/words/templates may recur across groups",
        "model_input": "BOS + tokenized prefix + tokenized assistant first line + tokenized tool observation and Assistant marker + tokenized assistant final line; each segment encoded separately to protect source boundaries",
        "loss_target": "next-token labels at assistant first and final segment IDs only; prompt and tool return visible but label=-100; no train-time test JSONL load",
        "tool_definition": TOOLS.strip(), "stats": stats,
        "files": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "stats": stats, "catalog": len(titles)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    prepare(args.out_dir)
