"""C38 third condition: keep v1 held-out questions and align BPE copy spans.

Assistant-produced lines now include their leading space after the host's A:
marker. Tool observations and assistant answers thus expose the same ByteLevel
BPE IDs for the same copied title, without claiming that copy is guaranteed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

from c38_tool_data import ROOT, PARENT, TOKENIZER, TOOLS, SEED, encode, sha

V1 = ROOT / "data/c38_tool_trajectories_v1"
V3 = ROOT / "data/c38_tool_trajectories_v3"


def segments_v3(row):
    prefix = TOOLS + "Q: " + row["question"] + "\nA:"
    first = " " + row["action"] + "\n"
    observation = "" if row["observation"] is None else "T: " + row["observation"] + "\nA:"
    answer = "" if row["observation"] is None else " " + row["answer"] + "\n"
    return prefix, first, observation, answer


def tokenized_v3(tok: Tokenizer, row: dict):
    chunks = [encode(tok, text) for text in segments_v3(row)]
    if (tok.token_to_id("<|pad|>"), tok.token_to_id("<|bos|>"), tok.token_to_id("<|eos|>")) != (0, 1, 2):
        raise RuntimeError("C18 control tokens changed")
    seq = [1] + sum(chunks, [])
    if len(seq) > 129:
        raise RuntimeError(f"long trace {len(seq)}: {row['question']}")
    tag = [0] * (1 + len(chunks[0])) + [1] * len(chunks[1]) + [0] * len(chunks[2]) + [1] * len(chunks[3])
    final = [0] * (1 + len(chunks[0]) + len(chunks[1]) + len(chunks[2])) + [1] * len(chunks[3])
    assert len(seq) == len(tag) == len(final)
    pad = 128 - len(seq) + 1
    obs_start = 1 + len(chunks[0]) + len(chunks[1]) + 2 if chunks[2] else -1
    obs_length = len(chunks[2]) - 4 if chunks[2] else 0
    answer_start = 1 + len(chunks[0]) + len(chunks[1]) + len(chunks[2])
    ordinals = [-1] * 128
    for r in range(len(chunks[3])):
        ordinals[answer_start - 1 + r] = r
    if row["kind"] in ("title", "multiply") and chunks[2][2:-2] != chunks[3]:
        raise RuntimeError("copy-aligned value IDs differ for " + row["kind"])
    return {"input_ids": seq[:-1] + [0] * pad,
            "labels": [value if marked else -100 for value, marked in zip(seq[1:], tag[1:])] + [-100] * pad,
            "attention_mask": [1] * (len(seq) - 1) + [0] * pad,
            "assistant_final_target_mask": final[1:] + [0] * pad,
            "copy_observation_value_start": obs_start,
            "copy_observation_value_length": obs_length,
            "copy_final_target_ordinals": ordinals,
            "prefix_token_count": 1 + len(chunks[0]),
            "assistant_first_targets": len(chunks[1]),
            "observation_visible_unscored_tokens": len(chunks[2]),
            "assistant_final_targets": len(chunks[3])}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(out: Path):
    if out.exists():
        raise RuntimeError("output exists")
    old = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "title_catalog.json"):
        if sha(V1 / name) != old["files"][name]:
            raise RuntimeError("v1 source drift: " + name)
    tok = Tokenizer.from_file(str(TOKENIZER))
    train = read(V1 / "train.jsonl")
    title_train = [row for row in train if row["kind"] == "title"]
    direct_train = [row for row in train if row["kind"] == "direct"]
    rng = random.Random(SEED + 3)
    simulated = []
    for row in title_train:
        alternatives = [other for other in title_train if other["article_id"] != row["article_id"]]
        for variant in range(6):
            other = rng.choice(alternatives)
            scenario = dict(row)
            scenario["observation"] = other["observation"]
            scenario["answer"] = other["observation"]
            scenario["author_simulated_catalog_snapshot"] = True
            scenario["simulated_value_from_training_article_id"] = other["article_id"]
            scenario["snapshot_variant"] = variant
            simulated.append(scenario)
    repeated_direct = []
    for row in direct_train:
        for repeat in range(4):
            extra = dict(row)
            extra["author_repetition_for_direct_balance"] = repeat
            repeated_direct.append(extra)
    numeric_direct = []
    for value in rng.sample(range(100, 10000), 200):
        s = str(value)
        numeric_direct.append({"kind": "direct", "question": f"Copy exactly the code {s}.",
                               "action": s, "observation": None, "answer": s,
                               "author_numeric_copy_practice": True})
    train.extend(simulated + repeated_direct + numeric_direct)
    rng.shuffle(train)
    splits = {"train": train, "validation": read(V1 / "validation.jsonl"), "test": read(V1 / "test.jsonl")}
    stats = {}
    for name, rows in splits.items():
        encoded = [tokenized_v3(tok, row) for row in rows]
        stats[name] = {"rows": len(rows), "kinds": {kind: sum(row["kind"] == kind for row in rows) for kind in ("multiply", "title", "direct", "missing")},
                       "max_input_positions": max(sum(item["attention_mask"]) for item in encoded),
                       "assistant_first_targets": sum(item["assistant_first_targets"] for item in encoded),
                       "observation_visible_unscored_tokens": sum(item["observation_visible_unscored_tokens"] for item in encoded),
                       "assistant_final_targets": sum(item["assistant_final_targets"] for item in encoded)}
    # Check the actual observation-value token span against the answer IDs,
    # including the leading-space ByteLevel token, for every training title.
    for row in title_train:
        value = row["observation"]
        observed = encode(tok, "T: " + value + "\nA:")[2:-2]
        answering = encode(tok, " " + value + "\n")
        if observed != answering:
            raise RuntimeError("v3 title observation/answer BPE span mismatch: " + value)
    out.mkdir(parents=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        if name in ("validation", "test"):
            path.write_bytes((V1 / f"{name}.jsonl").read_bytes())
        else:
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (out / "title_catalog.json").write_bytes((V1 / "title_catalog.json").read_bytes())
    manifest = {"origin": "v1 real titles/questions and same validation/test records; v3 author-simulated alternate catalog snapshots and balanced direct copy; BPE segment alignment changed",
                "format_version": 3, "v1_manifest_sha256": sha(V1 / "manifest.json"),
                "parent_c25_full_mix_checkpoint_sha256": sha(PARENT),
                "article_index_sha256": old["article_index_sha256"], "tokenizer_sha256": sha(TOKENIZER),
                "seed": SEED + 3, "context_positions": 128,
                "augmentation": {"simulated_train_catalog_rows": len(simulated), "direct_word_repetition_rows": len(repeated_direct),
                                 "numeric_direct_rows": len(numeric_direct), "alternate_title_values_only_from_v1_train": True},
                "split_identity": "validation/test JSONL byte-identical to v1; only v3 training scenarios and token-segment representation change",
                "model_input": "BOS + tokenized [Tools+Q+A:] + tokenized [' '+assistant first line] + tokenized [T: observation + A:] + tokenized [' '+assistant final line]",
                "loss_target": "only assistant first/final token targets score; tool return visible, target masked; leading space included in assistant-produced segment so ByteLevel IDs can match observed word IDs",
                "tool_definition": old["tool_definition"], "stats": stats,
                "files": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "stats": stats, "validation_same_bytes": sha(out / "validation.jsonl") == sha(V1 / "validation.jsonl"), "test_same_bytes": sha(out / "test.jsonl") == sha(V1 / "test.jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=V3)
    args = parser.parse_args()
    prepare(args.out_dir)
