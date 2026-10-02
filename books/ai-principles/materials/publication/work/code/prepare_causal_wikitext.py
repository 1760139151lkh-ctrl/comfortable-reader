"""C18: 有来源的文章 -> 本地训练的字节级 BPE -> 不跨文章的因果训练块。

运行：python work/code/prepare_causal_wikitext.py
依赖：C15 的 pyarrow/NumPy，及 tokenizers==0.23.2。
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import prepare_wikitext2 as c15


WORK = Path(__file__).resolve().parents[1]
SOURCE = WORK / "data" / "wikitext2_raw"
DEST = WORK / "data" / "wikitext2_causal"
BLOCK = 128
VOCAB = 8192
SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>"]
EXPECTED_ARTICLES = {"train": 600, "validation": 60, "test": 60}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_documents(split: str, old_manifest: dict) -> list[dict]:
    info = old_manifest["source_files"][split]
    path = WORK / info["path"]
    if sha256(path) != info["sha256"]:
        raise ValueError(f"{split} Parquet source differs from C15's measured bytes")
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    if len(rows) != info["rows"]:
        raise ValueError(f"{split} row count changed")
    parsed = c15.parse_articles(rows)
    if len(parsed) != EXPECTED_ARTICLES[split]:
        raise ValueError(f"{split} article count changed")
    docs = []
    for index, article in enumerate(parsed):
        start = article["title_source_row"]
        end = parsed[index + 1]["title_source_row"] if index + 1 < len(parsed) else len(rows)
        source_span = "".join(rows[start:end])
        # Keep internal blanks and source spacing. Only drop trailing blank lines
        # that belonged to the gap before the next article or file end.
        text = source_span.rstrip("\r\n")
        if not text.startswith(rows[start].rstrip("\r\n")):
            raise AssertionError("article title no longer starts its source span")
        docs.append({
            "index": index,
            "title": article["title"],
            "first_source_row": start,
            "end_source_row_exclusive": end,
            "source_span_sha256": digest_text(source_span),
            "model_text_sha256": digest_text(text),
            "text": text,
        })
    return docs


def train_tokenizer(train_docs: list[dict]) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL,
        show_progress=False,
    )
    tokenizer.train_from_iterator((doc["text"] for doc in train_docs), trainer=trainer, length=len(train_docs))
    if [tokenizer.token_to_id(symbol) for symbol in SPECIAL] != [0, 1, 2]:
        raise AssertionError("special IDs changed")
    if tokenizer.get_vocab_size() != VOCAB:
        raise AssertionError(f"expected {VOCAB} units, got {tokenizer.get_vocab_size()}")
    return tokenizer


def paragraph_fingerprints(docs: list[dict]) -> set[str]:
    result = set()
    for doc in docs:
        for part in re.split(r"\n\s*\n", doc["text"]):
            normalized = " ".join(part.casefold().split())
            if len(normalized.split()) >= 20:
                result.add(digest_text(normalized))
    return result


def encode_split(tokenizer: Tokenizer, docs: list[dict], split: str) -> tuple[dict, dict]:
    pad, bos, eos = (tokenizer.token_to_id(symbol) for symbol in SPECIAL)
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    block_doc: list[int] = []
    block_offset: list[int] = []
    article_token_counts = []
    roundtrip_failures = []
    for doc in docs:
        encoded = tokenizer.encode(doc["text"], add_special_tokens=False)
        content = encoded.ids
        restored = tokenizer.decode(content, skip_special_tokens=False)
        if restored != doc["text"]:
            roundtrip_failures.append(doc["index"])
        if pad in content or bos in content or eos in content:
            raise ValueError(f"literal reserved special token in {split} article {doc['index']}")
        sequence = [bos, *content, eos]
        article_token_counts.append(len(content))
        for offset in range(0, len(sequence) - 1, BLOCK):
            window = sequence[offset:offset + BLOCK + 1]
            valid = len(window) - 1
            if valid <= 0:
                continue
            input_row = np.full(BLOCK, pad, dtype=np.int32)
            target_row = np.full(BLOCK, -100, dtype=np.int32)
            visible = np.zeros(BLOCK, dtype=np.bool_)
            input_row[:valid] = window[:-1]
            target_row[:valid] = window[1:]
            visible[:valid] = True
            inputs.append(input_row)
            labels.append(target_row)
            masks.append(visible)
            block_doc.append(doc["index"])
            block_offset.append(offset)
    if roundtrip_failures:
        raise ValueError(f"{split} byte-level decoding not faithful in articles: {roundtrip_failures[:12]}")
    arrays = {
        "input_ids": np.stack(inputs),
        "labels": np.stack(labels),
        "attention_mask": np.stack(masks),
        "article_index": np.asarray(block_doc, dtype=np.int32),
        "first_token_offset": np.asarray(block_offset, dtype=np.int32),
    }
    if not np.array_equal(arrays["labels"] != -100, arrays["attention_mask"]):
        raise AssertionError("loss and real-input positions differ")
    actual_targets = int(arrays["attention_mask"].sum())
    expected_targets = sum(length + 1 for length in article_token_counts)  # content + EOS, never BOS
    if actual_targets != expected_targets:
        raise AssertionError("a target was lost or counted twice")
    if np.any((arrays["input_ids"] == pad) & arrays["attention_mask"]):
        raise AssertionError("PAD became a source word")
    if np.any(arrays["labels"][arrays["attention_mask"]] == bos):
        raise AssertionError("new article BOS became a prediction target")
    if int((arrays["labels"] == eos).sum()) != len(docs):
        raise AssertionError("each article must contribute one EOS target")
    if np.any(np.diff(arrays["article_index"]) < 0):
        raise AssertionError("blocks must remain ordered by article")
    report = {
        "articles": len(docs),
        "article_content_units": int(sum(article_token_counts)),
        "min_article_units": min(article_token_counts),
        "max_article_units": max(article_token_counts),
        "blocks": len(inputs),
        "actual_targets_including_eos": actual_targets,
        "eos_targets": len(docs),
        "full_capacity_slots": len(inputs) * BLOCK,
        "pad_slots_ignored_by_loss": len(inputs) * BLOCK - actual_targets,
        "pad_fraction": 1 - actual_targets / (len(inputs) * BLOCK),
        "all_articles_roundtrip_exact": True,
        "no_cross_article_window": True,
        "last_target_of_each_article_is_eos": True,
        "article_unit_quantiles": np.quantile(article_token_counts, [0, .25, .5, .75, 1]).tolist(),
    }
    return arrays, report


def main() -> None:
    began = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    old_path = SOURCE / "manifest.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    docs = {split: source_documents(split, old) for split in EXPECTED_ARTICLES}
    old_index = json.loads((SOURCE / "article_index.json").read_text(encoding="utf-8"))
    if sha256(SOURCE / "article_index.json") != old["article_index_sha256"]:
        raise ValueError("C15 article index changed")
    for split, items in docs.items():
        if split in old_index and [doc["title"] for doc in items] != [row["title"] for row in old_index[split]]:
            raise AssertionError(f"{split} article boundaries no longer match C15")
    complete_index = {
        split: [
            {key: value for key, value in doc.items() if key != "text"}
            for doc in items
        ]
        for split, items in docs.items()
    }
    complete_index_file = DEST / "article_source_index.json"
    complete_index_file.write_text(
        json.dumps(complete_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = train_tokenizer(docs["train"])
    tokenizer_file = DEST / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    fingerprints = {split: paragraph_fingerprints(items) for split, items in docs.items()}
    paragraph_overlap = {
        f"{a}_vs_{b}": len(fingerprints[a] & fingerprints[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    split_results = {}
    for split, items in docs.items():
        arrays, stats = encode_split(tokenizer, items, split)
        array_file = DEST / f"{split}_blocks.npz"
        np.savez_compressed(array_file, **arrays)
        split_results[split] = {
            "arrays_path": str(array_file.relative_to(WORK)).replace("\\", "/"),
            "arrays_sha256": sha256(array_file),
            "source_parquet_sha256": old["source_files"][split]["sha256"],
            **stats,
        }
        print(json.dumps({"split": split, **stats}, ensure_ascii=False), flush=True)

    first = docs["train"][0]
    first_pieces = tokenizer.encode(first["text"], add_special_tokens=False)
    unknown_example = "Homarus gammarus"
    unknown_ids = tokenizer.encode(unknown_example, add_special_tokens=False).ids
    trace = {
        "first_train_article": first["title"],
        "first_source_row": first["first_source_row"],
        "first_article_source_span_sha256": first["source_span_sha256"],
        "first_article_model_text_sha256": first["model_text_sha256"],
        "first_article_text_prefix": first["text"][:160],
        "first_24_pieces": first_pieces.tokens[:24],
        "first_24_ids": first_pieces.ids[:24],
        "first_block_first_24_inputs": np.load(DEST / "train_blocks.npz")["input_ids"][0, :24].tolist(),
        "first_block_first_24_targets": np.load(DEST / "train_blocks.npz")["labels"][0, :24].tolist(),
        "validation_rare_word_example": unknown_example,
        "validation_rare_word_ids": unknown_ids,
        "validation_rare_word_decoded": tokenizer.decode(unknown_ids),
    }
    manifest = {
        "task": "C18 one-source actual causal LM data, tokenizer fitted on train documents only",
        "upstream_c15_manifest_sha256": sha256(old_path),
        "upstream_article_index_sha256": sha256(SOURCE / "article_index.json"),
        "upstream_article_index_present_splits": sorted(old_index),
        "complete_article_index_path": str(complete_index_file.relative_to(WORK)).replace("\\", "/"),
        "complete_article_index_sha256": sha256(complete_index_file),
        "raw_variant_note": "WikiText-2 raw-v1 is pre-extracted/token-spaced, not untouched Wikipedia HTML",
        "article_text_rule": "join original Parquet text rows from top-level title to before next title; strip trailing CR/LF only",
        "tokenizer": {
            "algorithm": "byte-level BPE",
            "trained_on": "600 train articles only",
            "library": "tokenizers",
            "version": tokenizers.__version__,
            "vocab_size": tokenizer.get_vocab_size(),
            "special_ids": {symbol: tokenizer.token_to_id(symbol) for symbol in SPECIAL},
            "model_path": str(tokenizer_file.relative_to(WORK)).replace("\\", "/"),
            "model_sha256": sha256(tokenizer_file),
            "roundtrip_scope": "every model-text article in all three splits",
            "normalization": "none added by this script; upstream raw-v1 already changes source Wikipedia formatting",
        },
        "block_rule": {
            "context_positions": BLOCK,
            "per_article_stream": "BOS + byte-BPE content + EOS",
            "stride": BLOCK,
            "input": "stream[offset:offset+128]",
            "target": "stream[offset+1:offset+129]",
            "short_last_block": "PAD input; -100 label ignored; attention_mask false on PAD",
            "document_crossing": "forbidden: each article contributes its own blocks; no target or context from next article",
            "causal_visibility": "attention_mask on keys AND lower-triangular query/key mask in model",
        },
        "splits": split_results,
        "cross_split_exact_normalized_paragraphs_20plus_words": paragraph_overlap,
        "paragraph_audit_limit": "exact normalized long paragraphs only; no complete near-duplicate, quote, or web provenance audit",
        "length_weighting": "one loss term per real content unit and per-article EOS; longer articles contribute more targets",
        "trace": trace,
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - began,
    }
    manifest_file = DEST / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_file), "tokenizer_sha256": manifest["tokenizer"]["model_sha256"],
                      "paragraph_overlap": paragraph_overlap, "elapsed_seconds": manifest["elapsed_seconds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
