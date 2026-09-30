"""从 WikiText-2 训练文章学课堂版词内字节对合并，并核未知词可回读。"""

from __future__ import annotations

import collections
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "wikitext2_raw"
RESULT = WORK / "results" / "word_subword_probe.json"
MERGES = DATA / "word_internal_byte_merges.json"
TYPE_LIMIT = 30_000
MERGE_LIMIT = 256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_pair(sequence: tuple[int, ...], pair: tuple[int, int], replacement: int) -> tuple[int, ...]:
    output = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index:index + 2] == pair:
            output.append(replacement)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


def encode_field(field: str, merges: list[dict]) -> tuple[int, ...]:
    sequence = tuple(field.encode("utf-8"))
    for item in merges:
        sequence = replace_pair(sequence, tuple(item["pair"]), item["new_id"])
    return sequence


def decode_field(sequence: tuple[int, ...], piece_bytes: dict[int, bytes]) -> str:
    return b"".join(piece_bytes[index] for index in sequence).decode("utf-8")


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    train_file = DATA / "train-00000-of-00001.parquet"
    if sha256(train_file) != manifest["source_files"]["train"]["sha256"]:
        raise ValueError("子词训练所用文章不属于当前训练 split")
    rows = pq.read_table(train_file, columns=["text"])["text"].to_pylist()
    frequencies = collections.Counter(field for row in rows for field in row.split() if field.isalpha())
    selected_types = sorted(frequencies, key=lambda field: (-frequencies[field], field))[:TYPE_LIMIT]
    state = {word: tuple(word.encode("utf-8")) for word in selected_types}
    piece_bytes = {index: bytes([index]) for index in range(256)}
    merges = []
    for rank in range(MERGE_LIMIT):
        pair_counts = collections.Counter()
        for word in selected_types:
            sequence = state[word]
            if len(sequence) > 1:
                for pair in zip(sequence[:-1], sequence[1:]):
                    pair_counts[pair] += frequencies[word]
        if not pair_counts:
            break
        pair, count = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
        new_id = 256 + rank
        piece_bytes[new_id] = piece_bytes[pair[0]] + piece_bytes[pair[1]]
        merges.append({"rank": rank, "pair": list(pair), "new_id": new_id, "weighted_count": count,
                       "bytes_hex": piece_bytes[new_id].hex()})
        state = {word: replace_pair(sequence, pair, new_id) for word, sequence in state.items()}
        if (rank + 1) % 64 == 0:
            print(f"learned {rank + 1} merges from {len(selected_types)} training word types", flush=True)
    MERGES.write_text(json.dumps(merges, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    vocab = set(json.loads((DATA / "word_vocab.json").read_text(encoding="utf-8")))
    validation_rows = pq.read_table(DATA / "validation-00000-of-00001.parquet", columns=["text"])["text"].to_pylist()
    unseen_field = next(
        field for row in validation_rows for field in row.split()
        if field.isalpha() and len(field) >= 6 and field not in vocab
    )
    rare_pieces = encode_field(unseen_field, merges)
    reconstructed = decode_field(rare_pieces, piece_bytes)
    if reconstructed != unseen_field:
        raise AssertionError("验证集中未见字段不能无损回读")

    source_line = manifest["first_training_body_line"]["raw"].strip()
    fields = source_line.split()[:18]
    encoded_fields = [encode_field(field, merges) for field in fields]
    if any(decode_field(pieces, piece_bytes) != field for field, pieces in zip(fields, encoded_fields)):
        raise AssertionError("来源片段字段不能逐个回读")
    record = {
        "purpose": "classroom word-internal byte BPE only; main word LM still uses top-10k word vocabulary",
        "source_training_file_sha256": sha256(train_file),
        "vocab_sha256": manifest["vocab_sha256"],
        "input_field_rule": "Python split on supplied WikiText spacing; training merge counts only alphabetic fields, no cross-word merges",
        "training_alpha_word_types_total": len(frequencies),
        "training_alpha_token_occurrences": sum(frequencies.values()),
        "types_used_to_learn_merges": len(selected_types),
        "learned_merges": len(merges),
        "base_utf8_byte_symbols": 256,
        "possible_symbols_after_merges": 256 + len(merges),
        "merge_file": str(MERGES.relative_to(WORK)).replace("\\", "/"),
        "merge_file_sha256": sha256(MERGES),
        "first_ten_merges": merges[:10],
        "validation_word_unseen_by_top10k": {
            "original_field": unseen_field,
            "word_level_mapping": "<unk>",
            "utf8_byte_count": len(unseen_field.encode("utf-8")),
            "bpe_piece_ids": list(rare_pieces),
            "bpe_piece_hex": [piece_bytes[index].hex() for index in rare_pieces],
            "decoded": reconstructed,
        },
        "source_line_row": manifest["first_training_body_line"]["source_row"],
        "source_line_first_18_fields": fields,
        "source_line_first_18_field_counts": {
            "space_separated_fields": len(fields),
            "unicode_codepoints_inside_fields": sum(len(field) for field in fields),
            "utf8_bytes_inside_fields": sum(len(field.encode("utf-8")) for field in fields),
            "learned_bpe_pieces_inside_fields": sum(len(pieces) for pieces in encoded_fields),
        },
        "time_seconds": time.perf_counter() - started,
        "scope_note": "Can reconstruct individual original fields, not exact original inter-field spaces or article markup; does not claim GPT byte-level BPE identity or compare word/char/BPE perplexities.",
    }
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("learned_merges", "validation_word_unseen_by_top10k", "source_line_first_18_field_counts", "time_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
