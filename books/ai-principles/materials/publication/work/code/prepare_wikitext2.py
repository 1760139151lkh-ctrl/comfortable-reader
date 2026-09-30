"""核 WikiText-2 raw 镜像，保留文章边界，建仅来自训练文章的词表。"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data" / "wikitext2_raw"
BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e/wikitext-2-raw-v1/"
EXPECTED = {
    "train": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "validation": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
    "test": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
}
SPECIAL = ("<unk>", "<bos>", "<eos>", "<para>")
WORD_LIMIT = 10_000
TOP_HEADING = re.compile(r"^= [^=].* =$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_arrays_stably(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """固定 NPZ 成员顺序与时间戳，使相同字段数组重建后保留相同 SHA。"""
    temporary = path.with_suffix(path.suffix + ".part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    temporary.replace(path)


def download_and_read(split: str) -> tuple[list[str], dict]:
    name = f"{split}-00000-of-00001.parquet"
    path = DATA / name
    if not path.exists():
        temporary = DATA / (name + ".part")
        request = urllib.request.Request(BASE + name, headers={"User-Agent": "AI-textbook-local-study/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    actual_hash = sha256(path)
    if actual_hash != EXPECTED[split]:
        raise ValueError(f"{split}: 固定 Parquet 字节身份改变，实际 {actual_hash}")
    table = pq.read_table(path, columns=["text"])
    rows = table["text"].to_pylist()
    if not all(isinstance(value, str) for value in rows):
        raise ValueError(f"{split}: text 列并非完整字符串")
    return rows, {"path": str(path.relative_to(WORK)).replace("\\", "/"), "url": BASE + name,
                  "bytes": path.stat().st_size, "sha256": actual_hash, "rows": len(rows)}


def parse_articles(rows: list[str]) -> list[dict]:
    articles = []
    current = None
    pending_paragraph = False
    for source_row, raw in enumerate(rows):
        line = raw.strip()
        is_article_title = (
            TOP_HEADING.fullmatch(line)
            and (source_row == 0 or not rows[source_row - 1].strip())
            and (source_row + 1 == len(rows) or not rows[source_row + 1].strip())
        )
        if is_article_title:
            if current is not None:
                while current["tokens"] and current["tokens"][-1] == "<para>":
                    current["tokens"].pop()
                articles.append(current)
            current = {
                "title": line[2:-2],
                "title_source_row": source_row,
                "tokens": line.split(),
                "raw_lines": [raw],
                "source_rows": [source_row],
            }
            pending_paragraph = False
        elif not line:
            if current is not None and current["tokens"]:
                pending_paragraph = True
        else:
            if current is None:
                raise ValueError(f"文章标题前有非空正文，源行 {source_row}")
            if pending_paragraph and current["tokens"][-1] != "<para>":
                current["tokens"].append("<para>")
            current["tokens"].extend(line.split())
            current["raw_lines"].append(raw)
            current["source_rows"].append(source_row)
            pending_paragraph = False
    if current is not None:
        while current["tokens"] and current["tokens"][-1] == "<para>":
            current["tokens"].pop()
        articles.append(current)
    shorts = [(article["title"], len(article["tokens"])) for article in articles if len(article["tokens"]) < 10]
    if not articles or shorts:
        raise ValueError(f"文章解析为空或有异常短文: articles={len(articles)}, short={shorts[:12]}")
    return articles


def map_documents(articles: list[dict], stoi: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    flat = []
    offsets = [0]
    raw_count = 0
    unknown = 0
    for article in articles:
        tokens = article["tokens"]
        mapped = [stoi["<bos>"], stoi["<bos>"]]
        for token in tokens:
            mapped.append(stoi.get(token, stoi["<unk>"]))
            raw_count += 1
            unknown += token not in stoi
        mapped.append(stoi["<eos>"])
        flat.extend(mapped)
        offsets.append(len(flat))
    return np.asarray(flat, dtype=np.int32), np.asarray(offsets, dtype=np.int64), {
        "raw_token_count_including_para": raw_count,
        "unknown_target_count": unknown,
        "unknown_target_rate": unknown / raw_count,
        "next_token_examples_including_eos": raw_count + len(articles),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    source = {}
    articles = {}
    for split in ("train", "validation", "test"):
        rows, source[split] = download_and_read(split)
        articles[split] = parse_articles(rows)
    title_sets = {name: {article["title"] for article in docs} for name, docs in articles.items()}
    exact_content_sets = {
        name: {
            hashlib.sha256("\n".join(article["raw_lines"]).encode("utf-8")).hexdigest()
            for article in docs
        }
        for name, docs in articles.items()
    }
    overlap = {}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap[first + "_vs_" + second] = {
            "same_titles": sorted(title_sets[first] & title_sets[second]),
            "exact_article_content_hashes": len(exact_content_sets[first] & exact_content_sets[second]),
        }
    if any(value["same_titles"] or value["exact_article_content_hashes"] for value in overlap.values()):
        raise ValueError(f"文章标题／原文完整内容跨 split 重叠：{overlap}")

    train_counts = collections.Counter(
        token for article in articles["train"] for token in article["tokens"] if token not in SPECIAL
    )
    sorted_words = sorted(train_counts, key=lambda token: (-train_counts[token], token))[:WORD_LIMIT]
    vocab = list(SPECIAL) + sorted_words
    stoi = {token: index for index, token in enumerate(vocab)}
    arrays = {}
    stats = {}
    for split in ("train", "validation", "test"):
        token_ids, offsets, token_stats = map_documents(articles[split], stoi)
        arrays[split + "_tokens"] = token_ids
        arrays[split + "_offsets"] = offsets
        stats[split] = {
            "articles": len(articles[split]),
            "titles_first_five": [article["title"] for article in articles[split][:5]],
            "tokens_per_article_min": min(len(article["tokens"]) for article in articles[split]),
            "tokens_per_article_max": max(len(article["tokens"]) for article in articles[split]),
            **token_stats,
        }
    arrays_file = DATA / "word_documents.npz"
    save_arrays_stably(arrays_file, arrays)
    vocab_file = DATA / "word_vocab.json"
    vocab_file.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    article_index = {
        split: [
            {
                "title": article["title"],
                "title_source_row": article["title_source_row"],
                "first_body_source_row": next(
                    (row for row, raw in zip(article["source_rows"][1:], article["raw_lines"][1:])
                     if raw.strip() and not raw.strip().startswith("= =")),
                    None,
                ),
                "token_count": len(article["tokens"]),
                "exact_raw_sha256": hashlib.sha256(
                    "\n".join(article["raw_lines"]).encode("utf-8")
                ).hexdigest(),
            }
            for article in articles[split]
        ]
    }
    index_file = DATA / "article_index.json"
    index_file.write_text(json.dumps(article_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    first_article = articles["train"][0]
    first_nonheading = next(
        ({"source_row": row, "raw": raw, "split_tokens": raw.strip().split()[:18]}
         for row, raw in zip(first_article["source_rows"][1:], first_article["raw_lines"][1:])
         if raw.strip() and not raw.strip().startswith("=")),
        None,
    )
    manifest = {
        "upstream": "Salesforce/wikitext WikiText-2 raw variant, WikiText paper arXiv:1609.07843",
        "mirror": "https://huggingface.co/datasets/Salesforce/wikitext",
        "repository_commit": "b08601e",
        "source_files": source,
        "license_note": "HF tags cc-by-sa-3.0 and gfdl; card prose mentions CC BY-SA 4.0; consult upstream/each source before redistribution.",
        "line_processing": "top-level '= Title =' begins article; split() on supplied spacing, one <para> between blank-separated spans; two <bos> left context; one <eos> target; no cross-article examples",
        "special_ids": {token: stoi[token] for token in SPECIAL},
        "word_limit_excluding_specials": WORD_LIMIT,
        "vocab_size": len(vocab),
        "vocab_file": str(vocab_file.relative_to(WORK)).replace("\\", "/"),
        "vocab_sha256": sha256(vocab_file),
        "arrays_file": str(arrays_file.relative_to(WORK)).replace("\\", "/"),
        "arrays_sha256": sha256(arrays_file),
        "article_index_file": str(index_file.relative_to(WORK)).replace("\\", "/"),
        "article_index_sha256": sha256(index_file),
        "split_stats": stats,
        "cross_split_exact_audit": overlap,
        "first_training_article_title": first_article["title"],
        "first_training_body_line": first_nonheading,
        "note": "raw-v1 only means no dataset-inserted <unk>; source text already has spaces around many signs and @-@ artifacts. split() is not general English tokenization."
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"vocab_size": len(vocab), "stats": stats,
                      "first_training_body_line": first_nonheading}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
