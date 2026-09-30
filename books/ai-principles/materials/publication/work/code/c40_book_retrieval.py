"""C40: source-bound lexical retrieval over the actually published local book.

This is a transparent character-bigram BM25-style classroom implementation,
not a pretrained embedding model and not proof that the C19 LM understands the
retrieved Chinese paragraphs. It checks every publication SHA before indexing.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

from tokenizers import Tokenizer

WORK = Path(__file__).resolve().parents[1]
PACKAGE = WORK.parent
RUNS = WORK / "runs"
PUB = WORK / "publication.json"
TOKENIZER = WORK / "data/wikitext2_causal/tokenizer.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text.replace("\\(", "").replace("\\)", "").replace("**", "").replace("`", "")


def terms(text):
    text = clean(text).lower()
    out = re.findall(r"[a-z0-9_]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            out.append(run)
    return out


def paragraphs(path: Path, expected_sha: str):
    actual = sha(path)
    if actual != expected_sha:
        raise RuntimeError(f"published source drifted: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    current_anchor = None
    heading = ""
    pending = []
    first_line = None
    in_fence = False
    in_math = False
    output = []

    def flush(end_line):
        nonlocal pending, first_line
        value = clean(" ".join(pending)).strip()
        if len(value) >= 50 and len(terms(value)) >= 8:
            output.append({"path": str(path.relative_to(PACKAGE)).replace("\\", "/"),
                           "source_sha256": expected_sha, "anchor": current_anchor,
                           "heading": heading, "start_line": first_line, "end_line": end_line,
                           "text": value, "text_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()})
        pending = []
        first_line = None

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("~~~", "```")):
            flush(number - 1)
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped == r"\[":
            flush(number - 1)
            in_math = True
            continue
        if stripped == r"\]":
            in_math = False
            continue
        if in_math:
            continue
        anchor_match = re.fullmatch(r'<a id="([^"]+)"></a>', stripped)
        if anchor_match:
            flush(number - 1)
            current_anchor = anchor_match.group(1)
            continue
        if stripped.startswith("#"):
            flush(number - 1)
            heading = stripped.lstrip("# ")
            continue
        if not stripped or stripped.startswith(("|", "![", "<", "</", "1. ", "2. ", "3. ", "4. ")):
            flush(number - 1)
            continue
        if first_line is None:
            first_line = number
        pending.append(stripped)
    flush(len(lines))
    return output


def bm25(query, documents, counts, df, mean_len, k1=1.2, b=0.75):
    q = sorted(set(terms(query)))  # stable floating-point addition order
    scores = []
    n = len(documents)
    for index, tf in enumerate(counts):
        size = sum(tf.values())
        score = 0.0
        for term in q:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * size / mean_len))
        scores.append((score, index))
    return sorted(scores, reverse=True)


def questions():
    return [
        ("第十九章自己训练的文章模型有多少可训练参数？", "第十九章_从随机权重训练到真实生成.md", "nineteenth-model"),
        ("同一个字符串可能由不同BPE编号路径解出，这和字符串概率有何区别？", "第十八章_文章怎样进入下一词模型.md", "eighteenth-tokenizer"),
        ("查询与键的点积为什么要除以根号键维度？", "第十七章_每个位置怎样读到需要的信息.md", "seventeenth-from-alignment"),
        ("Dolly指令训练为什么提示可读却只对回答算损失？", "第二十五章_文章模型怎样学着回应要求.md", "twentyfifth-mask"),
        ("因果模型超过128位置后为什么不能只弹出旧KV缓存？", "第三十七章_训练好的模型怎样接住一次请求.md", "thirtyseventh-window"),
        ("工具返回为什么不是模型此次采样动作的对数概率？", "第三十八章_模型怎样学会请求工具并使用返回.md", "thirtyeighth-probability"),
        ("模型会写正确工具请求却不会照读标题返回，这个失败怎么发现的？", "第三十八章_模型怎样学会请求工具并使用返回.md", "thirtyeighth-first-failure"),
        ("没有收到远端回执时，为什么无法知道工具是否已经执行？", "第三十九章_请求发出以后怎样知道它做了什么.md", "thirtyninth-two-worlds"),
        ("同一个操作编号重试但参数改变时为什么要拒绝？", "第三十九章_请求发出以后怎样知道它做了什么.md", "thirtyninth-idempotence"),
        ("图节点重新编号后，节点分类输出为什么应同样重新排序？", "第三十六章_连接关系怎样进入学习.md", "thirtysixth-permutation"),
        ("双目图像视差和三维深度如何随基线与焦距变化？", "第三十四章_两张照片怎样约束三维世界.md", "thirtyfourth-disparity"),
        ("训练、验证、测试资料为什么不应按一条时间序列随机分段？", "第六章_训练得好下一次还会好吗.md", "sixth-splitting"),
    ]


def main(out_dir):
    out_dir = out_dir.resolve()
    if not out_dir.is_relative_to(RUNS.resolve()) or out_dir.exists():
        raise ValueError("choose a new work/runs directory")
    publication = json.loads(PUB.read_text(encoding="utf-8"))
    if len(publication["units"]) < 40:
        raise RuntimeError("the guide and first 39 chapters must be locally published")
    # Freeze the teaching corpus even after this chapter is published.
    source_units = publication["units"][:40]
    docs = []
    for unit in source_units:
        item = unit["artifact"]
        docs.extend(paragraphs(PACKAGE / item["path"], item["sha256"]))
    if not docs:
        raise RuntimeError("published corpus empty")
    counts = [collections.Counter(terms(doc["heading"] + " " + doc["text"])) for doc in docs]
    df = collections.Counter(token for tf in counts for token in tf)
    mean_len = sum(sum(tf.values()) for tf in counts) / len(counts)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    results = []
    for question, file_name, anchor in questions():
        scores = bm25(question, docs, counts, df, mean_len)
        hits = []
        gold_ranks = []
        for rank, (score, index) in enumerate(scores, 1):
            doc = docs[index]
            if doc["path"].endswith("/" + file_name) and doc["anchor"] == anchor:
                gold_ranks.append(rank)
            if rank <= 5:
                hits.append({"rank": rank, "score": score, "path": doc["path"],
                             "anchor": doc["anchor"], "start_line": doc["start_line"],
                             "text_sha256": doc["text_sha256"], "excerpt": doc["text"][:240]})
        top = docs[scores[0][1]]
        full_prompt = "Q: " + question + "\n[R1] " + top["text"] + "\nA:"
        fragments = [part.strip() for part in re.split(r"(?<=[。！？；])", top["text"]) if part.strip()]
        # Deliberately naive: this can select a heading/citation with no answer.
        short = min(fragments, key=lambda part: len(tokenizer.encode("Q: " + question + "\n[R1] " + part + "\nA:", add_special_tokens=False).ids)) if fragments else top["text"]
        # C19 generation prepends one BOS position before these prompt tokens.
        full_count = 1 + len(tokenizer.encode(full_prompt, add_special_tokens=False).ids)
        short_count = 1 + len(tokenizer.encode("Q: " + question + "\n[R1] " + short + "\nA:", add_special_tokens=False).ids)
        results.append({"question": question, "gold_file": file_name, "gold_anchor": anchor,
                        "best_gold_rank": min(gold_ranks) if gold_ranks else None,
                        "top5": hits,
                        "c19_context_check": {"context_limit": 128, "reserved_answer_tokens": 16,
                                              "top1_full_prompt_tokens_including_bos": full_count,
                                              "top1_full_fits_with_reserved_answer": full_count <= 112,
                                              "naive_shortest_fragment": short,
                                              "naive_shortest_fragment_prompt_tokens_including_bos": short_count,
                                              "naive_shortest_fragment_fits_with_reserved_answer": short_count <= 112,
                                              "naive_fragment_may_omit_answer": True}})
    out_dir.mkdir(parents=True)
    (out_dir / "paragraph_index.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_sha = sha(out_dir / "paragraph_index.json")
    source_example = next((doc for doc in docs if doc["path"].endswith("/第三十七章_训练好的模型怎样接住一次请求.md")
                           and doc["anchor"] == "thirtyseventh-window" and doc["start_line"] == 91), None)
    if source_example is None:
        raise RuntimeError("C37 source for hand-compressed example changed; review its meaning and locator")
    manual_evidence = "[R1 C37:91] Sliding resets positions; old KV has old positions and saw dropped tokens. Rebuild."
    example_question = questions()[4][0]
    manual_tokens = 1 + len(tokenizer.encode("Q: " + example_question + "\n" + manual_evidence + "\nA:", add_special_tokens=False).ids)
    manual_example = {"question": example_question, "source_path": source_example["path"],
                      "source_anchor": source_example["anchor"], "source_start_line": source_example["start_line"],
                      "source_text_sha256": source_example["text_sha256"], "author_paraphrase_not_verbatim": manual_evidence,
                      "prompt_tokens_including_bos": manual_tokens, "reserved_answer_tokens": 16,
                      "fits_128": manual_tokens + 16 <= 128,
                      "caveat": "human checked against the C37 paragraph; fits by using terse English but does not establish that C19 understands the Chinese question or can answer it"}
    report = {"scope": "current published guide+chapters1..39 paragraph lexical retrieval, author-chosen questions/gold anchors; no generative answer benchmark",
              "publication_sha256": sha(PUB), "publication_total_units_at_run": len(publication["units"]),
              "frozen_corpus_first_units": len(source_units), "tokenizer_sha256": sha(TOKENIZER), "paragraphs": len(docs),
              "mean_character_bigram_and_latin_term_count": mean_len, "index_sha256": index_sha,
              "scorer": "BM25-style character bigrams for contiguous Chinese plus lowercased Latin/digits; k1=1.2,b=.75,idf=log(1+(N-df+0.5)/(df+0.5))",
              "gold_questions": len(results), "recall_at_1": sum(x["best_gold_rank"] == 1 for x in results),
              "recall_at_5": sum(x["best_gold_rank"] is not None and x["best_gold_rank"] <= 5 for x in results),
              "recall_at_10": sum(x["best_gold_rank"] is not None and x["best_gold_rank"] <= 10 for x in results),
              "full_top1_context_fits_128_with_16_reserved": sum(x["c19_context_check"]["top1_full_fits_with_reserved_answer"] for x in results),
              "naive_shortest_fragment_fits": sum(x["c19_context_check"]["naive_shortest_fragment_fits_with_reserved_answer"] for x in results),
              "questions": results, "manual_source_checked_compression_example": manual_example,
              "limits": "gold question set author-selected after published source exists; lexical paragraph retrieval cannot establish final answer correctness, source authority, user memory or Chinese ability of weak C19 English LM"}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paragraphs": len(docs), "R1": report["recall_at_1"], "R5": report["recall_at_5"], "R10": report["recall_at_10"], "full_prompt_fit": report["full_top1_context_fits_128_with_16_reserved"], "naive_fragment_fit": report["naive_shortest_fragment_fits"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.out_dir)
