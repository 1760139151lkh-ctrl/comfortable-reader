"""Reconstruct one real CoVal A/B vote's Bradley-Terry score difference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from analyze_coval_preferences import ranking


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work/data/coval_eec757"


def sigma(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def votes(assessments: list[dict], kind: str) -> tuple[int, int, int]:
    a = b = tie = 0
    for record in assessments:
        blocks = record["ranking_blocks"][kind]
        if not blocks:
            continue
        ranks = ranking(blocks[0]["ranking"])
        if ranks["A"] < ranks["B"]:
            a += 1
        elif ranks["B"] < ranks["A"]:
            b += 1
        else:
            tie += 1
    return a, b, tie


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("work/results/c28_pairwise_preference_probe.json"))
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    source = DATA / "comparisons.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["files"]["comparisons.jsonl"]["sha256"]:
        raise RuntimeError("source changed")
    record = json.loads(source.open(encoding="utf-8").readline())
    assessments = record["metadata"]["assessments"]
    world = votes(assessments, "world")
    personal = votes(assessments, "personal")
    if world != (8, 3, 3) or personal != (7, 6, 1):
        raise RuntimeError("fixed source aggregate no longer matches")
    values = {}
    for name, (count_a, count_b, ties) in (("world", world), ("personal", personal)):
        difference = math.log(count_a / count_b)
        probability = sigma(difference)
        derivative = count_a * (1 - probability) - count_b * probability
        shifted = sigma((difference/2 + 3) - (-difference/2 + 3))
        values[name] = {
            "A_over_B": count_a, "B_over_A": count_b, "ties_excluded": ties,
            "empirical_A_share_among_non_ties": count_a / (count_a + count_b),
            "MLE_score_difference_A_minus_B": difference,
            "model_probability_at_MLE": probability,
            "log_likelihood_derivative_at_MLE": derivative,
            "probability_after_add_three_to_both_scores": shifted,
        }
    result = {
        "scope": "one CoVal source pair, actual human vote counts, no response/rationale/annotator text copied; BT likelihood hand check not user aesthetics",
        "source_sha256": manifest["files"]["comparisons.jsonl"]["sha256"],
        "prompt_id": record["prompt_id"],
        "topic": "whether people should stop eating beef; candidate A weighs several factors, B emphasizes individual choice",
        "comparisons": values,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("world delta", round(values["world"]["MLE_score_difference_A_minus_B"], 6),
          "personal delta", round(values["personal"]["MLE_score_difference_A_minus_B"], 6))


if __name__ == "__main__":
    main()
