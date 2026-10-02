"""Use a newly computed importance receipt in the fixed C46 importance arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arithmetic_curriculum as math_task
import c46_sequential_math_code as sequence

ROOT = math_task.ROOT


def main(importance_dir: Path, out_dir: Path, steps: int) -> None:
    report_path = importance_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    importance_path = ROOT / report["importance_path"]
    if importance_path.resolve() != (importance_dir / "importance.pt").resolve():
        raise RuntimeError("Importance report points to another file")
    if math_task.file_sha(importance_path) != report["importance_sha256"]:
        raise RuntimeError("Importance bytes differ from its receipt")
    if (report["math_checkpoint_sha256"] != sequence.EXPECTED_PARENT
            or report["old_math_train_examples"] != 96
            or report["math_manifest_sha256"] != math_task.load_manifest()[1]):
        raise RuntimeError("Own importance source is not this chapter's 96-example old math task")
    sequence.IMPORTANCE_PATH = importance_path
    sequence.EXPECTED_IMPORTANCE = report["importance_sha256"]
    sequence.train("importance", out_dir, steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--importance-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1600)
    args = ap.parse_args()
    main(ROOT / args.importance_dir, ROOT / args.out_dir, args.steps)
