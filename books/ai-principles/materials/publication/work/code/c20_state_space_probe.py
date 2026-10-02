"""Check fixed-state recurrence/convolution identity and an input-selected counterexample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fixed(x: np.ndarray, a: float) -> np.ndarray:
    state = 0.0
    result = []
    for value in x:
        state = a * state + float(value)
        result.append(state)
    return np.asarray(result)


def selected(x: np.ndarray) -> np.ndarray:
    state = 0.0
    result = []
    for value in x:
        keep = 0.9 if value == 0 else 0.1
        state = keep * state + float(value)
        result.append(state)
    return np.asarray(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dest = (ROOT / args.out).resolve()
    if not dest.is_relative_to(ROOT / "work") or dest.exists():
        raise ValueError("Choose a new output file under package/work")
    dest.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([1., 2., 0., -1.])
    a = .8
    recurrent = fixed(x, a)
    kernel = a ** np.arange(len(x), dtype=np.float64)
    convolution = np.convolve(x, kernel)[:len(x)]
    first = np.array([1., 0.])
    second = np.array([0., 2.])
    separate = selected(first) + selected(second)
    joint = selected(first + second)
    result = {
        "scope": "Author-defined two scalar recurrences; no trained S4/Mamba/long-context benchmark",
        "fixed_input": x.tolist(), "a": a, "fixed_kernel": kernel.tolist(),
        "fixed_recurrence": recurrent.tolist(), "fixed_convolution": convolution.tolist(),
        "fixed_max_absolute_difference": float(np.max(np.abs(recurrent - convolution))),
        "selection_rule": "keep=0.9 on zero input, else 0.1; add current input",
        "first": first.tolist(), "second": second.tolist(),
        "selected_first_plus_selected_second": separate.tolist(),
        "selected_joint": joint.tolist(),
        "selected_superposition_defect_last": float(joint[-1] - separate[-1]),
    }
    if result["fixed_max_absolute_difference"] > 1e-12:
        raise AssertionError("Fixed recurrence/convolution should agree")
    if not np.isclose(result["selected_superposition_defect_last"], -.8):
        raise AssertionError("Selected transition should break fixed linear convolution")
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
