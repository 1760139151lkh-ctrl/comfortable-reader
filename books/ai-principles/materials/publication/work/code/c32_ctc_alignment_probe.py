"""Enumerate tiny CTC paths and compare with PyTorch's exact path sum."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = {0: "", 1: "A", 2: "B"}


def collapse(path: tuple[int, ...]) -> str:
    result = []
    previous = None
    for symbol in path:
        if symbol != previous and symbol:
            result.append(SYMBOLS[symbol])
        previous = symbol
    return "".join(result)


def enumerate_probability(probs: torch.Tensor, target: str) -> tuple[torch.Tensor, list]:
    selected = []
    total = probs.new_zeros(())
    for path in itertools.product(range(probs.shape[-1]), repeat=probs.shape[0]):
        if collapse(path) == target:
            term = torch.stack([probs[t, symbol] for t, symbol in enumerate(path)]).prod()
            total = total + term
            selected.append({"path": [SYMBOLS[s] if s else "blank" for s in path],
                             "probability": float(term.detach())})
    return total, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"preserve existing CTC probe: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # A blank is the most likely single-frame symbol, yet the label A
    # collects more total probability over all alignments.
    two = torch.tensor([[0.4, 0.35, 0.25], [0.4, 0.35, 0.25]], dtype=torch.float64)
    masses = {target: float(enumerate_probability(two, target)[0])
              for target in ("", "A", "B", "AB", "BA")}
    if abs(sum(masses.values()) - 1.0) > 1e-12:
        raise AssertionError("enumerated label probabilities do not sum to one")
    best_path = tuple(two.argmax(dim=1).tolist())

    three = torch.tensor([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1],
                          [0.2, 0.3, 0.5]], dtype=torch.float64)
    comparisons = {}
    for target, ids in (("AB", [1, 2]), ("AA", [1, 1])):
        logits = three.log().clone().detach().requires_grad_()
        probs = F.softmax(logits, dim=-1)
        explicit, paths = enumerate_probability(probs, target)
        explicit_loss = -explicit.log()
        explicit_gradient = torch.autograd.grad(explicit_loss, logits,
                                                retain_graph=True)[0]
        ctc = F.ctc_loss(F.log_softmax(logits, dim=-1).unsqueeze(1),
                         torch.tensor(ids, dtype=torch.long),
                         torch.tensor([3], dtype=torch.long),
                         torch.tensor([len(ids)], dtype=torch.long),
                         blank=0, reduction="none")
        ctc_gradient = torch.autograd.grad(ctc, logits)[0]
        comparisons[target] = {
            "explicit_probability": float(explicit.detach()),
            "explicit_nll": float(explicit_loss.detach()),
            "torch_ctc_nll": float(ctc.detach()),
            "maximum_logit_gradient_difference": float(
                (ctc_gradient - explicit_gradient).abs().max()),
            "contributing_paths": paths,
        }
        if abs(comparisons[target]["explicit_nll"] -
               comparisons[target]["torch_ctc_nll"]) > 1e-12:
            raise AssertionError("CTC path sum does not match PyTorch")
    report = {
        "two_frames_each": [0.4, 0.35, 0.25],
        "two_frame_best_path": [SYMBOLS[s] if s else "blank" for s in best_path],
        "best_path_collapsed_label": collapse(best_path),
        "two_frame_label_masses": masses,
        "most_probable_label": max(masses, key=masses.get),
        "three_frame_probs": three.tolist(),
        "comparisons": comparisons,
        "torch_version": torch.__version__,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_path_label": report["best_path_collapsed_label"],
                      "most_probable_label": report["most_probable_label"],
                      "AB": comparisons["AB"]["explicit_nll"],
                      "AA": comparisons["AA"]["explicit_nll"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
