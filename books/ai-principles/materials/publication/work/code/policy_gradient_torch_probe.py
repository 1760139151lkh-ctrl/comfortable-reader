"""C23: show which PyTorch quantity actually carries a policy gradient."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("work/results/c23_torch_gradient_probe.json")
    )
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if out.exists():
        raise FileExistsError(f"keep previous result; choose another --out: {out}")

    dtype = torch.float64
    theta = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
    optimizer = torch.optim.SGD([theta], lr=0.2)
    old_branch_probabilities = (0.375, 0.125, 0.5)
    returns = (3.0, -3.0, 4.0)
    baseline = 2.75
    before = {
        "theta": float(theta.detach()),
        "p_express": float(torch.sigmoid(theta).detach()),
        "exact_objective": 2.75,
    }

    p = torch.sigmoid(theta)
    log_express = torch.log(p)
    log_slow = torch.log1p(-p)
    # Branch probabilities come from the OLD data-collection policy. They are
    # fixed numerical weights here, not a differentiable second copy of p.
    log_probabilities = (log_express, log_express, log_slow)
    loss = -sum(
        old_weight * (reward - baseline) * log_probability
        for old_weight, reward, log_probability
        in zip(old_branch_probabilities, returns, log_probabilities)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = float(theta.grad)
    if not math.isclose(gradient, 0.625, abs_tol=1e-12):
        raise AssertionError("expected loss gradient is positive 0.625")
    optimizer.step()
    after_theta = float(theta.detach())
    after_p = float(torch.sigmoid(theta).detach())
    after = {
        "theta": after_theta,
        "p_express": after_p,
        "exact_objective": 4 - 2.5 * after_p,
    }

    lucky_theta = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    lucky_log_probability = torch.log(torch.sigmoid(lucky_theta))
    lucky_loss = -(3.0 - baseline) * lucky_log_probability
    lucky_loss.backward()
    lucky_loss_gradient = float(lucky_theta.grad)
    if not math.isclose(lucky_loss_gradient, -0.125, abs_tol=1e-12):
        raise AssertionError("one lucky fast delivery has opposite gradient")

    disconnected = torch.tensor(1.0, dtype=dtype)
    try:
        disconnected.backward()
    except RuntimeError as exc:
        disconnected_error = type(exc).__name__
    else:
        raise AssertionError("a detached scalar must not backpropagate")

    output = {
        "torch_version": torch.__version__,
        "dtype": "float64",
        "device": "cpu",
        "scope": "C21 two-step courier, frozen branch weights at theta=0; reward constants are not differentiated; only chosen-action log probabilities carry derivatives",
        "branch_probabilities_under_old_policy": old_branch_probabilities,
        "returns": returns,
        "baseline": baseline,
        "expected_negative_surrogate_loss": float(loss.detach()),
        "before_step": before,
        "loss_gradient_theta": gradient,
        "after_optimizer_step": after,
        "single_lucky_fast_event": {
            "first_action": "express",
            "return": 3.0,
            "sample_loss_gradient_theta": lucky_loss_gradient,
            "meaning": "a positive sampled advantage can move opposite to the exact expected gradient",
        },
        "constant_reported_loss": {
            "value": float(disconnected),
            "requires_grad": disconnected.requires_grad,
            "backward_error_type": disconnected_error,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, out)
    print("expected loss gradient:", gradient)
    print("theta before/after:", before["theta"], after_theta)
    print("J before/after:", before["exact_objective"], after["exact_objective"])
    print("one lucky fast loss gradient:", lucky_loss_gradient)
    print("constant scalar backward:", disconnected_error)


if __name__ == "__main__":
    main()
