"""One reproducible classroom calculation; Python standard library only.

Reads one selected CSV and writes only the explicit output path, if provided.
Running this file is separate from the browser's built-in calculation.
"""
from pathlib import Path
import argparse
import csv
import json
import math


def load_rows(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["x", "y"]:
            raise ValueError("CSV needs exactly the columns x,y")
        rows = [(float(row["x"]), float(row["y"])) for row in reader]
    if not 2 <= len(rows) <= 10000 or any(not math.isfinite(x) or not math.isfinite(y) for x, y in rows):
        raise ValueError("need 2–10000 finite numeric records")
    return rows


def fit(rows: list[tuple[float, float]], rate: float, steps: int) -> dict:
    if not math.isfinite(rate) or not 0 < rate <= 0.3 or not 1 <= steps <= 200:
        raise ValueError("rate must be (0,0.3] and steps must be 1–200")
    w = b = 0.0
    trace = []
    for step in range(1, steps + 1):
        residuals = [(w * x + b) - y for x, y in rows]
        loss = sum(e * e for e in residuals) / len(rows)
        grad_w = 2 * sum(e * x for e, (x, _) in zip(residuals, rows)) / len(rows)
        grad_b = 2 * sum(residuals) / len(rows)
        before = {"w": w, "b": b}
        w -= rate * grad_w
        b -= rate * grad_b
        if not all(map(math.isfinite, (w, b, loss))):
            raise ArithmeticError("nonfinite update")
        trace.append({"step": step, "before": before, "loss_before": loss,
                      "gradient": {"w": grad_w, "b": grad_b}, "after": {"w": w, "b": b}})
    return {"identity": "local_python_run", "rate": rate, "steps": steps,
            "records": len(rows), "weights": {"w": w, "b": b},
            "predictions": [{"x": x, "target": y, "prediction": w * x + b} for x, y in rows],
            "trace": trace}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit y = w*x + b by batch gradient descent")
    parser.add_argument("--data", required=True, type=Path, help="CSV with x,y columns")
    parser.add_argument("--rate", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, help="optional path for this run's JSON")
    args = parser.parse_args()
    report = fit(load_rows(args.data), args.rate, args.steps)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error("output already exists; choose a new path to preserve the old result")
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
