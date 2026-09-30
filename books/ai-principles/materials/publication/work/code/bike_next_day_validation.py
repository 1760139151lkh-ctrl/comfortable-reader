"""C07：固定时间切分，比较三条一天提前预测规则。"""

import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np


WORK = Path(__file__).resolve().parents[1]
DATA = WORK / "data/uci_bike_sharing/day.csv"
EXPECTED_SHA256 = "a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be"
VALIDATION_END = date(2012, 6, 30)
TEST_START = date(2012, 7, 1)
CANDIDATE_ORDER = ("yesterday", "lag_linear", "lag_weekend")


def read_days():
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("day.csv 不等于已核对的 UCI 原始字节")
    with DATA.open("r", encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    if len(source_rows) != 731:
        raise ValueError("日记录行数不等于 731")
    days = []
    for index, row in enumerate(source_rows):
        day = date.fromisoformat(row["dteday"])
        if day != date(2011, 1, 1) + timedelta(days=index):
            raise ValueError(f"日期不连续或次序有变：{row['dteday']}")
        if int(row["instant"]) != index + 1:
            raise ValueError("原始记录序号有变")
        total = int(row["cnt"])
        if total != int(row["casual"]) + int(row["registered"]) or total < 0:
            raise ValueError(f"每日总次数不等于两组成分之和：{day}")
        days.append({"day": day, "count": total})
    if days[-1]["day"] != date(2012, 12, 31):
        raise ValueError("最后日期有变")
    return days


def forecasting_examples(days):
    examples = []
    for yesterday, today in zip(days[:-1], days[1:]):
        assert today["day"] - yesterday["day"] == timedelta(days=1)
        target_day = today["day"]
        examples.append({
            "target_date": target_day,
            "lag_date": yesterday["day"],
            "lag_count": yesterday["count"],
            "weekend": int(target_day.weekday() >= 5),
            "target": today["count"],
        })
    return examples


def split(examples):
    train = [row for row in examples if row["target_date"].year == 2011]
    valid = [
        row for row in examples
        if date(2012, 1, 1) <= row["target_date"] <= VALIDATION_END
    ]
    test = [row for row in examples if row["target_date"] >= TEST_START]
    if len(train) + len(valid) + len(test) != len(examples):
        raise ValueError("某些目标日期未分组")
    assert (len(train), len(valid), len(test)) == (364, 182, 184)
    return train, valid, test


def design(rows, name):
    lag = np.array([row["lag_count"] for row in rows], dtype=np.float64)
    ones = np.ones(len(rows), dtype=np.float64)
    if name == "lag_linear":
        return np.column_stack((ones, lag))
    if name == "lag_weekend":
        weekend = np.array([row["weekend"] for row in rows], dtype=np.float64)
        return np.column_stack((ones, lag, weekend))
    raise ValueError(name)


def fit_candidates(train):
    candidates = {"yesterday": {"kind": "fixed", "parameter_count": 0}}
    target = np.array([row["target"] for row in train], dtype=np.float64)
    for name in ("lag_linear", "lag_weekend"):
        matrix = design(train, name)
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        if rank != matrix.shape[1]:
            raise ValueError(f"{name} 训练矩阵不满列秩")
        candidates[name] = {
            "kind": "least_squares_on_2011_only",
            "parameter_count": matrix.shape[1],
            "coefficients": coefficients.tolist(),
        }
    return candidates


def predictions(rows, name, fitted):
    if name == "yesterday":
        return np.array([row["lag_count"] for row in rows], dtype=np.float64)
    return design(rows, name) @ np.array(fitted[name]["coefficients"], dtype=np.float64)


def metrics(rows, prediction):
    actual = np.array([row["target"] for row in rows], dtype=np.float64)
    error = prediction - actual
    return {
        "n": len(rows),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mean_signed_error": float(error.mean()),
        "target_mean": float(actual.mean()),
    }


def row_for_json(row):
    return {
        "target_date": row["target_date"].isoformat(),
        "lag_date": row["lag_date"].isoformat(),
        "lag_count": row["lag_count"],
        "weekend": row["weekend"],
        "target": row["target"],
    }


def main():
    days = read_days()
    train, valid, test = split(forecasting_examples(days))
    fitted = fit_candidates(train)  # 此函数的唯一输入是 2011 训练对
    by_split = {}
    for label, rows in (("train", train), ("validation", valid)):
        by_split[label] = {
            name: metrics(rows, predictions(rows, name, fitted))
            for name in CANDIDATE_ORDER
        }

    # 选择发生在测试段任何分数计算之前。相等时保留上列较简单者。
    selected = min(
        CANDIDATE_ORDER,
        key=lambda name: (
            by_split["validation"][name]["mae"],
            fitted[name]["parameter_count"],
        ),
    )
    test_predictions = {
        name: predictions(test, name, fitted) for name in CANDIDATE_ORDER
    }
    by_split["test"] = {
        name: metrics(test, test_predictions[name]) for name in CANDIDATE_ORDER
    }
    output_dir = WORK / "results"
    output_dir.mkdir(exist_ok=True)
    prediction_file = output_dir / "bike_next_day_test_rows.csv"
    with prediction_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "target_date", "lag_date", "lag_count", "weekend", "actual",
            *[f"prediction_{name}" for name in CANDIDATE_ORDER],
        ])
        for index, row in enumerate(test):
            writer.writerow([
                row["target_date"].isoformat(),
                row["lag_date"].isoformat(),
                row["lag_count"],
                row["weekend"],
                row["target"],
                *[float(test_predictions[name][index]) for name in CANDIDATE_ORDER],
            ])

    report = {
        "dataset_identity": "UCI Bike Sharing day.csv, SHA-256 " + EXPECTED_SHA256,
        "source_and_license": "work/data/uci_bike_sharing/SOURCE.md",
        "protocol": "work/verification/C07_bike_protocol.md",
        "forecast_origin": "目标日前一日结束，上一日实际 cnt 已知；目标日是否周末已知",
        "excluded_same_day_fields": [
            "weathersit", "temp", "atemp", "hum", "windspeed",
            "casual", "registered", "cnt",
        ],
        "raw_days": len(days),
        "target_examples": len(train) + len(valid) + len(test),
        "ranges": {
            "train": [train[0]["target_date"].isoformat(), train[-1]["target_date"].isoformat()],
            "validation": [valid[0]["target_date"].isoformat(), valid[-1]["target_date"].isoformat()],
            "test": [test[0]["target_date"].isoformat(), test[-1]["target_date"].isoformat()],
        },
        "split_counts": {"train": len(train), "validation": len(valid), "test": len(test)},
        "candidate_order_predeclared": CANDIDATE_ORDER,
        "fitted_from_training_only": fitted,
        "selection_metric": "validation MAE in rental counts per day, tie goes to fewer parameters",
        "selected_on_validation": selected,
        "metrics": by_split,
        "first_validation_row": row_for_json(valid[0]),
        "first_test_row": {
            **row_for_json(test[0]),
            "predictions": {
                name: float(test_predictions[name][0]) for name in CANDIDATE_ORDER
            },
        },
        "test_predictions_csv": "work/results/bike_next_day_test_rows.csv",
        "scope": "One city, 2011 fit; 2012 H1 selection; 2012 H2 once-reported test. Rolling one-day forecasts with actual prior-day count; not a closed-loop six-month forecast or universal generalization claim.",
    }
    result_file = output_dir / "bike_next_day_validation.json"
    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("目标日期条数 train/validation/test:", len(train), len(valid), len(test))
    print("验证段按 MAE 选择:", selected)
    for name in CANDIDATE_ORDER:
        print(name, "train/val/test MAE:",
              *[round(by_split[split_name][name]["mae"], 2)
                for split_name in ("train", "validation", "test")])
    print("首个测试日:", report["first_test_row"])


if __name__ == "__main__":
    main()
