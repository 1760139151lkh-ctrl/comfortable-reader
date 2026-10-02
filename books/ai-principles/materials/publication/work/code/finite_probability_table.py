"""第五章：一张有限表上的条件概率、期望、似然和信息量。"""

import json
import math
from fractions import Fraction
from pathlib import Path


# 本书事先制作的十二条教学记录；A/B 是可见盒子，red/blue 是后来才见的颜色。
COUNTS = {
    "A": {"red": 6, "blue": 2},
    "B": {"red": 1, "blue": 3},
}
COLORS = ("red", "blue")
BOXES = ("A", "B")
PAYOFF = {"red": 2, "blue": -1}


def records_from_counts(counts):
    records = []
    for box in BOXES:
        for color in COLORS:
            records.extend([(box, color)] * counts[box][color])
    return records


def finite_table(counts):
    total = sum(counts[box][color] for box in BOXES for color in COLORS)
    box_total = {box: sum(counts[box].values()) for box in BOXES}
    color_total = {
        color: sum(counts[box][color] for box in BOXES)
        for color in COLORS
    }
    joint = {
        box: {color: Fraction(counts[box][color], total) for color in COLORS}
        for box in BOXES
    }
    p_box = {box: Fraction(box_total[box], total) for box in BOXES}
    p_color = {color: Fraction(color_total[color], total) for color in COLORS}
    conditional = {
        box: {
            color: Fraction(counts[box][color], box_total[box])
            for color in COLORS
        }
        for box in BOXES
    }
    return total, joint, p_box, p_color, conditional


def expected_payoff(probs):
    return sum(probs[color] * PAYOFF[color] for color in COLORS)


def log_loss(probability_assigned_to_actual):
    if probability_assigned_to_actual == 0:
        return math.inf
    return -math.log(float(probability_assigned_to_actual))


def mean_log_loss(records, report):
    return sum(
        log_loss(report[box][color]) for box, color in records
    ) / len(records)


def record_likelihood(records, report):
    product = 1.0
    for box, color in records:
        product *= float(report[box][color])
    return product


def entropy(probs, base=2):
    return -sum(
        float(probability) * math.log(float(probability), base)
        for probability in probs.values()
        if probability > 0
    )


def expected_log_loss(true_red, announced_red):
    return (
        -float(true_red) * math.log(float(announced_red))
        -float(1 - true_red) * math.log(float(1 - announced_red))
    )


def expected_brier_loss(true_red, announced_red):
    return (
        float(true_red) * float(1 - announced_red) ** 2
        + float(1 - true_red) * float(announced_red) ** 2
    )


def as_strings(table):
    return {
        outer: {
            inner: str(value) for inner, value in row.items()
        } for outer, row in table.items()
    }


def main():
    records = records_from_counts(COUNTS)
    total, joint, p_box, p_color, conditional = finite_table(COUNTS)
    assert total == len(records) == 12
    assert sum(joint[b][c] for b in BOXES for c in COLORS) == 1
    assert conditional["A"]["red"] == Fraction(3, 4)
    assert conditional["B"]["red"] == Fraction(1, 4)
    assert joint["A"]["red"] / p_box["A"] == conditional["A"]["red"]
    assert joint["A"]["red"] / p_color["red"] == Fraction(6, 7)
    assert (
        conditional["A"]["red"] * p_box["A"] / p_color["red"]
        == Fraction(6, 7)
    )

    by_box_payoff = {
        box: expected_payoff(conditional[box]) for box in BOXES
    }
    overall_payoff = expected_payoff(p_color)
    assert by_box_payoff == {"A": Fraction(5, 4), "B": Fraction(-1, 4)}
    assert overall_payoff == sum(
        p_box[box] * by_box_payoff[box] for box in BOXES
    ) == Fraction(3, 4)

    unconditional_report = {box: p_color for box in BOXES}
    conditional_nats = mean_log_loss(records, conditional)
    unconditional_nats = mean_log_loss(records, unconditional_report)
    conditional_likelihood = record_likelihood(records, conditional)
    unconditional_likelihood = record_likelihood(records, unconditional_report)
    assert conditional_likelihood > unconditional_likelihood
    assert abs(math.log(conditional_likelihood) + total * conditional_nats) < 1e-12
    assert abs(math.log(unconditional_likelihood) + total * unconditional_nats) < 1e-12
    marginal_bits = entropy(p_color)
    conditional_bits = sum(
        float(p_box[box]) * entropy(conditional[box]) for box in BOXES
    )
    mutual_bits = sum(
        float(joint[box][color])
        * math.log(float(conditional[box][color] / p_color[color]), 2)
        for box in BOXES for color in COLORS
    )
    assert abs(conditional_nats / math.log(2) - conditional_bits) < 1e-12
    assert abs(unconditional_nats / math.log(2) - marginal_bits) < 1e-12
    assert abs(mutual_bits - (marginal_bits - conditional_bits)) < 1e-12
    assert conditional_nats < unconditional_nats

    # 这里的 p 是一个另行假定的下一次真实分布，不把十二条经验比例当成定理。
    p = Fraction(3, 4)
    q = Fraction(3, 5)
    honest_nats = expected_log_loss(p, p)
    mismatched_nats = expected_log_loss(p, q)
    kl_nats = mismatched_nats - honest_nats
    log_ratio_red = math.log(float(p / q))
    log_ratio_blue = math.log(float((1 - p) / (1 - q)))
    assert abs(float(p) * log_ratio_red + float(1 - p) * log_ratio_blue - kl_nats) < 1e-12
    reverse_kl_nats = expected_log_loss(q, p) - expected_log_loss(q, q)
    symmetric_divergence_nats = kl_nats + reverse_kl_nats
    brier_honest = expected_brier_loss(p, p)
    brier_mismatch = expected_brier_loss(p, q)
    assert kl_nats > 0
    assert abs(brier_mismatch - brier_honest - float((q - p) ** 2)) < 1e-12

    extra_blue = {
        box: {color: COUNTS[box][color] for color in COLORS}
        for box in BOXES
    }
    extra_blue["A"]["blue"] += 1
    _, _, _, _, changed_conditional = finite_table(extra_blue)
    assert changed_conditional["A"]["red"] == Fraction(2, 3)
    assert math.isinf(log_loss(Fraction(0)))

    report = {
        "data_identity": "12 条作者制作的 A/B 盒子与红蓝结果记录；未做实体抽取",
        "assumption_for_likelihood": "各次在已知盒子后按该盒固定比例有放回、条件独立",
        "counts": COUNTS,
        "joint": as_strings(joint),
        "p_box": {key: str(value) for key, value in p_box.items()},
        "p_color": {key: str(value) for key, value in p_color.items()},
        "conditional": as_strings(conditional),
        "expected_payoff_given_box": {
            key: str(value) for key, value in by_box_payoff.items()
        },
        "overall_expected_payoff": str(overall_payoff),
        "unconditional_training_log_loss_nats": unconditional_nats,
        "conditional_training_log_loss_nats": conditional_nats,
        "unconditional_record_likelihood": unconditional_likelihood,
        "conditional_record_likelihood": conditional_likelihood,
        "marginal_empirical_entropy_bits": marginal_bits,
        "conditional_empirical_entropy_bits": conditional_bits,
        "entropy_reduction_bits_on_this_table": marginal_bits - conditional_bits,
        "same_reduction_as_weighted_kl_bits": mutual_bits,
        "hypothetical_true_red_probability": str(p),
        "announced_red_probability": str(q),
        "honest_expected_log_loss_nats": honest_nats,
        "mismatched_expected_log_loss_nats": mismatched_nats,
        "kl_gap_nats": kl_nats,
        "log_ratio_if_red_nats": log_ratio_red,
        "log_ratio_if_blue_nats": log_ratio_blue,
        "reverse_kl_nats": reverse_kl_nats,
        "symmetric_divergence_nats": symmetric_divergence_nats,
        "honest_expected_binary_square_loss": brier_honest,
        "mismatched_expected_binary_square_loss": brier_mismatch,
        "after_one_extra_A_blue_p_red_A": str(changed_conditional["A"]["red"]),
        "zero_probability_on_observed_event_log_loss": "infinite",
        "scope": "有限表与假设分布的核算；没有独立测试资料或现实盒子概率。",
    }
    output = Path(__file__).resolve().parents[1] / "results/finite_probability_table.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("条件红概率 A/B:", conditional["A"]["red"], conditional["B"]["red"])
    print("给定盒子的期望得分 A/B:", by_box_payoff["A"], by_box_payoff["B"])
    print("训练平均对数损失：不看盒子/看盒子:",
          round(unconditional_nats, 6), round(conditional_nats, 6))
    print("本表条件熵 bits:", round(conditional_bits, 6))
    print("假定真实红概率 3/4，错报 3/5 的期望 log 损失额外量:",
          round(kl_nats, 6))
    print("红/蓝单次对数比与反向 KL:",
          round(log_ratio_red, 6), round(log_ratio_blue, 6),
          round(reverse_kl_nats, 6))


if __name__ == "__main__":
    main()
