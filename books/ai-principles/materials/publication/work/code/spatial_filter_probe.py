"""小图逐格乘加、位移与共享权重梯度的可检查探针。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / "results" / "spatial_filter_probe.json"
FIGURE = WORK / "figures" / "spatial_filter_probe.png"


def manual_correlation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape
    kh, kw = kernel.shape
    output = np.zeros((height - kh + 1, width - kw + 1), dtype=np.float64)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            patch = image[row:row + kh, col:col + kw]
            output[row, col] = float(np.sum(patch * kernel))
    return output


def shift_right_down(image: np.ndarray) -> np.ndarray:
    shifted = np.zeros_like(image)
    shifted[1:, 1:] = image[:-1, :-1]
    return shifted


def main() -> None:
    image = np.zeros((7, 7), dtype=np.float64)
    image[1:5, 2] = 1.0
    image[4, 2:5] = 1.0
    image[2, 3] = 0.5
    kernel = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    shifted = shift_right_down(image)
    original_map = manual_correlation(image, kernel)
    shifted_map = manual_correlation(shifted, kernel)

    input_tensor = torch.tensor(image, dtype=torch.float64).view(1, 1, 7, 7)
    weight = torch.tensor(kernel, dtype=torch.float64).view(1, 1, 3, 3).requires_grad_()
    torch_map = F.conv2d(input_tensor, weight).detach().numpy()[0, 0]
    if not np.array_equal(original_map, torch_map):
        raise AssertionError("手写互相关与 PyTorch Conv2d 不一致")
    if not np.array_equal(shifted_map[1:, 1:], original_map[:-1, :-1]):
        raise AssertionError("有效内域的位移关系不成立")
    response = F.conv2d(input_tensor, weight)
    (response[0, 0, 1, 1] + response[0, 0, 2, 2]).backward()
    actual_grad = weight.grad.detach().numpy()[0, 0]
    expected_grad = image[1:4, 1:4] + image[2:5, 2:5]
    if not np.array_equal(actual_grad, expected_grad):
        raise AssertionError("共享核的梯度没有等于两处局部贡献之和")

    figure, axes = plt.subplots(1, 4, figsize=(10.8, 3.1), layout="constrained")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for ax, value, title in zip(
        axes,
        [image, kernel, original_map, shifted_map],
        ["7×7 输入", "共享 3×3 核", "原响应", "位移后响应"],
    ):
        ax.imshow(value, cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(value.shape[1]))
        ax.set_yticks(range(value.shape[0]))
        ax.grid(color="#888888", alpha=0.45)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    result = {
        "purpose": "one stride-1 valid cross-correlation; no classifier training",
        "input": image.tolist(),
        "kernel": kernel.tolist(),
        "manual_and_torch_response": original_map.tolist(),
        "shifted_input": shifted.tolist(),
        "shifted_response": shifted_map.tolist(),
        "overlap_identity": "shifted_response[1:,1:] == original_response[:-1,:-1]",
        "overlap_max_abs_error": float(np.max(np.abs(shifted_map[1:, 1:] - original_map[:-1, :-1]))),
        "gradient_from_two_positions": actual_grad.tolist(),
        "gradient_expected_patch_sum": expected_grad.tolist(),
        "free_parameters_shared_with_bias": int(kernel.size + 1),
        "free_parameters_if_every_output_position_had_own_kernel_and_bias": int(original_map.size * (kernel.size + 1)),
        "torch_version": torch.__version__,
    }
    RESULT.parent.mkdir(exist_ok=True)
    FIGURE.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("overlap_max_abs_error", "free_parameters_shared_with_bias", "free_parameters_if_every_output_position_had_own_kernel_and_bias")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
