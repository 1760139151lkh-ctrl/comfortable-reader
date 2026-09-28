"""逐项展开 RNN、现代 LSTM 与 PyTorch GRU 的一个时间步。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WORK = Path(__file__).resolve().parents[1]
RESULT = WORK / "results" / "recurrent_gate_probe.json"


def explicit_rnn(cell: nn.RNNCell, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
    new_h = torch.tanh(
        F.linear(x, cell.weight_ih, cell.bias_ih)
        + F.linear(h, cell.weight_hh, cell.bias_hh)
    )
    return new_h, {}


def explicit_lstm(
    cell: nn.LSTMCell, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    input_part = F.linear(x, cell.weight_ih, cell.bias_ih)
    history_part = F.linear(h, cell.weight_hh, cell.bias_hh)
    i_raw, f_raw, g_raw, o_raw = (input_part + history_part).chunk(4, dim=1)
    i, f, g, o = torch.sigmoid(i_raw), torch.sigmoid(f_raw), torch.tanh(g_raw), torch.sigmoid(o_raw)
    new_c = f * c + i * g
    new_h = o * torch.tanh(new_c)
    return new_h, new_c, {"input": i, "forget": f, "candidate": g, "output": o}


def explicit_gru(cell: nn.GRUCell, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, dict]:
    i_r, i_z, i_n = F.linear(x, cell.weight_ih, cell.bias_ih).chunk(3, dim=1)
    h_r, h_z, h_n = F.linear(h, cell.weight_hh, cell.bias_hh).chunk(3, dim=1)
    r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)  # PyTorch: reset applied after the hidden linear map
    new_h = (1 - z) * n + z * h
    return new_h, {"reset": r, "keep": z, "candidate": n}


def main() -> None:
    torch.manual_seed(20260924)
    x = torch.tensor([[0.8, -0.5]], dtype=torch.float64)
    h = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    c = torch.tensor([[0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    rnn = nn.RNNCell(2, 2, nonlinearity="tanh").double()
    lstm = nn.LSTMCell(2, 2).double()
    gru = nn.GRUCell(2, 2).double()

    rnn_manual, _ = explicit_rnn(rnn, x, h)
    rnn_builtin = rnn(x, h)
    lstm_manual_h, lstm_manual_c, lstm_gates = explicit_lstm(lstm, x, h, c)
    lstm_builtin_h, lstm_builtin_c = lstm(x, (h, c))
    gru_manual, gru_gates = explicit_gru(gru, x, h)
    gru_builtin = gru(x, h)

    errors = {
        "rnn": float((rnn_manual - rnn_builtin).detach().abs().max()),
        "lstm_h": float((lstm_manual_h - lstm_builtin_h).detach().abs().max()),
        "lstm_c": float((lstm_manual_c - lstm_builtin_c).detach().abs().max()),
        "gru": float((gru_manual - gru_builtin).detach().abs().max()),
    }
    if max(errors.values()) > 1e-12:
        raise AssertionError(f"显式时间步与 PyTorch 模块不一致：{errors}")
    carry_gradient = torch.autograd.grad(lstm_manual_c.sum(), c)[0]
    if not torch.allclose(carry_gradient, lstm_gates["forget"], atol=1e-12):
        raise AssertionError("固定 h 前提下 c_t 对 c_{t-1} 的直接导数不等于 f_t")

    # 非交换的两种重置顺序：原始论文把 r 乘到 h 上，PyTorch 在 W_h h 后乘 r。
    swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    reset = torch.tensor([0.8, 0.2], dtype=torch.float64)
    prior = torch.tensor([1.0, 1.0], dtype=torch.float64)
    original_order = swap @ (reset * prior)
    pytorch_order = reset * (swap @ prior)
    if torch.equal(original_order, pytorch_order):
        raise AssertionError("展示两种 GRU 重置顺序的例子不再区分它们")

    result = {
        "purpose": "one-step math/code identity; not a natural-language model or trained memory result",
        "input": x.tolist(),
        "previous_hidden": h.tolist(),
        "previous_lstm_cell": c.detach().tolist(),
        "max_abs_error_against_pytorch": errors,
        "lstm_gates": {name: tensor.detach().tolist() for name, tensor in lstm_gates.items()},
        "lstm_new_cell": lstm_manual_c.detach().tolist(),
        "lstm_new_hidden": lstm_manual_h.detach().tolist(),
        "lstm_direct_cell_carry_gradient_with_previous_h_fixed": carry_gradient.detach().tolist(),
        "gru_gates": {name: tensor.detach().tolist() for name, tensor in gru_gates.items()},
        "gru_new_hidden": gru_manual.detach().tolist(),
        "original_gru_reset_before_matrix_example": original_order.tolist(),
        "pytorch_gru_reset_after_matrix_example": pytorch_order.tolist(),
        "torch_version": torch.__version__,
    }
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"max_abs_error_against_pytorch": errors,
                      "original_order": original_order.tolist(), "pytorch_order": pytorch_order.tolist()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
