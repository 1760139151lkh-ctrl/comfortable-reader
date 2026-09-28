"""One known heat equation, three distinct scientific-learning questions.

All fields here are mathematical/simulated, never telescope measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

KAPPA = 0.15


def exact_numpy(x: np.ndarray, t: np.ndarray, coefs: tuple[float, ...]) -> np.ndarray:
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(t)), dtype=np.float64)
    for n, a in enumerate(coefs, start=1):
        result = result + a * np.exp(-KAPPA * (n * np.pi) ** 2 * t) * np.sin(n * np.pi * x)
    return result


class CoordinateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                    nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((x, t), dim=1))


def residual(model: CoordinateNet, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    u = model(x, t)
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    return ut - KAPPA * uxx


def initial_boundary(device: str) -> tuple[torch.Tensor, ...]:
    x0 = torch.linspace(0, 1, 65, device=device).unsqueeze(1)
    zero = torch.zeros_like(x0)
    tb = torch.linspace(0, 1, 65, device=device).unsqueeze(1)
    xb0 = torch.zeros_like(tb)
    xb1 = torch.ones_like(tb)
    return x0, zero, tb, xb0, xb1


def train_coordinate(with_physics: bool, start: dict, device: str, steps: int) -> tuple[CoordinateNet, dict]:
    model = CoordinateNet().to(device)
    model.load_state_dict(copy.deepcopy(start))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    x0, zero, tb, xb0, xb1 = initial_boundary(device)
    target0 = torch.sin(torch.pi * x0) + 0.25 * torch.sin(2 * torch.pi * x0)
    generator = torch.Generator(device=device).manual_seed(4501)
    history = []
    start_sec = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        initial_loss = torch.mean((model(x0, zero) - target0) ** 2)
        boundary_loss = (torch.mean(model(xb0, tb) ** 2) + torch.mean(model(xb1, tb) ** 2)) / 2
        data_loss = initial_loss + boundary_loss
        if with_physics:
            xr = torch.rand((128, 1), device=device, generator=generator).requires_grad_(True)
            tr = torch.rand((128, 1), device=device, generator=generator).requires_grad_(True)
            physics_loss = torch.mean(residual(model, xr, tr) ** 2)
            loss = data_loss + physics_loss
        else:
            physics_loss = None
            loss = data_loss
        loss.backward()
        optimizer.step()
        if step in (1, 100, 500, 1000, steps):
            history.append({"step": step, "data_loss": float(data_loss.detach()),
                            "physics_loss": float(physics_loss.detach()) if physics_loss is not None else None})
    elapsed = time.perf_counter() - start_sec
    return model, {"steps": steps, "seconds": elapsed, "history": history,
                   "parameter_count": sum(p.numel() for p in model.parameters())}


def evaluate_coordinate(model: CoordinateNet, device: str) -> dict:
    model.eval()
    xs, ts = np.meshgrid(np.linspace(0, 1, 81), np.linspace(0, 1, 81))
    with torch.no_grad():
        estimate = model(torch.tensor(xs.reshape(-1, 1), dtype=torch.float32, device=device),
                         torch.tensor(ts.reshape(-1, 1), dtype=torch.float32, device=device)).cpu().numpy().reshape(xs.shape)
    truth = exact_numpy(xs, ts, (1.0, 0.25))
    probe_x = torch.tensor(np.linspace(0.025, 0.975, 60).reshape(-1, 1), device=device, dtype=torch.float32).requires_grad_(True)
    probe_t = torch.tensor(np.linspace(0.04, 0.96, 60).reshape(-1, 1), device=device, dtype=torch.float32).requires_grad_(True)
    r = residual(model, probe_x, probe_t).detach().cpu().numpy()
    return {"grid_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
            "final_t_rmse": float(np.sqrt(np.mean((estimate[-1] - truth[-1]) ** 2))),
            "residual_probe_rmse": float(np.sqrt(np.mean(r ** 2)))}


def finite_difference() -> dict:
    nx = 50; dx = 1 / nx; dt = 0.001; steps = 1000
    ratio = KAPPA * dt / dx**2
    x = np.linspace(0, 1, nx + 1)
    u = exact_numpy(x, 0.0, (1.0, 0.25))
    for _ in range(steps):
        v = u.copy()
        v[1:-1] = u[1:-1] + ratio * (u[2:] - 2*u[1:-1] + u[:-2])
        v[0] = v[-1] = 0
        u = v
    truth = exact_numpy(x, 1.0, (1.0, 0.25))
    return {"dx": dx, "dt": dt, "stability_ratio": ratio, "final_t_rmse": float(np.sqrt(np.mean((u - truth) ** 2)))}


def spectral_operator() -> dict:
    rng = np.random.default_rng(4502)
    training_coefficients = rng.uniform(-1, 1, size=(256, 3))
    heldout_coefficients = rng.uniform(-1, 1, size=(64, 3))
    t_final = 0.1
    ngrid = 64
    x = np.arange(ngrid + 1) / ngrid
    basis = np.stack([np.sin(k * np.pi * x) for k in (1, 2, 3)], axis=1)
    # This discrete sine family is exactly orthogonal on the sampled interior grid.
    input_field = training_coefficients @ basis.T
    measured_amplitudes = (2 / ngrid) * (input_field @ basis)
    true_gains = np.exp(-KAPPA * (np.arange(1, 4) * np.pi) ** 2 * t_final)
    target_field = (training_coefficients * true_gains) @ basis.T
    tensor_a = torch.tensor(measured_amplitudes, dtype=torch.float64)
    tensor_basis = torch.tensor(basis, dtype=torch.float64)
    target = torch.tensor(target_field, dtype=torch.float64)
    gains = nn.Parameter(torch.ones(3, dtype=torch.float64))
    optimizer = torch.optim.Adam([gains], lr=0.06)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        prediction = (tensor_a * gains) @ tensor_basis.T
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
    learned = gains.detach().numpy()
    def score(grid: int, coefs: np.ndarray) -> float:
        xx = np.arange(grid + 1) / grid
        bb = np.stack([np.sin(k * np.pi * xx) for k in (1, 2, 3)], axis=1)
        aa = (2 / grid) * ((coefs @ bb.T) @ bb)
        pred = (aa * learned) @ bb.T
        true = (coefs * true_gains) @ bb.T
        return float(np.sqrt(np.mean((pred - true)**2)))
    xx = np.arange(129) / 128
    fourth = np.sin(4 * np.pi * xx)
    missing_fourth_true = np.exp(-KAPPA * (4*np.pi)**2 * t_final) * fourth
    return {"task": "map initial field to its field at t=0.1 for a three-mode heat family",
            "train_examples": 256, "heldout_examples": 64,
            "learned_gains": learned.tolist(), "analytic_gains": true_gains.tolist(),
            "heldout_rmse_grid64": score(64, heldout_coefficients),
            "same_coefficients_grid128_rmse": score(128, heldout_coefficients),
            "unrepresented_mode4_grid128_rmse": float(np.sqrt(np.mean(missing_fourth_true**2))),
            "note": "A three-gain learned Fourier multiplier, not a full FNO and not a telescope fit."}


def sparse_identification() -> dict:
    rng = np.random.default_rng(4503)
    lam = KAPPA * np.pi**2
    t = np.linspace(0, 1, 101)
    starts = np.array([-1.0, -0.7, -0.4, 0.4, 0.7, 1.0])
    x = np.arange(65) / 64
    first = np.sin(np.pi * x)
    second = np.sin(2 * np.pi * x)
    full_fields = (
        starts[:, None, None] * np.exp(-lam * t[None, :, None]) * first[None, None, :]
        + 0.25 * starts[:, None, None] * np.exp(-4 * lam * t[None, :, None]) * second[None, None, :]
    )
    clean = (2 / 64) * np.sum(full_fields * first[None, None, :], axis=2)
    projection_error = float(np.max(np.abs(clean - starts[:, None] * np.exp(-lam * t[None, :]))))
    deriv = -lam * clean
    noisy = clean + rng.normal(0, 0.01, size=clean.shape)
    finite_diff = np.gradient(noisy, t, axis=1)

    def fit(a: np.ndarray, derivative: np.ndarray) -> list[float]:
        values = a.reshape(-1)
        theta = np.stack([np.ones_like(values), values, values**2], axis=1)
        coefficients = np.linalg.lstsq(theta, derivative.reshape(-1), rcond=None)[0]
        active = np.abs(coefficients) >= 0.1
        for _ in range(3):
            refined = np.zeros(3)
            refined[active] = np.linalg.lstsq(theta[:, active], derivative.reshape(-1), rcond=None)[0]
            next_active = np.abs(refined) >= 0.1
            if np.array_equal(next_active, active):
                break
            active = next_active
        return refined.tolist()
    return {"library": ["1", "a", "a^2"], "true_coefficient_on_a": float(-lam),
            "true_derivative_fit": fit(clean, deriv), "noisy_difference_fit": fit(noisy, finite_diff),
            "first_mode_projection_max_error": projection_error,
            "noise_std_in_a": 0.01,
            "raw_derivative_rmse_due_noise": float(np.sqrt(np.mean((finite_diff - deriv)**2))),
            "meaning": "First Fourier-mode amplitude of an analytically generated heat solution, not recovered from observations."}


def main(out: Path, steps: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    figure_path = out.with_suffix(".png")
    if figure_path.exists():
        raise FileExistsError(figure_path)
    torch.manual_seed(4500)
    np.random.seed(4500)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = CoordinateNet().to(device)
    initial_state = copy.deepcopy(base.state_dict())
    data_model, data_train = train_coordinate(False, initial_state, device, steps)
    physics_model, physics_train = train_coordinate(True, initial_state, device, steps)
    report = {"identity": "Author-generated exact heat-equation fields; no physical measurements",
              "equation": "u_t=0.15 u_xx on x,t in [0,1], homogeneous endpoint values",
              "single_initial_field": "sin(pi*x)+0.25*sin(2*pi*x)", "steps": steps, "device": device,
              "data_only": {"training": data_train, "evaluation": evaluate_coordinate(data_model, device)},
              "pinn": {"training": physics_train, "evaluation": evaluate_coordinate(physics_model, device)},
              "finite_difference": finite_difference(), "spectral_operator": spectral_operator(),
              "sparse_identification": sparse_identification(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True, sharey=True)
    xplot = np.linspace(0, 1, 101)
    for ax, moment in zip(axes, (0.25, 1.0)):
        x_torch = torch.tensor(xplot.reshape(-1, 1), dtype=torch.float32, device=device)
        t_torch = torch.full_like(x_torch, moment)
        with torch.no_grad():
            data_curve = data_model(x_torch, t_torch).cpu().numpy().ravel()
            physics_curve = physics_model(x_torch, t_torch).cpu().numpy().ravel()
        ax.plot(xplot, exact_numpy(xplot, moment, (1.0, 0.25)), color="black", linewidth=2, label="analytic heat solution")
        ax.plot(xplot, data_curve, color="#d27e27", linewidth=1.5, label="data only")
        ax.plot(xplot, physics_curve, color="#216b9b", linewidth=1.5, label="PINN")
        ax.set(xlabel="position x", title=f"t = {moment:g}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("temperature u")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    report["figure"] = str(figure_path)
    report["figure_sha256"] = hashlib.sha256(figure_path.read_bytes()).hexdigest()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("device", "data_only", "pinn", "finite_difference", "spectral_operator", "sparse_identification")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    main(args.out, args.steps)
