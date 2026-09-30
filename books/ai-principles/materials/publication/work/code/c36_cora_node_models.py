"""Same labels/features/weights, compare MLP, true-citation GCN and rewired GCN."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA = Path("work/data/cora_linqs_original/author_graph_v1")


def normalized_adjacency(edges: np.ndarray, n: int, device: torch.device) -> torch.Tensor:
    i = np.arange(n, dtype=np.int64)
    row = np.concatenate([edges[:, 0], edges[:, 1], i])
    col = np.concatenate([edges[:, 1], edges[:, 0], i])
    deg = np.bincount(row, minlength=n).astype(np.float32)
    values = 1.0 / np.sqrt(deg[row] * deg[col])
    index = torch.from_numpy(np.stack((row, col), axis=0)).long().to(device)
    return torch.sparse_coo_tensor(index, torch.from_numpy(values).to(device), (n, n)).coalesce()


class NodeNet(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.first = nn.Linear(1433, 64)
        self.second = nn.Linear(64, 7)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
        x = self.first(features)
        if adjacency is not None:
            x = torch.sparse.mm(adjacency, x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.second(x)
        if adjacency is not None:
            x = torch.sparse.mm(adjacency, x)
        return x


def load_data(mode: str, device: torch.device, include_test_labels: bool = False):
    x = torch.from_numpy(np.load(DATA / "features_uint8.npy").astype(np.float32)).to(device)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(1)
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    train = torch.tensor(split["train"], device=device)
    val = torch.tensor(split["validation"], device=device)
    if include_test_labels:
        y = torch.from_numpy(np.load(DATA / "labels_int64.npy")).long().to(device)
    else:
        visible = json.loads((DATA / "train_validation_labels_only.json").read_text(encoding="utf-8"))
        y = torch.full((len(x),), -1, dtype=torch.long, device=device)
        y[train] = torch.tensor(visible["train_labels_in_split_order"], device=device)
        y[val] = torch.tensor(visible["validation_labels_in_split_order"], device=device)
    adj = None
    if mode in ("gcn", "rewired"):
        edge_name = "undirected_unique_edges_int64.npy" if mode == "gcn" else "degree_preserving_rewired_edges_int64.npy"
        edges = np.load(DATA / edge_name)
        adj = normalized_adjacency(edges, len(x), device)
    return x, y, train, val, adj


@torch.inference_mode()
def measure(model, x, y, indices, adjacency):
    model.eval()
    logits = model(x, adjacency)[indices]
    labels = y[indices]
    return {"nodes": len(indices), "nll": float(F.cross_entropy(logits, labels)), "accuracy": float((logits.argmax(dim=-1) == labels).float().mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mlp", "gcn", "rewired"], required=True)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise RuntimeError("nonempty run dir")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, train, val, adj = load_data(a.mode, device)
    if len(train) != 140 or len(val) != 500:
        raise RuntimeError("author split changed")
    model = NodeNet(a.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0005)
    history, best = [], float("inf")
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        output = model(x, adj)
        loss = F.cross_entropy(output[train], y[train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        val_result = measure(model, x, y, val, adj)
        history.append({"epoch": epoch, "train_nll": float(loss.detach()), "validation": val_result})
        if val_result["nll"] < best:
            best = val_result["nll"]
            torch.save({"model": model.state_dict(), "mode": a.mode, "epoch": epoch, "data_manifest_sha256": hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest(), "edge_sha256": hashlib.sha256((DATA / ("degree_preserving_rewired_edges_int64.npy" if a.mode == "rewired" else "undirected_unique_edges_int64.npy")).read_bytes()).hexdigest() if a.mode != "mlp" else None}, a.out_dir / "best.pt")
        if epoch <= 5 or epoch % 25 == 0:
            print(json.dumps({"epoch": epoch, "train_nll": float(loss.detach()), "val_nll": val_result["nll"], "val_acc": val_result["accuracy"], "best": best}), flush=True)
    report = {"mode": a.mode, "seed": a.seed, "epochs": a.epochs, "node_features_visible_during_training": 2708, "node_edges_visible_during_training": a.mode != "mlp", "training_labels": len(train), "validation_labels": len(val), "training_process_label_file": "train_validation_labels_only.json; full labels_int64.npy not opened in this revised loader", "test_labels_used": False, "all_models_same_two_linear_layer_shapes": True, "parameters": sum(p.numel() for p in model.parameters()), "device": str(device), "elapsed_seconds": time.time() - t0, "selection": "minimum author-internal validation NLL, test only after all three arms chosen", "history": history}
    (a.out_dir / "train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
