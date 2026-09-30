"""C08：按已写协议比较六种组织同一二类预测的办法。"""

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prepare_iris import RAW_SHA256, SEED, assign_groups, load_raw, parse_rows


WORK = Path(__file__).resolve().parents[1]
SPLIT_FILE = WORK / "data/uci_iris_split.json"
RESULTS = WORK / "results"
CANDIDATES = ("fisher", "knn5", "tree2", "svm_linear", "svm_rbf", "forest100")


def verified_splits():
    raw = load_raw()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise ValueError("原数据字节身份不符")
    rows = parse_rows(raw)
    computed, _ = assign_groups(rows)
    saved = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if saved["iris_data_sha256"] != RAW_SHA256 or saved["seed"] != SEED:
        raise ValueError("已保存切分的原包或随机种子不符")
    for part, records in computed.items():
        actual_ids = [row["source_row"] for row in records]
        saved_ids = [row["source_row"] for row in saved["splits"][part]]
        if actual_ids != saved_ids:
            raise ValueError(f"{part} 切分与预定行号不符")
    return computed


def arrays(records):
    x = np.asarray([row["petal_cm"] for row in records], dtype=np.float64)
    y = np.asarray([row["label"] for row in records], dtype=np.int64)
    ids = np.asarray([row["source_row"] for row in records], dtype=np.int64)
    return x, y, ids


class FisherLine:
    """训练期两类均值与类内散布给出一条课堂版 Fisher 直线。"""

    def fit(self, x, y):
        neg = x[y == -1]
        pos = x[y == 1]
        if len(neg) != 30 or len(pos) != 30:
            raise ValueError("Fisher 规则应只收到各 30 株训练花")
        self.mean_neg = neg.mean(axis=0)
        self.mean_pos = pos.mean(axis=0)
        scatter = (
            (neg - self.mean_neg).T @ (neg - self.mean_neg)
            + (pos - self.mean_pos).T @ (pos - self.mean_pos)
        )
        if np.linalg.matrix_rank(scatter) != 2:
            raise ValueError("类内散布矩阵不可逆，协议未授权临时补正则化")
        self.direction = np.linalg.solve(scatter, self.mean_pos - self.mean_neg)
        self.threshold = float(self.direction @ ((self.mean_pos + self.mean_neg) / 2))
        return self

    def score(self, x):
        return x @ self.direction - self.threshold

    def predict(self, x):
        return np.where(self.score(x) > 0, 1, -1)


class FiveNearest:
    """按距离、再按原文件行号决定最近五株，不从验证段学习距离。"""

    def fit(self, x, y, row_ids):
        self.x = x.copy()
        self.y = y.copy()
        self.row_ids = row_ids.copy()
        return self

    def neighbors(self, x):
        squared = np.sum((self.x - x) ** 2, axis=1)
        order = np.lexsort((self.row_ids, squared))[:5]
        return [
            {
                "source_row": int(self.row_ids[i]),
                "label": int(self.y[i]),
                "distance": float(math.sqrt(squared[i])),
            }
            for i in order
        ]

    def predict(self, x):
        return np.asarray(
            [1 if sum(n["label"] for n in self.neighbors(row)) > 0 else -1
             for row in x],
            dtype=np.int64,
        )


def metrics(y, prediction):
    if len(y) != len(prediction):
        raise ValueError("标签与预测的行数不同")
    wrong = int(np.count_nonzero(y != prediction))
    return {
        "n": len(y),
        "wrong": wrong,
        "error_rate": wrong / len(y),
        "versicolor_wrong": int(np.count_nonzero((y == -1) & (prediction != y))),
        "virginica_wrong": int(np.count_nonzero((y == 1) & (prediction != y))),
    }


def tree_questions(model, point, mean, std):
    tree = model.tree_
    node = 0
    questions = []
    while tree.children_left[node] != tree.children_right[node]:
        feature = int(tree.feature[node])
        threshold_scaled = float(tree.threshold[node])
        threshold_cm = float(mean[feature] + std[feature] * threshold_scaled)
        answer_yes = bool(point[feature] <= threshold_scaled)
        questions.append({
            "feature": ["petal_length_cm", "petal_width_cm"][feature],
            "value_cm": float(mean[feature] + std[feature] * point[feature]),
            "threshold_cm": threshold_cm,
            "answer_value_at_most_threshold": answer_yes,
            "training_rows_at_node": int(tree.n_node_samples[node]),
        })
        node = int(tree.children_left[node] if answer_yes else tree.children_right[node])
    return questions


def draw_decision_regions(fitted, raw, mean, std, highlighted):
    """只画训练花和六条规则的决策区域；黑色 X 是一株验证花。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    all_petal = np.concatenate([raw[part][0] for part in ("train", "validation", "test")])
    x_axis = np.linspace(float(all_petal[:, 0].min() - 0.3),
                         float(all_petal[:, 0].max() + 0.3), 160)
    y_axis = np.linspace(float(all_petal[:, 1].min() - 0.2),
                         float(all_petal[:, 1].max() + 0.2), 120)
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid_cm = np.column_stack((xx.ravel(), yy.ravel()))
    grid_scaled = (grid_cm - mean) / std
    titles = {
        "fisher": "Fisher 判别线",
        "knn5": "5 近邻",
        "tree2": "深度 2 的树",
        "svm_linear": "线性 SVM",
        "svm_rbf": "高斯核 SVM",
        "forest100": "100 棵树的森林",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    train_cm, train_y, _ = raw["train"]
    for ax, name in zip(axes.flat, CANDIDATES):
        prediction = fitted[name].predict(grid_scaled).reshape(xx.shape)
        ax.contourf(xx, yy, prediction, levels=[-1.5, 0, 1.5],
                    colors=["#dceaf6", "#f9e5d0"], alpha=0.95)
        ax.contour(xx, yy, prediction, levels=[0], colors=["#30343b"], linewidths=0.8)
        ax.scatter(train_cm[train_y == -1, 0], train_cm[train_y == -1, 1],
                   s=25, c="#24527a", edgecolors="white", linewidths=0.5,
                   label="变色鸢尾 versicolor")
        ax.scatter(train_cm[train_y == 1, 0], train_cm[train_y == 1, 1],
                   s=25, c="#a54d13", edgecolors="white", linewidths=0.5,
                   label="维吉尼亚鸢尾 virginica")
        if highlighted is not None:
            ax.scatter([highlighted["petal_cm"][0]], [highlighted["petal_cm"][1]],
                       marker="X", s=95, c="black", edgecolors="white",
                       linewidths=0.5, label="验证花，第 134 行")
        ax.set_title(titles[name], fontsize=11)
        ax.grid(alpha=0.15)
    for ax in axes[1]:
        ax.set_xlabel("花瓣长度（厘米）")
    for ax in axes[:, 0]:
        ax.set_ylabel("花瓣宽度（厘米）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("同一批训练花，六种组织判断的办法", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output = WORK / "figures/iris_six_rules.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    splits = verified_splits()
    raw = {part: arrays(rows) for part, rows in splits.items()}
    x_train, y_train, train_ids = raw["train"]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    if np.any(std == 0):
        raise ValueError("训练段有零标准差的测量")
    scaled = {part: ((x - mean) / std, y, ids) for part, (x, y, ids) in raw.items()}
    train_x, train_y, train_ids = scaled["train"]

    fitted = {
        "fisher": FisherLine().fit(train_x, train_y),
        "knn5": FiveNearest().fit(train_x, train_y, train_ids),
        "tree2": DecisionTreeClassifier(
            criterion="gini", max_depth=2, random_state=SEED
        ).fit(train_x, train_y),
        "svm_linear": SVC(
            kernel="linear", C=1.0, random_state=SEED
        ).fit(train_x, train_y),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma=1.0, random_state=SEED
        ).fit(train_x, train_y),
        "forest100": RandomForestClassifier(
            n_estimators=100, criterion="gini", bootstrap=True,
            max_depth=2, max_features=1, random_state=SEED, n_jobs=1
        ).fit(train_x, train_y),
    }

    # 到此仅评分训练与验证；测试资料未交给拟合器或选择器。
    scores = {}
    for part in ("train", "validation"):
        x, y, _ = scaled[part]
        scores[part] = {
            name: metrics(y, fitted[name].predict(x))
            for name in CANDIDATES
        }
    selected = min(CANDIDATES, key=lambda name: (
        scores["validation"][name]["wrong"], CANDIDATES.index(name)
    ))

    test_x, test_y, test_ids = scaled["test"]
    test_predictions = {name: fitted[name].predict(test_x) for name in CANDIDATES}
    scores["test"] = {
        name: metrics(test_y, test_predictions[name])
        for name in CANDIDATES
    }

    val_x, val_y, val_ids = scaled["validation"]
    first = 0
    validation_predictions = {
        name: fitted[name].predict(val_x) for name in CANDIDATES
    }
    disagreement_indices = [
        i for i in range(len(val_y))
        if len({int(validation_predictions[name][i]) for name in CANDIDATES}) > 1
    ]
    first_disagreement = None
    if disagreement_indices:
        i = disagreement_indices[0]
        first_disagreement = {
            "source_row": int(val_ids[i]),
            "actual_label": int(val_y[i]),
            "petal_cm": raw["validation"][0][i].tolist(),
            "petal_scaled": val_x[i].tolist(),
            "predictions": {
                name: int(validation_predictions[name][i])
                for name in CANDIDATES
            },
            "fisher_score": float(fitted["fisher"].score(val_x[i:i+1])[0]),
            "five_neighbors": fitted["knn5"].neighbors(val_x[i]),
            "tree_questions": tree_questions(fitted["tree2"], val_x[i], mean, std),
            "svm_decision_values": {
                name: float(fitted[name].decision_function(val_x[i:i+1])[0])
                for name in ("svm_linear", "svm_rbf")
            },
            "forest_positive_votes": int(sum(
                int(tree.predict(val_x[i:i+1])[0] == 1)
                for tree in fitted["forest100"].estimators_
            )),
        }
    root_feature = int(fitted["tree2"].tree_.feature[0])
    root_threshold_scaled = float(fitted["tree2"].tree_.threshold[0])
    result = {
        "protocol": "work/verification/C08_iris_protocol.md",
        "data_source": "work/data/uci_iris_SOURCE.md",
        "iris_data_sha256": RAW_SHA256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "feature_names": ["petal_length_cm", "petal_width_cm"],
        "source_row_ids": {
            part: ids.tolist() for part, (_, _, ids) in raw.items()
        },
        "train_only_scaling": {"mean_cm": mean.tolist(), "std_cm": std.tolist()},
        "candidate_order_predeclared": CANDIDATES,
        "selected_on_validation": selected,
        "scores": scores,
        "fisher": {
            "class_means_scaled": {
                "versicolor": fitted["fisher"].mean_neg.tolist(),
                "virginica": fitted["fisher"].mean_pos.tolist(),
            },
            "direction_scaled": fitted["fisher"].direction.tolist(),
            "threshold_scaled": fitted["fisher"].threshold,
        },
        "tree_first_question": {
            "feature": ["petal_length_cm", "petal_width_cm"][root_feature],
            "threshold_scaled": root_threshold_scaled,
            "threshold_cm": float(mean[root_feature] + std[root_feature] * root_threshold_scaled),
        },
        "support_vector_count": {
            name: fitted[name].n_support_.tolist()
            for name in ("svm_linear", "svm_rbf")
        },
        "first_validation_row": {
            "source_row": int(val_ids[first]),
            "actual_label": int(val_y[first]),
            "petal_cm": raw["validation"][0][first].tolist(),
            "petal_scaled": val_x[first].tolist(),
            "predictions": {
                name: int(fitted[name].predict(val_x[first:first+1])[0])
                for name in CANDIDATES
            },
            "fisher_score": float(fitted["fisher"].score(val_x[first:first+1])[0]),
            "five_neighbors": fitted["knn5"].neighbors(val_x[first]),
            "forest_positive_votes": int(sum(
                int(tree.predict(val_x[first:first+1])[0] == 1)
                for tree in fitted["forest100"].estimators_
            )),
        },
        "first_validation_disagreement": first_disagreement,
        "decision_regions_figure": "work/figures/iris_six_rules.png",
        "scope": "UCI 1988 public archive, 100 rows of two species, two petal measurements; one predetermined 60/20/20 grouped stratified split. Teaching comparison of rule organization, not new ecological validation.",
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "iris_method_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "iris_test_rows.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_row", "petal_length_cm", "petal_width_cm", "actual_label",
                         *[f"prediction_{name}" for name in CANDIDATES]])
        for index, source_id in enumerate(test_ids):
            writer.writerow([
                int(source_id),
                *raw["test"][0][index].tolist(),
                int(test_y[index]),
                *[int(test_predictions[name][index]) for name in CANDIDATES],
            ])
    draw_decision_regions(fitted, raw, mean, std, first_disagreement)
    print("验证段选中:", selected)
    for name in CANDIDATES:
        print(name, "训练/验证/测试错数:",
              *[scores[part][name]["wrong"] for part in ("train", "validation", "test")])
    print("第一条验证记录:", result["first_validation_row"])
    print("首条规则有分歧的验证记录:", first_disagreement)


if __name__ == "__main__":
    main()
