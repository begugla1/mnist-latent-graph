"""Build figures and LaTeX tables from the supplied historical measurements."""

from __future__ import annotations
import json
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "checks/mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from mnist_components import load_mnist, grid_edges, graph_metrics
from run_mnist import baselines

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def ms(values, digits=3):
    return f"${np.mean(values):.{digits}f} \\pm {np.std(values, ddof=1):.{digits}f}$"


def table(name, caption, headers, rows, align):
    text = (
        "\\begin{table}[H]\\centering\n\\small\n"
        + "\\caption{"
        + caption
        + "}\n"
        + "\\begin{tabular}{@{}"
        + align
        + "@{}}\\toprule\n"
        + " & ".join(headers)
        + " \\\\\n\\midrule\n"
        + "\n".join(" & ".join(str(x) for x in row) + " \\\\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (ROOT / "reports" / name).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, required=True, help="External MNIST IDX ZIP"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="External directory with repeated_metrics.json and pilot_graph.npz",
    )
    args = parser.parse_args()
    payload = json.loads(
        (args.results_dir / "repeated_metrics.json").read_text(encoding="utf-8")
    )
    runs = payload["chosen_runs"]
    rows = []
    for run in runs:
        g = run["graph_metrics"]
        rows.append(
            [
                run["permutation_seed"],
                g["recovered_edges"],
                f"{g['precision']:.3f}",
                f"{g['recall']:.3f}",
                f"{g['edge_f1']:.3f}",
                g["recovered_components"],
                g["recovered_largest_component"],
            ]
        )
    table(
        "structure_table.tex",
        "Восстановленная структура по каждому seed.",
        ["Seed", "$|E|$", "Precision", "Recall", "$F_1$", "Компоненты", "Макс. размер"],
        rows,
        "rrrrrrr",
    )
    methods = [("Восстановленный", [r["graph_metrics"] for r in runs])]
    methods += [
        (label, [r["baseline_graph_metrics"][name] for r in runs])
        for name, label in (
            ("Correlation top-E", "Correlation top-$E$"),
            ("Random edge-matched", "Случайный"),
        )
    ]
    table(
        "baseline_table.tex",
        "Метрики структуры: среднее и выборочное SD.",
        ["Метод", "Precision", "Recall", "$F_1$"],
        [
            [name]
            + [ms([g[k] for g in groups]) for k in ("precision", "recall", "edge_f1")]
            for name, groups in methods
        ],
        "lccc",
    )
    names = [
        ("Oracle", "Решётка"),
        ("Recovered", "Восстановленный"),
        ("Correlation top-E", "Correlation top-$E$"),
        ("Random degree-matched", "Случайный"),
        ("Wrong shuffled grid", "Неверная решётка"),
    ]
    rows = []
    for arch in ("GCN", "GraphSAGE"):
        reference = np.array([r["gnn"][f"{arch} / Oracle"]["accuracy"] for r in runs])
        for name, label in names:
            v = np.array([r["gnn"][f"{arch} / {name}"]["accuracy"] for r in runs])
            rows.append([arch, label, ms(v), ms(100 * (v - reference), 2)])
    table(
        "gnn_table.tex",
        "Test accuracy и парная разность к эталону (п.п.).",
        ["Модель", "Граф", "Accuracy", "$\\Delta$ к решётке"],
        rows,
        "llcc",
    )
    rows = []
    for penalty in (0.08, 0.10, 0.12):
        candidates = [
            c
            for r in payload["all_recovery_runs"]
            for c in r["candidates"]
            if c["penalty"] == penalty
        ]
        rows.append(
            [
                penalty,
                ms([c["edges"] for c in candidates], 1),
                ms([c["unpenalized_val_loss"] for c in candidates], 2),
            ]
        )
    table(
        "penalty_table.tex",
        "Зависимость размера графа и проверочной ошибки от штрафа.",
        ["$\\lambda$", "Рёбра", "$L_{\\rm val}$"],
        rows,
        "rcc",
    )

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, arch in zip(axs, ("GCN", "GraphSAGE")):
        vals = [
            np.array([r["gnn"][f"{arch} / {n}"]["accuracy"] for r in runs])
            for n, _ in names
        ]
        ax.bar(
            np.arange(5),
            [v.mean() for v in vals],
            yerr=[v.std(ddof=1) for v in vals],
            capsize=3,
            color=["#66788a", "#dc7925", "#2279a6", "#afb7be", "#b54b4b"],
        )
        ax.set_xticks(
            range(5),
            ["Решётка", "Восстановл.", "Корреляц.", "Случайный", "Неверный"],
            rotation=30,
            ha="right",
        )
        ax.set_title(arch)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/performance.pdf")
    plt.close(fig)

    raw, _, test, _ = load_mnist(args.zip)
    x = raw[:10000].astype(np.float32) / 255
    archive = np.load(args.results_dir / "pilot_graph.npz")
    permutation = archive["permutation"]
    inverse = np.argsort(permutation)
    active = archive["active_mask_shuffled"]
    rng = np.random.default_rng(20260904)
    assert np.array_equal(rng.permutation(784), permutation)
    rec = archive["edges_shuffled"]
    corr, random = baselines(x[:, permutation], active, len(rec), rng)
    reference = grid_edges()
    active_original = active[inverse]
    panels = [
        ("Исходное изображение", None),
        ("Эталонная решётка · 1512", reference),
        ("Восстановленный · 1046", permutation[rec]),
        ("Корреляционный · 1046", permutation[corr]),
        ("Случайный · 1046", permutation[random]),
        ("Неверная решётка · 1512", permutation[reference]),
    ]
    for name, edges in (
        ("rec", rec),
        ("Correlation top-E", corr),
        ("Random edge-matched", random),
    ):
        actual = graph_metrics(permutation[edges], reference, active_original, x.var(0))
        expected = (
            runs[0]["graph_metrics"]
            if name == "rec"
            else runs[0]["baseline_graph_metrics"][name]
        )
        for field in ("TP", "FP", "FN", "recovered_edges", "active_vertices"):
            assert actual[field] == expected[field], (
                name,
                field,
                actual[field],
                expected[field],
            )
    xy = np.array([(i % 28, 27 - i // 28) for i in range(784)])
    fig, axs = plt.subplots(3, 2, figsize=(7, 9))
    for ax, (title, edges) in zip(axs.ravel(), panels):
        if edges is None:
            ax.imshow(test[0].reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        else:
            ax.add_collection(
                LineCollection(
                    xy[edges],
                    colors="#b73c3c",
                    linewidths=0.35,
                    alpha=0.7 if len(edges) < 1100 else 0.5,
                )
            )
            ax.scatter(*xy[~active_original].T, s=3, color="#2476b0", zorder=3)
            ax.scatter(*xy[active_original].T, s=3, color="#df8625", zorder=3)
            ax.set_xlim(-1, 28)
            ax.set_ylim(-1, 28)
            ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "figures/graphs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 1.55))
    ax.axis("off")
    labels = [
        "Матрица X",
        "Активность\nи кандидаты",
        "Lasso\nи BIC",
        "Ridge-энергия\nи pruning",
        "Отжиг\nи pruning",
        "Граф G",
    ]
    for k, label in enumerate(labels):
        xx = 0.07 + k * 0.17
        ax.text(
            xx,
            0.53,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=.5", "fc": "#f0f4f7", "ec": "#6a8498"},
        )
        if k < 5:
            ax.annotate(
                "",
                xy=(xx + 0.105, 0.53),
                xytext=(xx + 0.066, 0.53),
                arrowprops={"arrowstyle": "->", "color": "#466079"},
            )
    fig.tight_layout()
    fig.savefig(ROOT / "figures/pipeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Tables and vector figures built; historical edge metrics checked.")


if __name__ == "__main__":
    main()
