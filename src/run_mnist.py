"""Portable reproduction of the three MNIST runs described in the reports.

Run from any directory: python src/run_mnist.py [--smoke].
Results are written separately from the supplied historical measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from mnist_components import (
    adjacency,
    graph_metrics,
    grid_edges,
    load_mnist,
    train_frozen_gnn,
)
from latent_graph import RecoveryConfig, recover_graph

PACKAGE = Path(__file__).resolve().parents[1]


def baselines(x, active_mask, edge_count, rng):
    """Pearson top-E and uniformly random E-matched edges (not degree-matched)."""
    active = np.flatnonzero(active_mask)
    z = x[:, active]
    z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    correlation = np.abs(np.corrcoef(z, rowvar=False))
    upper = np.triu_indices(len(active), 1)
    if edge_count == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)
    chosen = np.argpartition(correlation[upper], -edge_count)[-edge_count:]
    corr = np.column_stack((active[upper[0][chosen]], active[upper[1][chosen]]))
    chosen = rng.choice(len(upper[0]), edge_count, replace=False)
    random = np.column_stack((active[upper[0][chosen]], active[upper[1][chosen]]))
    return corr.astype(np.int32), random.astype(np.int32)


def diagnostics(model):
    """Log actual stage transitions; these were not saved in historical MNIST runs."""
    return {
        "configuration": asdict(model.config),
        "start_candidates": model.start_candidates_,
        "selected_start": model.selected_start_,
        "lasso_directed_proposals": model.lasso_directed_count_,
        "stage_transitions": model.stage_transitions_,
        "accepted_add_moves": model.accepted_adds_,
        "accepted_delete_moves": model.accepted_deletes_,
        "initial_pruned": model.initial_pruned_edges_,
        "final_pruned": model.post_pruned_edges_,
        "tested_pairs": model.unique_tested_pairs_,
        "screened_pairs": model.screened_pair_count_,
        "fit_seconds": model.fit_seconds_,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
        help="External MNIST IDX ZIP; dataset is not stored in Git",
    )
    parser.add_argument("--output", type=Path, default=PACKAGE / "reruns")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Small execution check, not reported research results",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, y, test, test_y = load_mnist(args.zip)
    x, test = x.astype(np.float32) / 255, test.astype(np.float32) / 255
    truth = grid_edges()
    seeds = (20260904,) if args.smoke else (20260904, 20260917, 20261003)
    penalties = (0.08,) if args.smoke else (0.08, 0.10, 0.12)
    ng, nv, nc, ncv = (200, 100, 200, 100) if args.smoke else (10000, 2000, 10000, 2000)
    if args.smoke:
        test, test_y = test[:200], test_y[:200]
    payload = {"smoke": args.smoke, "all_recovery_runs": [], "chosen_runs": []}
    for run_id, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(784)  # new index -> original pixel
        inverse = np.argsort(permutation)
        xp, tp = x[:, permutation], test[:, permutation]
        candidates = []
        for penalty in penalties:
            config = RecoveryConfig(
                ridge_alpha=0.01,
                edge_penalty=penalty,
                anneal_steps=10 if args.smoke else 2000,
                seed=42 + run_id,
            )
            a, model = recover_graph(xp[:ng], xp[ng : ng + nv], config)
            raw_loss = float(model.best_energy_ - 2 * penalty * len(model.edges_))
            candidates.append(
                {
                    "penalty": penalty,
                    "edges": len(model.edges_),
                    "unpenalized_val_loss": raw_loss,
                    "adjacency": a,
                    "model": model,
                }
            )
        best = min(c["unpenalized_val_loss"] for c in candidates)
        eligible = [
            c
            for c in candidates
            if c["unpenalized_val_loss"] <= best + 0.01 * abs(best)
        ]
        chosen = min(eligible, key=lambda c: c["edges"])
        model = chosen["model"]
        corr, random = baselines(xp[:ng], model.active_mask_, len(model.edges_), rng)
        oracle = adjacency(784, truth)[np.ix_(permutation, permutation)]
        graph_set = {
            "Oracle": oracle,
            "Recovered": chosen["adjacency"],
            "Correlation top-E": adjacency(784, corr),
            "Random edge-matched": adjacency(784, random),
            "Wrong shuffled grid": adjacency(784, truth),
        }
        classification = {}
        offset = ng + nv
        for architecture in ("GCN", "GraphSAGE"):
            metrics = train_frozen_gnn(
                xp[offset : offset + nc],
                y[offset : offset + nc],
                xp[offset + nc : offset + nc + ncv],
                y[offset + nc : offset + nc + ncv],
                tp,
                test_y,
                oracle,
                graph_set,
                100 + run_id,
                architecture=architecture,
                epochs=1 if args.smoke else 6,
                batch=512,
            )
            classification.update(
                {f"{architecture} / {name}": m for name, m in metrics.items()}
            )
        gm = graph_metrics(
            permutation[model.edges_], truth, model.active_mask_[inverse], x[:ng].var(0)
        )
        payload["chosen_runs"].append(
            {
                "permutation_seed": seed,
                "chosen_penalty": chosen["penalty"],
                "graph_metrics": gm,
                "gnn": classification,
                "diagnostics": diagnostics(model),
                "baseline_graph_metrics": {
                    name: graph_metrics(
                        permutation[e],
                        truth,
                        model.active_mask_[inverse],
                        x[:ng].var(0),
                    )
                    for name, e in (
                        ("Correlation top-E", corr),
                        ("Random edge-matched", random),
                    )
                },
            }
        )
        payload["all_recovery_runs"].append(
            {
                "permutation_seed": seed,
                "candidates": [
                    {k: v for k, v in c.items() if k not in ("adjacency", "model")}
                    for c in candidates
                ],
            }
        )
        np.savez_compressed(
            args.output / f"graph_{seed}.npz",
            permutation=permutation,
            edges=model.edges_,
            active_mask=model.active_mask_,
            correlation_edges=corr,
            random_edges=random,
        )
        (args.output / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Finished seed={seed}, edges={len(model.edges_)}, F1={gm['edge_f1']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
