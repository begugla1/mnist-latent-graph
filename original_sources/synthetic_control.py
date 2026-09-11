"""Synthetic positive control for blind graph recovery.

Observations are sampled from a Gaussian Markov random field whose precision
matrix has exactly the edges of a 28x28 four-neighbour grid.  Before recovery,
vertices are permuted; coordinates, adjacency and target edge count are used
only by this evaluation script and never passed to ``recover_graph``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_graph import RecoveryConfig, recover_graph


def grid_adjacency(side: int = 28) -> np.ndarray:
    p = side * side
    adjacency = np.zeros((p, p), dtype=np.float32)
    for row in range(side):
        for col in range(side):
            vertex = row * side + col
            if row + 1 < side:
                other = vertex + side
                adjacency[vertex, other] = adjacency[other, vertex] = 1
            if col + 1 < side:
                other = vertex + 1
                adjacency[vertex, other] = adjacency[other, vertex] = 1
    return adjacency


def sample_gmrf(
    n_samples: int,
    adjacency: np.ndarray,
    coupling: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample N(0, Q^-1), Q = I + coupling*L, without forming Q^-1."""
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    precision = np.eye(len(adjacency)) + coupling * laplacian
    chol = np.linalg.cholesky(precision)
    noise = rng.standard_normal((len(adjacency), n_samples))
    # If Q=L L^T and x=L^-T z, then Cov(x)=Q^-1.
    samples = np.linalg.solve(chol.T, noise).T
    return samples.astype(np.float32)


def edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.where(np.triu(adjacency, 1) != 0)
    return set(zip(rows.tolist(), cols.tolist()))


def graph_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    expected, recovered = edge_set(truth), edge_set(estimate)
    tp = len(expected & recovered)
    fp = len(recovered - expected)
    fn = len(expected - recovered)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "true_edges": len(expected), "recovered_edges": len(recovered),
        "true_positive_edges": tp, "false_positive_edges": fp,
        "false_negative_edges": fn, "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-15),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--coupling", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--edge-penalty", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_control.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    truth = grid_adjacency()
    samples = sample_gmrf(args.samples, truth, args.coupling, rng)
    permutation = rng.permutation(samples.shape[1])
    observed = samples[:, permutation]
    # adjacency in the anonymous/permuted feature indexing seen by recovery
    permuted_truth = truth[np.ix_(permutation, permutation)]
    cut = int(0.8 * len(observed))
    config = RecoveryConfig(
        variance_threshold=0.001,
        activation_threshold=0.01,
        activation_frequency=0.001,
        top_k=12,
        lasso_alphas=(0.006, 0.012, 0.025, 0.05),
        ridge_alpha=args.ridge_alpha,
        edge_penalty=args.edge_penalty,
        anneal_steps=args.steps,
        proposal_batch=12,
        seed=args.seed,
    )
    estimate, model = recover_graph(observed[:cut], observed[cut:], config)
    result = {
        "generator": "GMRF with precision Q = I + coupling * grid_laplacian",
        "side": 28, "vertices": 784, "samples": args.samples,
        "coupling": args.coupling, "permuted_before_recovery": True,
        "recovery_received_coordinates": False,
        "recovery_received_true_graph": False,
        **graph_metrics(permuted_truth, estimate),
        "active_vertices": int(model.active_mask_.sum()),
        "screened_pairs": model.screened_pair_count_,
        "candidate_pool_edges": model.pool_size_,
        "fit_seconds": model.fit_seconds_,
        "config": model.metadata_,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
