"""Unsupervised latent graph recovery from a sample-by-feature matrix.

The estimator never receives coordinates, labels, a reference adjacency, or a
target edge count.  Such information belongs exclusively to the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
import time
import numpy as np
from sklearn.linear_model import Lasso


@dataclass
class RecoveryConfig:
    variance_threshold: float = 0.02
    activation_threshold: float = 0.05
    activation_frequency: float = 0.01
    blocks: int = 4
    top_k: int = 24
    lasso_alphas: tuple[float, ...] = (0.003, 0.006, 0.012, 0.025, 0.05)
    ridge_alpha: float = 0.01
    edge_penalty: float | None = 0.08
    anneal_steps: int = 2000
    proposal_batch: int = 12
    global_add_probability: float = 0.10
    history_beta: float = 0.9
    seed: int = 42


class LatentGraphRecovery:
    """Recover a sparse undirected predictive-dependency graph."""
    def __init__(self, config: RecoveryConfig | None = None):
        self.config = config or RecoveryConfig()

    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None):
        t0 = time.perf_counter()
        cfg, rng = self.config, np.random.default_rng(self.config.seed)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must have shape [samples, vertices]")
        if X_val is None:
            order = rng.permutation(len(X)); cut = max(2, int(.8 * len(X)))
            X, X_val = X[order[:cut]], X[order[cut:]]
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
        self.n_vertices_ = X.shape[1]
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0)
        freq = (np.abs(X) > cfg.activation_threshold).mean(0)
        self.active_mask_ = ((self.scale_ > cfg.variance_threshold) &
                             (freq > cfg.activation_frequency))
        self.active_ = np.flatnonzero(self.active_mask_)
        if len(self.active_) < 2:
            raise ValueError("Fewer than two informative vertices")
        safe_scale = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        Z = ((X - self.mean_) / safe_scale)[:, self.active_]
        Zv = ((X_val - self.mean_) / safe_scale)[:, self.active_]
        p, n = Z.shape[1], Z.shape[0]

        # This is the sole exhaustive pass over all active pairs.
        full_corr = np.corrcoef(Z, rowvar=False).astype(np.float32)
        block_corr = []
        for ids in np.array_split(rng.permutation(n), cfg.blocks):
            block_corr.append(np.corrcoef(Z[ids], rowvar=False))
        bc = np.stack(block_corr)
        # Sparse graph signals can make a vertex constant inside an individual
        # block even when it is informative globally.  Such undefined block
        # correlations must be treated as absent evidence, not propagated as
        # NaNs into candidate ranking.
        full_corr = np.nan_to_num(full_corr, nan=0.0, posinf=0.0, neginf=0.0)
        bc = np.nan_to_num(bc, nan=0.0, posinf=0.0, neginf=0.0)
        sign_stability = np.abs(np.mean(np.sign(bc), axis=0))
        magnitude_std = np.std(np.abs(bc), axis=0)
        score = np.abs(full_corr) * sign_stability / (1.0 + magnitude_std)
        np.fill_diagonal(score, -np.inf)
        k = min(cfg.top_k, p - 1)
        candidates = [np.argpartition(score[i], -k)[-k:] for i in range(p)]
        pool = {tuple(sorted((i, int(j)))) for i in range(p) for j in candidates[i] if i != j}
        self.screened_pair_count_ = p * (p - 1) // 2
        self.pool_size_ = len(pool)

        # BIC-selected local Lasso supports form several possible warm starts.
        directed = set()
        for i, ci in enumerate(candidates):
            y, A = Z[:, i], Z[:, ci]
            best = (np.inf, None)
            for alpha in cfg.lasso_alphas:
                model = Lasso(alpha=alpha, fit_intercept=False, max_iter=3000,
                              selection="cyclic").fit(A, y)
                rss = max(float(np.sum((y - model.predict(A)) ** 2)), 1e-12)
                nz = int(np.count_nonzero(np.abs(model.coef_) > 1e-8))
                bic = n * np.log(rss / n) + nz * np.log(n)
                if bic < best[0]: best = (bic, model.coef_.copy())
            for j, coef in zip(ci, best[1]):
                if abs(coef) > 1e-8: directed.add((i, int(j)))
        union = {tuple(sorted(e)) for e in directed}
        mutual = {tuple(sorted((i, j))) for i, j in directed if (j, i) in directed}
        self.lasso_directed_count_ = len(directed)
        self.lasso_union_count_ = len(union)
        self.lasso_mutual_count_ = len(mutual)

        # Gram matrices make exact endpoint Ridge rescoring inexpensive.
        self._G = (Z.T @ Z) / len(Z)
        self._C = (Zv.T @ Zv) / len(Zv)
        penalty = cfg.edge_penalty
        if penalty is None:
            penalty = (np.log(max(n, 2)) + np.log(max(p, 2))) / len(Zv)
        self.edge_penalty_ = float(penalty)

        def local_score(i, neigh):
            if not neigh: return float(np.log(max(self._C[i, i], 1e-12)))
            q = np.fromiter(sorted(neigh), dtype=int)
            M = self._G[np.ix_(q, q)] + cfg.ridge_alpha * np.eye(len(q))
            w = np.linalg.solve(M, self._G[q, i])
            mse = self._C[i, i] - 2 * w @ self._C[q, i] + w @ self._C[np.ix_(q, q)] @ w
            return float(np.log(max(mse, 1e-12)) + penalty * len(q))

        def neighbors(edges):
            out = [set() for _ in range(p)]
            for a, b in edges: out[a].add(b); out[b].add(a)
            return out

        def total(edges):
            ns = neighbors(edges)
            return sum(local_score(i, ns[i]) for i in range(p))

        start_names = ["empty", "lasso_union", "lasso_mutual"]
        starts = [set(), union, mutual]
        energies = [total(e) for e in starts]
        selected_start = int(np.argmin(energies))
        edges = set(starts[selected_start]); ns = neighbors(edges)
        selected_start_edges = set(edges)
        self.start_candidates_ = [
            {"name": name, "edges": len(edge_set), "energy": float(energy)}
            for name, edge_set, energy in zip(start_names, starts, energies)
        ]
        self.selected_start_ = start_names[selected_start]
        self.selected_start_edge_count_ = len(edges)
        current = min(energies); best_energy = current; best_edges = set(edges)

        all_pairs = None
        history = {}; unique_tested = set(); accepted = 0; pruned = 0
        accepted_adds = 0; accepted_deletes = 0
        # Calibrate T from a small set of exact toggles.
        def delta(edge):
            a, b = edge
            old = local_score(a, ns[a]) + local_score(b, ns[b])
            na, nb = set(ns[a]), set(ns[b])
            if edge in edges: na.remove(b); nb.remove(a)
            else: na.add(b); nb.add(a)
            return local_score(a, na) + local_score(b, nb) - old

        # A dense Lasso union cannot be cleaned by a limited number of random
        # DELETE proposals.  Deterministic coordinate pruning removes every
        # edge whose exact current validation contribution is harmful.
        initial_pruned = 0
        for _ in range(6):
            removed_this_pass = 0
            for e in sorted(list(edges)):
                d = delta(e); unique_tested.add(e)
                if d < -1e-10:
                    a,b=e; edges.remove(e); ns[a].remove(b); ns[b].remove(a)
                    current += d; pruned += 1; initial_pruned += 1; removed_this_pass += 1
                    if current < best_energy: best_energy,best_edges=current,set(edges)
            if not removed_this_pass: break
        after_initial_pruning_edges = set(edges)

        probes = rng.choice(len(tuple(pool)), size=min(64, len(pool)), replace=False)
        pool_list = tuple(pool)
        probe_d = [abs(delta(pool_list[x])) for x in probes]
        T0 = max(float(np.median(probe_d)), 1e-3); Tf = T0 * 1e-3
        for step in range(cfg.anneal_steps):
            proposals = []
            for _ in range(cfg.proposal_batch):
                if edges and rng.random() < .55:
                    proposals.append(tuple(list(edges)[rng.integers(len(edges))]))
                elif rng.random() < cfg.global_add_probability:
                    if all_pairs is None: all_pairs = tuple(combinations(range(p), 2))
                    for _ in range(20):
                        e = all_pairs[rng.integers(len(all_pairs))]
                        if e not in edges: break
                    proposals.append(e)
                else:
                    for _ in range(20):
                        e = pool_list[rng.integers(len(pool_list))]
                        if e not in edges: break
                    proposals.append(e)
            ranked = []
            history_weight = np.clip((step / cfg.anneal_steps - .10) / .20, 0, 1)
            for e in proposals:
                d = delta(e); unique_tested.add(e)
                h = history.get(e, (d, d*d, np.sign(d)))
                uncertainty = max(h[1] - h[0] ** 2, 0) ** .5
                rank = d + history_weight * (.15 * uncertainty - .10 * abs(h[2]))
                ranked.append((rank, d, e))
            _, d, e = min(ranked)
            T = T0 * (Tf / T0) ** (step / max(cfg.anneal_steps - 1, 1))
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-12)):
                a, b = e
                if e in edges:
                    edges.remove(e); ns[a].remove(b); ns[b].remove(a)
                    accepted_deletes += 1
                else:
                    edges.add(e); ns[a].add(b); ns[b].add(a)
                    accepted_adds += 1
                current += d; accepted += 1
                if current < best_energy:
                    best_energy, best_edges = current, set(edges)
            oldm, oldq, olds = history.get(e, (d, d*d, np.sign(d)))
            beta = cfg.history_beta
            history[e] = (beta*oldm+(1-beta)*d, beta*oldq+(1-beta)*d*d,
                          beta*olds+(1-beta)*np.sign(d))

        # Return to the best visited state and make it deletion-locally optimal.
        after_annealing_best_edges = set(best_edges)
        edges=set(best_edges); ns=neighbors(edges); current=best_energy
        post_pruned=0
        for _ in range(6):
            removed_this_pass=0
            for e in sorted(list(edges)):
                d=delta(e); unique_tested.add(e)
                if d < -1e-10:
                    a,b=e; edges.remove(e); ns[a].remove(b); ns[b].remove(a)
                    current += d; pruned += 1; post_pruned += 1; removed_this_pass += 1
            if not removed_this_pass: break
        if current < best_energy: best_energy,best_edges=current,set(edges)

        def transition(before, after):
            union_edges = before | after
            return {
                "before": len(before),
                "after": len(after),
                "kept": len(before & after),
                "added": len(after - before),
                "removed": len(before - after),
                "symmetric_difference": len(before ^ after),
                "jaccard": len(before & after) / max(len(union_edges), 1),
            }

        self.edges_active_ = np.array(sorted(best_edges), dtype=np.int32).reshape(-1, 2)
        self.edges_ = self.active_[self.edges_active_] if len(best_edges) else np.empty((0,2), np.int32)
        self.best_energy_ = best_energy
        self.acceptance_rate_ = accepted / max(cfg.anneal_steps, 1)
        self.accepted_adds_ = accepted_adds
        self.accepted_deletes_ = accepted_deletes
        self.pruned_edges_ = pruned
        self.initial_pruned_edges_ = initial_pruned
        self.post_pruned_edges_ = post_pruned
        self.after_initial_pruning_edge_count_ = len(after_initial_pruning_edges)
        self.after_annealing_best_edge_count_ = len(after_annealing_best_edges)
        self.stage_transitions_ = {
            "start_to_initial_pruning": transition(selected_start_edges, after_initial_pruning_edges),
            "initial_pruning_to_annealing_best": transition(after_initial_pruning_edges, after_annealing_best_edges),
            "annealing_best_to_final_pruning": transition(after_annealing_best_edges, best_edges),
            "selected_start_to_final": transition(selected_start_edges, best_edges),
        }
        self.unique_tested_pairs_ = len(unique_tested)
        self.fit_seconds_ = time.perf_counter() - t0
        self.metadata_ = asdict(cfg)
        return self

    def adjacency(self) -> np.ndarray:
        A = np.zeros((self.n_vertices_, self.n_vertices_), dtype=np.float32)
        if len(self.edges_):
            A[self.edges_[:, 0], self.edges_[:, 1]] = 1
            A[self.edges_[:, 1], self.edges_[:, 0]] = 1
        return A


def recover_graph(X: np.ndarray, X_val: np.ndarray | None = None,
                  config: RecoveryConfig | None = None):
    """Requested matrix-to-graph function; returns adjacency and fitted estimator."""
    model = LatentGraphRecovery(config).fit(X, X_val)
    return model.adjacency(), model
