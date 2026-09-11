"""MNIST loading, graph evaluation, and frozen classifiers.

Computational definitions extracted unchanged from original_sources/experiment.py.
The SAGE variant uses symmetric normalization, not the canonical neighbor mean.
"""

from __future__ import annotations
import struct
import zipfile
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, f1_score, log_loss


def idx(raw: bytes):
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2051:
        rows, cols = struct.unpack(">II", raw[8:16])
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows * cols)
    if magic == 2049:
        return np.frombuffer(raw, np.uint8, offset=8)
    raise ValueError(f"Unknown IDX magic {magic}")


def load_mnist(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        def one(token):
            matches = [x for x in names if x.split("/")[-1] == token]
            # Some supplied archives contain tiny duplicate/link-like entries.
            chosen = max(matches, key=lambda x: z.getinfo(x).file_size)
            return idx(z.read(chosen))

        return (
            one("train-images.idx3-ubyte"),
            one("train-labels.idx1-ubyte"),
            one("t10k-images.idx3-ubyte"),
            one("t10k-labels.idx1-ubyte"),
        )


def grid_edges(side=28):
    e = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if r + 1 < side:
                e.append((i, i + side))
            if c + 1 < side:
                e.append((i, i + 1))
    return np.asarray(e, np.int32)


def adjacency(n, edges):
    A = np.zeros((n, n), np.float32)
    if len(edges):
        A[edges[:, 0], edges[:, 1]] = A[edges[:, 1], edges[:, 0]] = 1
    return A


def graph_metrics(pred_edges, truth_edges, active, variances):
    active = set(np.flatnonzero(active).tolist())
    truth = {
        tuple(sorted(map(int, e)))
        for e in truth_edges
        if int(e[0]) in active and int(e[1]) in active
    }
    pred = {tuple(sorted(map(int, e))) for e in pred_edges}
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denom = sum(np.sqrt(variances[a] * variances[b]) for a, b in truth)
    weighted = sum(np.sqrt(variances[a] * variances[b]) for a, b in truth & pred) / max(
        denom, 1e-12
    )
    n = len(active)
    vertices_total = len(variances)
    Ap = adjacency(
        vertices_total, np.asarray(list(pred), dtype=np.int32).reshape(-1, 2)
    )
    At = adjacency(
        vertices_total, np.asarray(list(truth), dtype=np.int32).reshape(-1, 2)
    )
    cp, labp = connected_components(
        csr_matrix(Ap[np.ix_(sorted(active), sorted(active))]), directed=False
    )
    ct, labt = connected_components(
        csr_matrix(At[np.ix_(sorted(active), sorted(active))]), directed=False
    )
    largest_p = int(np.bincount(labp).max()) if len(labp) else 0
    largest_t = int(np.bincount(labt).max()) if len(labt) else 0
    deg = Ap.sum(0)
    return dict(
        vertices_total=vertices_total,
        original_grid_edges_full=len(truth_edges),
        active_vertices=n,
        truth_informative_edges=len(truth),
        truth_informative_components=int(ct),
        truth_largest_component=largest_t,
        recovered_edges=len(pred),
        recovered_average_degree=float(2 * len(pred) / max(n, 1)),
        recovered_max_degree=int(deg.max()),
        recovered_density=float(len(pred) / max(n * (n - 1) / 2, 1)),
        recovered_components=int(cp),
        recovered_largest_component=largest_p,
        TP=tp,
        FP=fp,
        FN=fn,
        precision=float(precision),
        recall=float(recall),
        edge_f1=float(f1),
        informative_weighted_recall=float(weighted),
    )


def norm_sparse(A, self_loops=True):
    A = A.copy()
    np.fill_diagonal(A, 1 if self_loops else 0)
    d = np.maximum(A.sum(1), 1)
    A = A / np.sqrt(d[:, None] * d[None, :])
    r, c = np.nonzero(A)
    ind = torch.tensor(np.stack([r, c]), dtype=torch.long)
    return torch.sparse_coo_tensor(ind, torch.tensor(A[r, c]), A.shape).coalesce()


class TinyGCN(nn.Module):
    def __init__(self, n, hidden=12):
        super().__init__()
        self.lin1 = nn.Linear(1, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden * n, 10)

    def agg(self, A, h):
        b, n, c = h.shape
        return (
            torch.sparse.mm(A, h.permute(1, 0, 2).reshape(n, b * c))
            .reshape(n, b, c)
            .permute(1, 0, 2)
        )

    def forward(self, x, A):
        h = torch.relu(self.lin1(x[:, :, None]))
        h = torch.relu(self.lin2(self.agg(A, h)))
        h = self.agg(A, h)
        # Fixed vertex identities are legitimate (all samples share them); no 2-D
        # coordinates or inverse permutation are supplied.
        return self.out(h.flatten(1))


class FrozenGraphSAGE(nn.Module):
    """GraphSAGE-inspired concatenation with symmetrically normalized neighbors."""

    def __init__(self, n, hidden=12):
        super().__init__()
        self.lin1 = nn.Linear(2, hidden)
        self.lin2 = nn.Linear(hidden * 2, hidden)
        self.out = nn.Linear(hidden * n, 10)

    def mean_neigh(self, A, h):
        b, n, c = h.shape
        return (
            torch.sparse.mm(A, h.permute(1, 0, 2).reshape(n, b * c))
            .reshape(n, b, c)
            .permute(1, 0, 2)
        )

    def forward(self, x, A):
        h0 = x[:, :, None]
        h = torch.relu(self.lin1(torch.cat([h0, self.mean_neigh(A, h0)], 2)))
        h = torch.relu(self.lin2(torch.cat([h, self.mean_neigh(A, h)], 2)))
        return self.out(h.flatten(1))


def train_frozen_gnn(
    Xtr,
    ytr,
    Xv,
    yv,
    Xte,
    yte,
    A_train,
    A_eval,
    seed,
    architecture="GCN",
    epochs=8,
    batch=256,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = (
        TinyGCN(Xtr.shape[1])
        if architecture == "GCN"
        else FrozenGraphSAGE(Xtr.shape[1])
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss()
    As = norm_sparse(A_train, self_loops=architecture == "GCN")
    best = None
    bestloss = np.inf
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(Xtr))
        for st in range(0, len(order), batch):
            q = order[st : st + batch]
            xb = torch.from_numpy(Xtr[q])
            yb = torch.from_numpy(ytr[q].astype(np.int64))
            opt.zero_grad()
            loss = lossfn(model(xb, As), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xv), As)
            vl = lossfn(logits, torch.from_numpy(yv.astype(np.int64))).item()
        if vl < bestloss:
            bestloss = vl
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    results = {}
    # Weights stay frozen; only the adjacency used by message passing changes.
    with torch.no_grad():
        for name, A in A_eval.items():
            Ae = norm_sparse(A, self_loops=architecture == "GCN")
            probs = []
            for st in range(0, len(Xte), batch):
                probs.append(
                    torch.softmax(
                        model(torch.from_numpy(Xte[st : st + batch]), Ae), 1
                    ).numpy()
                )
            p = np.concatenate(probs)
            pred = p.argmax(1)
            results[name] = dict(
                accuracy=accuracy_score(yte, pred),
                macro_f1=f1_score(yte, pred, average="macro"),
                nll=log_loss(yte, p),
            )
    return results
