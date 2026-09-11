"""Mathematical and provenance checks; does not train classifiers."""

import ast
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mnist_components import load_mnist
from latent_graph import LatentGraphRecovery, RecoveryConfig

ROOT = Path(__file__).resolve().parents[1]


def computational_tree(node):
    """Ignore documentation strings, but compare all executable AST nodes."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if child.body and isinstance(child.body[0], ast.Expr):
                value = child.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    child.body.pop(0)
    return ast.dump(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip", type=Path, help="Optionally check an external MNIST IDX ZIP"
    )
    args = parser.parse_args()
    checks = {}
    for filename in ("latent_graph.py",):
        a = ast.dump(
            ast.parse((ROOT / "src" / filename).read_text(encoding="utf-8-sig"))
        )
        b = ast.dump(
            ast.parse(
                (ROOT / "original_sources" / filename).read_text(encoding="utf-8-sig")
            )
        )
        assert a == b, filename
    checks["formatted_sources_ast_equivalent"] = True
    old = ast.parse(
        (ROOT / "original_sources/experiment.py").read_text(encoding="utf-8-sig")
    )
    new = ast.parse((ROOT / "src/mnist_components.py").read_text(encoding="utf-8"))
    definitions = {
        n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    for node in new.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert computational_tree(node) == computational_tree(
                definitions[node.name]
            ), node.name
    checks["extracted_components_computational_ast_equivalent"] = True
    rng = np.random.default_rng(13)
    z, v = rng.normal(size=(90, 8)), rng.normal(size=(35, 8))
    c, gram_v = z.T @ z / len(z), v.T @ v / len(v)
    q, i, alpha = np.array([0, 3, 5]), 2, 0.01
    w = np.linalg.solve(c[np.ix_(q, q)] + alpha * np.eye(len(q)), c[q, i])
    direct = np.mean((v[:, i] - v[:, q] @ w) ** 2)
    gram = gram_v[i, i] - 2 * w @ gram_v[q, i] + w @ gram_v[np.ix_(q, q)] @ w
    assert np.isclose(direct, gram, atol=1e-12)
    checks["ridge_mse_gram_matches_direct"] = True
    small = rng.normal(size=(120, 10))
    model = LatentGraphRecovery(RecoveryConfig(top_k=4, anneal_steps=30, seed=7)).fit(
        small[:90], small[90:]
    )
    a = model.adjacency()
    assert np.array_equal(a, a.T) and not np.diag(a).any()
    assert a.sum() == 2 * len(model.edges_)
    for transition in model.stage_transitions_.values():
        assert (
            transition["after"]
            == transition["before"] + transition["added"] - transition["removed"]
        )
    checks["symmetry_self_loops_and_stage_accounting"] = True
    if args.zip:
        data, labels, test, test_labels = load_mnist(args.zip)
        assert data.shape == (60000, 784) and test.shape == (10000, 784)
        assert labels.shape == (60000,) and test_labels.shape == (10000,)
        checks["mnist_shapes"] = True
        checks["data_sha256"] = hashlib.sha256(args.zip.read_bytes()).hexdigest()
    (ROOT / "checks").mkdir(exist_ok=True)
    (ROOT / "checks/validation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
