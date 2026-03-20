from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from msrhgnn_model import (
    LOCAL_DD_TRAIN_RELATION,
    MultiViewMSRHGNN,
    build_metapath_relations,
)

DRUGSIM_GIP_RELATION = "drug__drugsim_gip__drug"
DRUGSIM_DRSIE_RELATION = "drug__drugsim_drsie__drug"
UNSAFE_DISEASE_SIM_METHODS = {"shared_drug_jaccard_fallback"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_tensor_to_pairs(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    if edge_index.numel() == 0:
        return []
    return list(map(tuple, edge_index.t().cpu().tolist()))


def split_positive_edges(
    pairs: Sequence[Tuple[int, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else max(0, n - n_train - 1)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]
    if len(test_pairs) == 0 and len(val_pairs) > 1:
        test_pairs.append(val_pairs.pop())
    if len(val_pairs) == 0 and len(test_pairs) > 1:
        val_pairs.append(test_pairs.pop())
    return train_pairs, val_pairs, test_pairs


def sample_negative_pairs(
    num_drugs: int,
    num_diseases: int,
    positive_pairs: set[Tuple[int, int]],
    num_samples: int,
    rng: random.Random,
    forbidden: set[Tuple[int, int]] | None = None,
) -> List[Tuple[int, int]]:
    forbidden = forbidden or set()
    chosen: set[Tuple[int, int]] = set()
    while len(chosen) < num_samples:
        pair = (rng.randrange(num_drugs), rng.randrange(num_diseases))
        if pair in positive_pairs or pair in forbidden or pair in chosen:
            continue
        chosen.add(pair)
    return list(chosen)


def pairs_to_tensor(pairs: Sequence[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def evaluate_scores(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> Dict[str, float]:
    logits = torch.cat([pos_logits, neg_logits], dim=0).detach().cpu().numpy()
    labels = np.concatenate([np.ones(len(pos_logits)), np.zeros(len(neg_logits))])
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "aupr": float(average_precision_score(labels, probs)),
    }


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    matrix = np.nan_to_num(matrix.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(matrix.min())
    max_v = float(matrix.max())
    if math.isclose(min_v, max_v):
        return np.zeros_like(matrix)
    return (matrix - min_v) / (max_v - min_v)


def sparsify_top_k(matrix: np.ndarray, k: int = 5, symmetric: bool = True) -> np.ndarray:
    n = matrix.shape[0]
    sparse = np.zeros_like(matrix)
    if n == 0:
        return sparse
    for i in range(n):
        row = matrix[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:] if k < n else np.where(np.isfinite(row))[0]
        idx = [j for j in idx if j != i and np.isfinite(row[j]) and row[j] > 0]
        for j in idx:
            sparse[i, j] = matrix[i, j]
    if symmetric:
        sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def adjacency_to_edge_tensors(matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = np.argwhere(matrix > 0)
    if coords.size == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
    edge_index = torch.tensor(coords.T, dtype=torch.long)
    edge_weight = torch.tensor(matrix[coords[:, 0], coords[:, 1]], dtype=torch.float)
    return edge_index, edge_weight


def build_train_only_gip_relation(
    num_drugs: int,
    num_diseases: int,
    train_pairs: Sequence[Tuple[int, int]],
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    y = np.zeros((num_drugs, num_diseases), dtype=float)
    for drug_idx, disease_idx in train_pairs:
        y[int(drug_idx), int(disease_idx)] = 1.0
    row_norm = np.sum(y**2, axis=1)
    gamma = 1.0 / np.mean(row_norm[row_norm > 0]) if np.any(row_norm > 0) else 1.0
    dist = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    sim = np.exp(-gamma * dist)
    sim = normalize_matrix(sim)
    sim = sparsify_top_k(sim, k=top_k, symmetric=True)
    return adjacency_to_edge_tensors(sim)


def build_train_only_drsie_relation(
    num_drugs: int,
    train_pairs: Sequence[Tuple[int, int]],
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    disease_sets: Dict[int, set[int]] = {}
    disease_counts: Dict[int, int] = {}
    for drug_idx, disease_idx in train_pairs:
        disease_sets.setdefault(int(drug_idx), set()).add(int(disease_idx))
        disease_counts[int(disease_idx)] = disease_counts.get(int(disease_idx), 0) + 1

    sim = np.zeros((num_drugs, num_drugs), dtype=float)
    total_drugs = max(1, num_drugs)
    for i in range(num_drugs):
        set_i = disease_sets.get(i, set())
        for j in range(i, num_drugs):
            set_j = disease_sets.get(j, set())
            union = set_i | set_j
            if not union:
                score = 0.0
            else:
                common = set_i & set_j

                def weight(did: int) -> float:
                    return math.log(1.0 + total_drugs / max(1, disease_counts.get(int(did), 1)))

                common_w = sum(weight(did) for did in common)
                union_w = sum(weight(did) for did in union)
                score = common_w / union_w if union_w else 0.0
            sim[i, j] = sim[j, i] = score
    sim = normalize_matrix(sim)
    sim = sparsify_top_k(sim, k=top_k, symmetric=True)
    return adjacency_to_edge_tensors(sim)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def summarize_high_order_views(relation_edges: Dict[str, torch.Tensor], meta_edges: Dict[str, torch.Tensor]) -> Dict[str, object]:
    def has_relation(name: str) -> bool:
        edge_index = relation_edges.get(name)
        return edge_index is not None and edge_index.numel() > 0

    has_dti = has_relation("drug__targets__gene")
    has_dg = has_relation("disease__associated_with__gene")
    has_gg = has_relation("gene__interacts__gene")

    active_paths: List[str] = []
    if "drug__meta_dgd__disease" in meta_edges:
        active_paths.append("Drug-Protein-Disease")
    if "disease__meta_dgd__disease" in meta_edges:
        active_paths.append("Disease-Protein-Disease")
    if "disease__meta_dggd__disease" in meta_edges:
        active_paths.append("Disease-Protein-Protein-Disease")

    missing_prerequisites: List[str] = []
    if not has_dti:
        missing_prerequisites.append("drug__targets__gene")
    if not has_dg:
        missing_prerequisites.append("disease__associated_with__gene")
    if not has_gg:
        missing_prerequisites.append("gene__interacts__gene")

    return {
        "implemented_paths": [
            "Drug-Protein-Disease",
            "Disease-Protein-Disease",
            "Disease-Protein-Protein-Disease",
        ],
        "active_paths": active_paths,
        "complete": has_dti and has_dg and has_gg,
        "missing_prerequisites": missing_prerequisites,
        "active_metapath_relations": list(meta_edges.keys()),
        "note": (
            "Disease-Protein-Protein-Disease depends on gene__interacts__gene. "
            "If that relation is absent, training uses partial high-order views."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the project-local simplified MSRHGNN model.")
    parser.add_argument("--graph", type=str, default="data/final/final_graph_data.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="artifacts/train_run")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--meta-top-k", type=int, default=10)
    parser.add_argument("--drug-sim-top-k", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = torch.load(args.graph, map_location="cpu")
    disease_g_method = graph.get("metadata", {}).get("disease__disimnet_g__disease", {}).get("method")
    if disease_g_method in UNSAFE_DISEASE_SIM_METHODS:
        raise RuntimeError(
            "Unsafe graph detected: disease__disimnet_g__disease uses "
            f"{disease_g_method}, which leaks labels via full drug-disease edges. "
            "Rebuild the graph with the updated pipeline before training."
        )
    drug_x = graph["node_features"]["drug"].float().to(device)
    disease_x = graph["node_features"]["disease"].float().to(device)
    gene_x = graph["node_features"]["gene"].float().to(device)

    relation_edges_cpu = {k: v.long().cpu() for k, v in graph["edge_index"].items()}
    raw_edge_weight = graph.get("edge_weight", {}) or {}
    relation_weights_cpu = {
        k: raw_edge_weight[k].float().cpu() if k in raw_edge_weight else torch.ones((v.size(1),), dtype=torch.float)
        for k, v in relation_edges_cpu.items()
    }

    positive_pairs = edge_tensor_to_pairs(relation_edges_cpu["drug__treats__disease"])
    train_pairs, val_pairs, test_pairs = split_positive_edges(
        positive_pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    all_positive = set(positive_pairs)
    rng = random.Random(args.seed)
    val_neg = sample_negative_pairs(len(graph["node_ids"]["drug"]), len(graph["node_ids"]["disease"]), all_positive, len(val_pairs), rng)
    test_neg = sample_negative_pairs(
        len(graph["node_ids"]["drug"]),
        len(graph["node_ids"]["disease"]),
        all_positive,
        len(test_pairs),
        rng,
        forbidden=set(val_neg),
    )

    train_gip_edge_index, train_gip_edge_weight = build_train_only_gip_relation(
        num_drugs=len(graph["node_ids"]["drug"]),
        num_diseases=len(graph["node_ids"]["disease"]),
        train_pairs=train_pairs,
        top_k=args.drug_sim_top_k,
    )
    train_drsie_edge_index, train_drsie_edge_weight = build_train_only_drsie_relation(
        num_drugs=len(graph["node_ids"]["drug"]),
        train_pairs=train_pairs,
        top_k=args.drug_sim_top_k,
    )
    relation_edges_cpu[DRUGSIM_GIP_RELATION] = train_gip_edge_index
    relation_weights_cpu[DRUGSIM_GIP_RELATION] = train_gip_edge_weight
    relation_edges_cpu[DRUGSIM_DRSIE_RELATION] = train_drsie_edge_index
    relation_weights_cpu[DRUGSIM_DRSIE_RELATION] = train_drsie_edge_weight

    meta_edges, meta_weights = build_metapath_relations(
        relation_edges_cpu,
        relation_weights_cpu,
        num_drugs=len(graph["node_ids"]["drug"]),
        num_diseases=len(graph["node_ids"]["disease"]),
        top_k=args.meta_top_k,
    )
    relation_edges_cpu.update(meta_edges)
    relation_weights_cpu.update(meta_weights)
    high_order_views = summarize_high_order_views(relation_edges_cpu, meta_edges)

    relation_edges = {k: v.to(device) for k, v in relation_edges_cpu.items()}
    relation_weights = {k: w.to(device) for k, w in relation_weights_cpu.items()}

    relation_edges[LOCAL_DD_TRAIN_RELATION] = pairs_to_tensor(train_pairs, device).t().contiguous() if train_pairs else torch.empty((2, 0), dtype=torch.long, device=device)
    relation_weights[LOCAL_DD_TRAIN_RELATION] = torch.ones((len(train_pairs),), dtype=torch.float, device=device)

    model = MultiViewMSRHGNN(
        drug_dim=drug_x.size(1),
        disease_dim=disease_x.size(1),
        gene_dim=gene_x.size(1),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = -math.inf
    history: List[Dict[str, float]] = []

    train_tensor = pairs_to_tensor(train_pairs, device)
    val_pos_tensor = pairs_to_tensor(val_pairs, device)
    val_neg_tensor = pairs_to_tensor(val_neg, device)
    test_pos_tensor = pairs_to_tensor(test_pairs, device)
    test_neg_tensor = pairs_to_tensor(test_neg, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_neg_pairs = sample_negative_pairs(
            len(graph["node_ids"]["drug"]),
            len(graph["node_ids"]["disease"]),
            all_positive,
            len(train_pairs),
            rng,
        )
        train_neg_tensor = pairs_to_tensor(train_neg_pairs, device)

        drug_repr, disease_repr, aux = model.encode(drug_x, disease_x, gene_x, relation_edges, relation_weights)
        pos_logits = model.score_pairs(drug_repr, disease_repr, train_tensor)
        neg_logits = model.score_pairs(drug_repr, disease_repr, train_neg_tensor)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        record = {"epoch": epoch, "loss": float(loss.item())}
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                drug_repr, disease_repr, aux = model.encode(drug_x, disease_x, gene_x, relation_edges, relation_weights)
                val_pos_logits = model.score_pairs(drug_repr, disease_repr, val_pos_tensor)
                val_neg_logits = model.score_pairs(drug_repr, disease_repr, val_neg_tensor)
                metrics = evaluate_scores(val_pos_logits, val_neg_logits)
                record.update({"val_auc": metrics["auc"], "val_aupr": metrics["aupr"]})
                if metrics["aupr"] > best_val:
                    best_val = metrics["aupr"]
                    best_state = {
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "metrics": metrics,
                        "aux": aux,
                    }
        history.append(record)
        if "val_auc" in record:
            print(f"epoch {epoch:03d} | loss={record['loss']:.4f} | val_auc={record['val_auc']:.4f} | val_aupr={record['val_aupr']:.4f}")
        else:
            print(f"epoch {epoch:03d} | loss={record['loss']:.4f}")

    if best_state is None:
        raise RuntimeError("Training finished without a valid validation state.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    with torch.no_grad():
        drug_repr, disease_repr, aux = model.encode(drug_x, disease_x, gene_x, relation_edges, relation_weights)
        test_pos_logits = model.score_pairs(drug_repr, disease_repr, test_pos_tensor)
        test_neg_logits = model.score_pairs(drug_repr, disease_repr, test_neg_tensor)
        test_metrics = evaluate_scores(test_pos_logits, test_neg_logits)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "best_val": best_state["metrics"],
            "test_metrics": test_metrics,
            "aux": aux,
            "splits": {
                "train_pos": train_pairs,
                "val_pos": val_pairs,
                "test_pos": test_pairs,
                "val_neg": val_neg,
                "test_neg": test_neg,
            },
            "metapath_relations": list(meta_edges.keys()),
        },
        out_dir / "model.pt",
    )

    summary = {
        "best_val": best_state["metrics"],
        "test_metrics": test_metrics,
        "leakage_fix": {
            "status": "enabled",
            "train_only_relations": [DRUGSIM_GIP_RELATION, DRUGSIM_DRSIE_RELATION, LOCAL_DD_TRAIN_RELATION],
            "drug_sim_top_k": args.drug_sim_top_k,
            "note": "GIP and DRSIE are rebuilt from train-only drug-disease edges after splitting.",
        },
        "drug_sim_attention": aux["drug_sim_weights"],
        "disease_sim_attention": aux["disease_sim_weights"],
        "drug_local_attention": aux["drug_local_weights"],
        "disease_local_attention": aux["disease_local_weights"],
        "drug_global_attention": aux["drug_global_weights"],
        "disease_global_attention": aux["disease_global_weights"],
        "num_train_pos": len(train_pairs),
        "num_val_pos": len(val_pairs),
        "num_test_pos": len(test_pairs),
        "device": str(device),
        "metapath_relations": list(meta_edges.keys()),
        "high_order_views": high_order_views,
    }
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "history.json", {"history": history})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
