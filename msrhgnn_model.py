from __future__ import annotations

"""
MSRHGNN-style model for the current project.

Implemented to match the requested structure:
1. 3 disease similarity networks + 3 drug similarity networks
2. GIN update per similarity view + adaptive fusion + graph transformer
3. Heterogeneous low-order / high-order relation modeling
4. Gate fusion + concat + bilinear decoding

Reference ideas come from:
- `MSRHGNN/graph_transformer*.py`
- `MSRHGNN/SRHGN.py`

This file is self-contained and does not depend on DGL.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SIM_DRUG_RELATIONS = [
    "drug__drugsim_morgan__drug",
    "drug__drugsim_gip__drug",
    "drug__drugsim_drsie__drug",
]
SIM_DISEASE_RELATIONS = [
    "disease__disimnet_o__disease",
    "disease__disimnet_h__disease",
    "disease__disimnet_g__disease",
]
LOCAL_DTI_RELATION = "drug__targets__gene"
LOCAL_DG_RELATION = "disease__associated_with__gene"
LOCAL_DD_TRAIN_RELATION = "drug__treats__disease_train"
GENE_NETWORK_RELATION = "gene__interacts__gene"
META_DRUG_DISEASE_RELATION = "drug__meta_dgd__disease"
META_DISEASE_SHARED_RELATION = "disease__meta_dgd__disease"
META_DISEASE_INTERACT_RELATION = "disease__meta_dggd__disease"


def ensure_edge_weight(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    if edge_weight is None:
        return torch.ones((edge_index.size(1),), dtype=torch.float, device=device)
    return edge_weight.float().to(device)


def weighted_mean_aggregate(
    src_x: torch.Tensor,
    edge_index: torch.Tensor,
    num_dst: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((num_dst, src_x.size(1)), device=src_x.device, dtype=src_x.dtype)
    src, dst = edge_index[0], edge_index[1]
    weight = ensure_edge_weight(edge_index, edge_weight, src_x.device).unsqueeze(-1).to(src_x.dtype)
    out = torch.zeros((num_dst, src_x.size(1)), device=src_x.device, dtype=src_x.dtype)
    deg = torch.zeros((num_dst, 1), device=src_x.device, dtype=src_x.dtype)
    out.index_add_(0, dst, src_x[src] * weight)
    deg.index_add_(0, dst, weight)
    return out / deg.clamp_min(1e-8)


def reverse_edges(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if edge_index.numel() == 0:
        return edge_index.clone(), edge_weight.clone() if edge_weight is not None else None
    return torch.stack([edge_index[1], edge_index[0]], dim=0), edge_weight.clone() if edge_weight is not None else None


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    max_scores = torch.full((num_segments,), -float("inf"), device=scores.device, dtype=scores.dtype)
    max_scores.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    stabilized = scores - max_scores[index]
    exp_scores = torch.exp(stabilized)
    denom = torch.zeros((num_segments,), device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, index, exp_scores)
    return exp_scores / denom[index].clamp_min(1e-12)


def merge_relation_graphs(
    relation_names: Sequence[str],
    relation_edges: Dict[str, torch.Tensor],
    relation_weights: Dict[str, torch.Tensor],
    fusion_weights: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_chunks: List[torch.Tensor] = []
    weight_chunks: List[torch.Tensor] = []
    device = fusion_weights.device
    for scalar, name in zip(fusion_weights, relation_names):
        edge_index = relation_edges.get(name)
        if edge_index is None or edge_index.numel() == 0:
            continue
        weight = relation_weights.get(name)
        if weight is None or weight.numel() == 0:
            weight = torch.ones((edge_index.size(1),), device=device, dtype=torch.float)
        edge_chunks.append(edge_index)
        weight_chunks.append(weight * scalar)
    if not edge_chunks:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float, device=device)
    edge_index = torch.cat(edge_chunks, dim=1)
    edge_weight = torch.cat(weight_chunks, dim=0)
    linear = edge_index[0] * num_nodes + edge_index[1]
    unique_linear, inverse = torch.unique(linear, return_inverse=True)
    merged_weight = torch.zeros((unique_linear.size(0),), device=device, dtype=edge_weight.dtype)
    merged_weight.index_add_(0, inverse, edge_weight)
    merged_edge_index = torch.stack([unique_linear // num_nodes, unique_linear % num_nodes], dim=0)
    return merged_edge_index.long(), merged_weight


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if math.isclose(min_v, max_v):
        return [1.0 for _ in values]
    return [(x - min_v) / (max_v - min_v) for x in values]


def build_lookup(edge_index: torch.Tensor) -> Dict[int, set[int]]:
    lookup: Dict[int, set[int]] = {}
    if edge_index.numel() == 0:
        return lookup
    for src, dst in edge_index.t().cpu().tolist():
        lookup.setdefault(int(src), set()).add(int(dst))
    return lookup


def build_weighted_lookup(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> Dict[int, List[Tuple[int, float]]]:
    lookup: Dict[int, List[Tuple[int, float]]] = {}
    if edge_index.numel() == 0:
        return lookup
    weights = edge_weight.cpu().tolist() if edge_weight is not None and edge_weight.numel() else [1.0] * edge_index.size(1)
    for (src, dst), w in zip(edge_index.t().cpu().tolist(), weights):
        lookup.setdefault(int(src), []).append((int(dst), float(w)))
    return lookup


def best_cross_gene_score(genes_a: set[int], genes_b: set[int], gene_neighbors: Dict[int, List[Tuple[int, float]]]) -> float:
    if not genes_a or not genes_b or not gene_neighbors:
        return 0.0
    total = 0.0
    for gene in genes_a:
        best = 0.0
        for nbr, score in gene_neighbors.get(gene, []):
            if nbr in genes_b:
                best = max(best, score)
        total += best
    return total / max(len(genes_a), 1)


def topk_sparse_edges(
    rows: List[int],
    cols: List[int],
    vals: List[float],
    num_src: int,
    num_dst: int,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    buckets: Dict[int, List[Tuple[int, float]]] = {}
    for r, c, v in zip(rows, cols, vals):
        if v <= 0:
            continue
        buckets.setdefault(r, []).append((c, v))
    out_rows: List[int] = []
    out_cols: List[int] = []
    out_vals: List[float] = []
    for src in range(num_src):
        items = sorted(buckets.get(src, []), key=lambda x: x[1], reverse=True)[:top_k]
        for dst, score in items:
            out_rows.append(src)
            out_cols.append(dst)
            out_vals.append(score)
    out_vals = min_max_normalize(out_vals)
    if not out_rows:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
    edge_index = torch.tensor([out_rows, out_cols], dtype=torch.long)
    edge_weight = torch.tensor(out_vals, dtype=torch.float)
    linear = edge_index[0] * num_dst + edge_index[1]
    unique_linear, inverse = torch.unique(linear, return_inverse=True)
    merged_weight = torch.zeros((unique_linear.size(0),), dtype=torch.float)
    merged_weight.index_add_(0, inverse, edge_weight)
    merged_edge_index = torch.stack([unique_linear // num_dst, unique_linear % num_dst], dim=0)
    return merged_edge_index.long(), merged_weight


def build_metapath_relations(
    relation_edges: Dict[str, torch.Tensor],
    relation_weights: Dict[str, torch.Tensor],
    num_drugs: int,
    num_diseases: int,
    top_k: int = 10,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    new_edges: Dict[str, torch.Tensor] = {}
    new_weights: Dict[str, torch.Tensor] = {}

    dti = relation_edges.get(LOCAL_DTI_RELATION)
    dg = relation_edges.get(LOCAL_DG_RELATION)
    gg = relation_edges.get(GENE_NETWORK_RELATION)
    gg_w = relation_weights.get(GENE_NETWORK_RELATION)
    if dti is None or dg is None or dti.numel() == 0 or dg.numel() == 0:
        return new_edges, new_weights

    drug_to_genes = build_lookup(dti)
    disease_to_genes = build_lookup(dg)
    gene_neighbors = build_weighted_lookup(gg, gg_w) if gg is not None and gg.numel() else {}

    drug_rows: List[int] = []
    drug_cols: List[int] = []
    drug_vals: List[float] = []
    for drug_idx, drug_genes in drug_to_genes.items():
        for disease_idx, disease_genes in disease_to_genes.items():
            shared = len(drug_genes & disease_genes)
            interaction = best_cross_gene_score(drug_genes, disease_genes, gene_neighbors)
            score = float(shared) + interaction
            if score > 0:
                drug_rows.append(drug_idx)
                drug_cols.append(disease_idx)
                drug_vals.append(score)
    edge_index, edge_weight = topk_sparse_edges(drug_rows, drug_cols, drug_vals, num_src=num_drugs, num_dst=num_diseases, top_k=top_k)
    if edge_index.numel() > 0:
        new_edges[META_DRUG_DISEASE_RELATION] = edge_index
        new_weights[META_DRUG_DISEASE_RELATION] = edge_weight

    dis_rows: List[int] = []
    dis_cols: List[int] = []
    dis_vals: List[float] = []
    dis_rows_gg: List[int] = []
    dis_cols_gg: List[int] = []
    dis_vals_gg: List[float] = []
    for i in range(num_diseases):
        genes_i = disease_to_genes.get(i, set())
        if not genes_i:
            continue
        for j in range(num_diseases):
            if i == j:
                continue
            genes_j = disease_to_genes.get(j, set())
            if not genes_j:
                continue
            shared = len(genes_i & genes_j)
            if shared > 0:
                dis_rows.append(i)
                dis_cols.append(j)
                dis_vals.append(float(shared))
            interaction = best_cross_gene_score(genes_i, genes_j, gene_neighbors)
            if interaction > 0:
                dis_rows_gg.append(i)
                dis_cols_gg.append(j)
                dis_vals_gg.append(interaction)

    edge_index, edge_weight = topk_sparse_edges(dis_rows, dis_cols, dis_vals, num_src=num_diseases, num_dst=num_diseases, top_k=top_k)
    if edge_index.numel() > 0:
        new_edges[META_DISEASE_SHARED_RELATION] = edge_index
        new_weights[META_DISEASE_SHARED_RELATION] = edge_weight

    edge_index, edge_weight = topk_sparse_edges(dis_rows_gg, dis_cols_gg, dis_vals_gg, num_src=num_diseases, num_dst=num_diseases, top_k=top_k)
    if edge_index.numel() > 0:
        new_edges[META_DISEASE_INTERACT_RELATION] = edge_index
        new_weights[META_DISEASE_INTERACT_RELATION] = edge_weight

    return new_edges, new_weights


class WeightedGINLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        aggregated = weighted_mean_aggregate(x, edge_index, x.size(0), edge_weight)
        return self.mlp((1.0 + self.eps) * x + aggregated)


class MultiPoolAdaptiveFusion(nn.Module):
    """
    Adaptive fusion using GAP / GMP / GSP as requested in the specification.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, reps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reps:
            raise ValueError("Adaptive fusion received no views.")
        if len(reps) == 1:
            one = torch.ones((1,), device=reps[0].device, dtype=reps[0].dtype)
            return reps[0], one
        pooled = []
        for rep in reps:
            gap = rep.mean(dim=0)
            gmp = rep.max(dim=0).values
            gsp = rep.sum(dim=0)
            pooled.append(torch.cat([gap, gmp, gsp], dim=0))
        pooled = torch.stack(pooled, dim=0)
        weights = torch.softmax(self.scorer(pooled).squeeze(-1), dim=0)
        fused = sum(w * rep for w, rep in zip(weights, reps))
        return fused, weights


class DualAttentionFusion(nn.Module):
    """
    Lightweight semantic-aware + relation-aware dual attention,
    inspired by SRHGN high-order fusion.
    """

    def __init__(self, hidden_dim: int, max_relations: int) -> None:
        super().__init__()
        self.semantic_src = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_dst = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_score = nn.Linear(hidden_dim, 1)
        self.relation_emb = nn.Parameter(torch.randn(max_relations, hidden_dim))
        self.relation_score = nn.Linear(hidden_dim, 1, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.relation_emb)

    def forward(self, anchor: torch.Tensor, reps: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not reps:
            raise ValueError("DualAttentionFusion received no views.")
        if len(reps) == 1:
            one = torch.ones((1,), device=reps[0].device, dtype=reps[0].dtype)
            return reps[0], one
        anchor_summary = anchor.mean(dim=0, keepdim=True).expand(len(reps), -1)
        rep_summary = torch.stack([rep.mean(dim=0) for rep in reps], dim=0)
        semantic = self.semantic_score(
            torch.tanh(self.semantic_src(anchor_summary) + self.semantic_dst(rep_summary))
        ).squeeze(-1)
        relation = self.relation_score(self.relation_emb[: len(reps)]).squeeze(-1)
        alpha = torch.sigmoid(self.alpha_logit)
        weights = torch.softmax(alpha * semantic + (1.0 - alpha) * relation, dim=0)
        fused = sum(w * rep for w, rep in zip(weights, reps))
        return fused, weights


class SparseGraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm2(x + self.ffn(self.norm1(x)))
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        src, dst = edge_index[0], edge_index[1]
        scores = (q[dst] * k[src]).sum(dim=-1) * self.scale
        if edge_weight is not None and edge_weight.numel():
            scores = scores + self.edge_proj(edge_weight.unsqueeze(-1)).squeeze(-1)
        attn = segment_softmax(scores, dst, x.size(0))
        message = v[src] * attn.unsqueeze(-1)
        out = torch.zeros_like(v)
        out.index_add_(0, dst, message)
        x = self.norm1(x + self.dropout(out) + self.self_proj(x))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MultiViewMSRHGNN(nn.Module):
    def __init__(
        self,
        drug_dim: int,
        disease_dim: int,
        gene_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.disease_proj = nn.Linear(disease_dim, hidden_dim)
        self.gene_proj = nn.Linear(gene_dim, hidden_dim)

        self.drug_sim_layers = nn.ModuleList([WeightedGINLayer(hidden_dim) for _ in SIM_DRUG_RELATIONS])
        self.disease_sim_layers = nn.ModuleList([WeightedGINLayer(hidden_dim) for _ in SIM_DISEASE_RELATIONS])
        self.drug_sim_fusion = MultiPoolAdaptiveFusion(hidden_dim)
        self.disease_sim_fusion = MultiPoolAdaptiveFusion(hidden_dim)
        self.drug_sim_transformer = SparseGraphTransformerLayer(hidden_dim, dropout)
        self.disease_sim_transformer = SparseGraphTransformerLayer(hidden_dim, dropout)

        self.gene_transformer = SparseGraphTransformerLayer(hidden_dim, dropout)

        self.drug_local_gene = nn.Linear(hidden_dim, hidden_dim)
        self.drug_local_disease = nn.Linear(hidden_dim, hidden_dim)
        self.disease_local_gene = nn.Linear(hidden_dim, hidden_dim)
        self.disease_local_drug = nn.Linear(hidden_dim, hidden_dim)
        self.local_drug_fusion = MultiPoolAdaptiveFusion(hidden_dim)
        self.local_disease_fusion = MultiPoolAdaptiveFusion(hidden_dim)

        self.drug_global_relation = nn.Linear(hidden_dim, hidden_dim)
        self.disease_global_drug = nn.Linear(hidden_dim, hidden_dim)
        self.disease_global_shared = nn.Linear(hidden_dim, hidden_dim)
        self.disease_global_interact = nn.Linear(hidden_dim, hidden_dim)
        self.global_drug_fusion = DualAttentionFusion(hidden_dim, 1)
        self.global_disease_fusion = DualAttentionFusion(hidden_dim, 3)

        self.drug_hetero_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.disease_hetero_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.final_drug_norm = nn.LayerNorm(hidden_dim * 2)
        self.final_disease_norm = nn.LayerNorm(hidden_dim * 2)
        self.bilinear_weight = nn.Parameter(torch.empty(hidden_dim * 2, hidden_dim * 2))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.bilinear_weight)

    def encode(
        self,
        drug_x: torch.Tensor,
        disease_x: torch.Tensor,
        gene_x: torch.Tensor,
        relation_edges: Dict[str, torch.Tensor],
        relation_weights: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
        drug_base = self.dropout(F.relu(self.drug_proj(drug_x)))
        disease_base = self.dropout(F.relu(self.disease_proj(disease_x)))
        gene_base = self.dropout(F.relu(self.gene_proj(gene_x)))

        drug_sim_views: List[torch.Tensor] = []
        for layer, relation in zip(self.drug_sim_layers, SIM_DRUG_RELATIONS):
            if relation in relation_edges:
                view = self.dropout(F.relu(layer(drug_base, relation_edges[relation], relation_weights.get(relation))))
                drug_sim_views.append(view)
        disease_sim_views: List[torch.Tensor] = []
        for layer, relation in zip(self.disease_sim_layers, SIM_DISEASE_RELATIONS):
            if relation in relation_edges:
                view = self.dropout(F.relu(layer(disease_base, relation_edges[relation], relation_weights.get(relation))))
                disease_sim_views.append(view)

        drug_sim_fused, drug_sim_weights = self.drug_sim_fusion(drug_sim_views if drug_sim_views else [drug_base])
        disease_sim_fused, disease_sim_weights = self.disease_sim_fusion(disease_sim_views if disease_sim_views else [disease_base])

        available_drug_relations = [r for r in SIM_DRUG_RELATIONS if r in relation_edges]
        available_disease_relations = [r for r in SIM_DISEASE_RELATIONS if r in relation_edges]
        drug_merge_weights = drug_sim_weights if available_drug_relations else torch.ones((1,), device=drug_base.device, dtype=drug_base.dtype)
        disease_merge_weights = disease_sim_weights if available_disease_relations else torch.ones((1,), device=disease_base.device, dtype=disease_base.dtype)
        merged_drug_edges, merged_drug_weights = merge_relation_graphs(available_drug_relations, relation_edges, relation_weights, drug_merge_weights, drug_base.size(0))
        merged_disease_edges, merged_disease_weights = merge_relation_graphs(available_disease_relations, relation_edges, relation_weights, disease_merge_weights, disease_base.size(0))
        drug_sim = self.drug_sim_transformer(drug_sim_fused, merged_drug_edges, merged_drug_weights)
        disease_sim = self.disease_sim_transformer(disease_sim_fused, merged_disease_edges, merged_disease_weights)

        gene_ctx = gene_base
        if GENE_NETWORK_RELATION in relation_edges:
            gene_ctx = self.gene_transformer(
                gene_base,
                relation_edges[GENE_NETWORK_RELATION],
                relation_weights.get(GENE_NETWORK_RELATION),
            )

        drug_local_inputs: List[torch.Tensor] = []
        if LOCAL_DTI_RELATION in relation_edges:
            rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DTI_RELATION], relation_weights.get(LOCAL_DTI_RELATION))
            agg = weighted_mean_aggregate(gene_ctx, rev_edges, drug_base.size(0), rev_weights)
            drug_local_inputs.append(self.dropout(F.relu(self.drug_local_gene(agg))))
        if LOCAL_DD_TRAIN_RELATION in relation_edges and relation_edges[LOCAL_DD_TRAIN_RELATION].numel() > 0:
            rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DD_TRAIN_RELATION], relation_weights.get(LOCAL_DD_TRAIN_RELATION))
            agg = weighted_mean_aggregate(disease_base, rev_edges, drug_base.size(0), rev_weights)
            drug_local_inputs.append(self.dropout(F.relu(self.drug_local_disease(agg))))
        drug_local, drug_local_weights = self.local_drug_fusion(drug_local_inputs if drug_local_inputs else [drug_base])

        disease_local_inputs: List[torch.Tensor] = []
        if LOCAL_DG_RELATION in relation_edges:
            rev_edges, rev_weights = reverse_edges(relation_edges[LOCAL_DG_RELATION], relation_weights.get(LOCAL_DG_RELATION))
            agg = weighted_mean_aggregate(gene_ctx, rev_edges, disease_base.size(0), rev_weights)
            disease_local_inputs.append(self.dropout(F.relu(self.disease_local_gene(agg))))
        if LOCAL_DD_TRAIN_RELATION in relation_edges and relation_edges[LOCAL_DD_TRAIN_RELATION].numel() > 0:
            agg = weighted_mean_aggregate(drug_base, relation_edges[LOCAL_DD_TRAIN_RELATION], disease_base.size(0), relation_weights.get(LOCAL_DD_TRAIN_RELATION))
            disease_local_inputs.append(self.dropout(F.relu(self.disease_local_drug(agg))))
        disease_local, disease_local_weights = self.local_disease_fusion(disease_local_inputs if disease_local_inputs else [disease_base])

        drug_global_inputs: List[torch.Tensor] = []
        if META_DRUG_DISEASE_RELATION in relation_edges:
            rev_edges, rev_weights = reverse_edges(relation_edges[META_DRUG_DISEASE_RELATION], relation_weights.get(META_DRUG_DISEASE_RELATION))
            agg = weighted_mean_aggregate(disease_base, rev_edges, drug_base.size(0), rev_weights)
            drug_global_inputs.append(self.dropout(F.relu(self.drug_global_relation(agg))))
        drug_global, drug_global_weights = self.global_drug_fusion(drug_base, drug_global_inputs if drug_global_inputs else [drug_local])

        disease_global_inputs: List[torch.Tensor] = []
        if META_DRUG_DISEASE_RELATION in relation_edges:
            agg = weighted_mean_aggregate(drug_base, relation_edges[META_DRUG_DISEASE_RELATION], disease_base.size(0), relation_weights.get(META_DRUG_DISEASE_RELATION))
            disease_global_inputs.append(self.dropout(F.relu(self.disease_global_drug(agg))))
        if META_DISEASE_SHARED_RELATION in relation_edges:
            agg = weighted_mean_aggregate(disease_base, relation_edges[META_DISEASE_SHARED_RELATION], disease_base.size(0), relation_weights.get(META_DISEASE_SHARED_RELATION))
            disease_global_inputs.append(self.dropout(F.relu(self.disease_global_shared(agg))))
        if META_DISEASE_INTERACT_RELATION in relation_edges:
            agg = weighted_mean_aggregate(disease_base, relation_edges[META_DISEASE_INTERACT_RELATION], disease_base.size(0), relation_weights.get(META_DISEASE_INTERACT_RELATION))
            disease_global_inputs.append(self.dropout(F.relu(self.disease_global_interact(agg))))
        disease_global, disease_global_weights = self.global_disease_fusion(disease_base, disease_global_inputs if disease_global_inputs else [disease_local])

        drug_gate = torch.sigmoid(self.drug_hetero_gate(torch.cat([drug_local, drug_global], dim=-1)))
        disease_gate = torch.sigmoid(self.disease_hetero_gate(torch.cat([disease_local, disease_global], dim=-1)))
        drug_hetero = drug_gate * drug_local + (1.0 - drug_gate) * drug_global
        disease_hetero = disease_gate * disease_local + (1.0 - disease_gate) * disease_global

        drug_repr = self.final_drug_norm(torch.cat([drug_sim, drug_hetero], dim=-1))
        disease_repr = self.final_disease_norm(torch.cat([disease_sim, disease_hetero], dim=-1))

        aux = {
            "drug_sim_weights": [float(x) for x in drug_sim_weights.detach().cpu().tolist()],
            "disease_sim_weights": [float(x) for x in disease_sim_weights.detach().cpu().tolist()],
            "drug_local_weights": [float(x) for x in drug_local_weights.detach().cpu().tolist()],
            "disease_local_weights": [float(x) for x in disease_local_weights.detach().cpu().tolist()],
            "drug_global_weights": [float(x) for x in drug_global_weights.detach().cpu().tolist()],
            "disease_global_weights": [float(x) for x in disease_global_weights.detach().cpu().tolist()],
        }
        return drug_repr, disease_repr, aux

    def score_pairs(self, drug_repr: torch.Tensor, disease_repr: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.empty((0,), device=drug_repr.device)
        d = drug_repr[pairs[:, 0]]
        t = disease_repr[pairs[:, 1]]
        return torch.sum((d @ self.bilinear_weight) * t, dim=-1) + self.bias
