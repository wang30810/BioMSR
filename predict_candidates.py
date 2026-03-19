from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from msrhgnn_model import LOCAL_DD_TRAIN_RELATION, MultiViewMSRHGNN, build_metapath_relations
from train_model import (
    DRUGSIM_DRSIE_RELATION,
    DRUGSIM_GIP_RELATION,
    build_train_only_drsie_relation,
    build_train_only_gip_relation,
)


def load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pairs_to_edge_index(pairs: Sequence[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def load_graph(graph_path: Path) -> Dict:
    return torch.load(graph_path, map_location='cpu')


def build_inference_relations(graph: Dict, checkpoint: Dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    relation_edges = {k: v.long().cpu() for k, v in graph['edge_index'].items()}
    raw_edge_weight = graph.get('edge_weight', {}) or {}
    relation_weights = {
        k: raw_edge_weight[k].float().cpu() if k in raw_edge_weight else torch.ones((v.size(1),), dtype=torch.float)
        for k, v in relation_edges.items()
    }

    train_pairs = [tuple(x) for x in checkpoint.get('splits', {}).get('train_pos', [])]
    config = checkpoint.get('config', {})
    drug_sim_top_k = int(config.get('drug_sim_top_k', 5))
    meta_top_k = int(config.get('meta_top_k', 10))

    num_drugs = len(graph['node_ids']['drug'])
    num_diseases = len(graph['node_ids']['disease'])

    train_gip_edge_index, train_gip_edge_weight = build_train_only_gip_relation(
        num_drugs=num_drugs,
        num_diseases=num_diseases,
        train_pairs=train_pairs,
        top_k=drug_sim_top_k,
    )
    train_drsie_edge_index, train_drsie_edge_weight = build_train_only_drsie_relation(
        num_drugs=num_drugs,
        train_pairs=train_pairs,
        top_k=drug_sim_top_k,
    )
    relation_edges[DRUGSIM_GIP_RELATION] = train_gip_edge_index
    relation_weights[DRUGSIM_GIP_RELATION] = train_gip_edge_weight
    relation_edges[DRUGSIM_DRSIE_RELATION] = train_drsie_edge_index
    relation_weights[DRUGSIM_DRSIE_RELATION] = train_drsie_edge_weight

    meta_edges, meta_weights = build_metapath_relations(
        relation_edges,
        relation_weights,
        num_drugs=num_drugs,
        num_diseases=num_diseases,
        top_k=meta_top_k,
    )
    relation_edges.update(meta_edges)
    relation_weights.update(meta_weights)

    relation_edges[LOCAL_DD_TRAIN_RELATION] = pairs_to_edge_index(train_pairs)
    relation_weights[LOCAL_DD_TRAIN_RELATION] = torch.ones((len(train_pairs),), dtype=torch.float)
    return relation_edges, relation_weights


def disease_name_map(graph: Dict, drug_disease_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if drug_disease_csv.exists():
        df = pd.read_csv(drug_disease_csv)
        for _, row in df[['DiseaseID', 'DiseaseName']].drop_duplicates().iterrows():
            mapping[str(row['DiseaseID'])] = str(row['DiseaseName'])
    for did in graph['node_ids']['disease']:
        mapping.setdefault(str(did), str(did))
    return mapping


def find_disease_id(graph: Dict, drug_disease_csv: Path, query: str) -> str:
    query_lower = query.strip().lower()
    if query in graph['node_ids']['disease']:
        return query

    name_map = disease_name_map(graph, drug_disease_csv)
    exact = [did for did, name in name_map.items() if name.lower() == query_lower]
    if len(exact) == 1:
        return exact[0]

    fuzzy = [did for did, name in name_map.items() if query_lower in name.lower() or query_lower in did.lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if not fuzzy:
        raise ValueError(f'Disease not found: {query}')
    raise ValueError('Disease is ambiguous. Use a more specific name or DiseaseID: ' + ', '.join(f'{did} ({name_map[did]})' for did in fuzzy[:10]))


def load_drug_catalog(path: Path) -> Dict[str, Dict]:
    return load_json(path)


def known_pairs_set(graph: Dict) -> set[Tuple[int, int]]:
    edge_index = graph['edge_index']['drug__treats__disease']
    if edge_index.numel() == 0:
        return set()
    return set(map(tuple, edge_index.t().cpu().tolist()))


def build_model(graph: Dict, checkpoint: Dict) -> MultiViewMSRHGNN:
    config = checkpoint['config']
    model = MultiViewMSRHGNN(
        drug_dim=graph['node_features']['drug'].size(1),
        disease_dim=graph['node_features']['disease'].size(1),
        gene_dim=graph['node_features']['gene'].size(1),
        hidden_dim=int(config.get('hidden_dim', 128)),
        dropout=float(config.get('dropout', 0.2)),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_for_disease(
    model: MultiViewMSRHGNN,
    graph: Dict,
    relation_edges: Dict[str, torch.Tensor],
    relation_weights: Dict[str, torch.Tensor],
    disease_id: str,
    drug_catalog: Dict[str, Dict],
    disease_names: Dict[str, str],
    top_k: int,
    exclude_known: bool,
) -> pd.DataFrame:
    drug_ids: List[str] = list(graph['node_ids']['drug'])
    disease_ids: List[str] = list(graph['node_ids']['disease'])
    disease_idx = disease_ids.index(disease_id)

    known_pairs = known_pairs_set(graph) if exclude_known else set()

    with torch.no_grad():
        drug_x = graph['node_features']['drug'].float()
        disease_x = graph['node_features']['disease'].float()
        gene_x = graph['node_features']['gene'].float()
        drug_repr, disease_repr, _ = model.encode(drug_x, disease_x, gene_x, relation_edges, relation_weights)

        rows = []
        for drug_idx, drug_id in enumerate(drug_ids):
            if (drug_idx, disease_idx) in known_pairs:
                continue
            pair_tensor = torch.tensor([[drug_idx, disease_idx]], dtype=torch.long)
            logit = model.score_pairs(drug_repr, disease_repr, pair_tensor)[0].item()
            score = torch.sigmoid(torch.tensor(logit)).item()
            drug_info = drug_catalog.get(drug_id, {})
            rows.append({
                'rank': 0,
                'drugbank_id': drug_id,
                'drug_name': drug_info.get('name', drug_id),
                'disease_id': disease_id,
                'disease_name': disease_names.get(disease_id, disease_id),
                'score': score,
                'logit': logit,
                'num_target_genes': len(drug_info.get('target_genes', []) or []),
                'target_genes': ';'.join((drug_info.get('target_genes', []) or [])[:20]),
            })

    df = pd.DataFrame(rows).sort_values(['score', 'drugbank_id'], ascending=[False, True]).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df.head(top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description='Output Top-K candidate drugs for a disease.')
    parser.add_argument('--model', type=str, default='artifacts/high_order_status_smoke/model.pt')
    parser.add_argument('--graph', type=str, default='data/final/final_graph_data.pt')
    parser.add_argument('--drug-catalog', type=str, default='data/interim/drug_catalog.json')
    parser.add_argument('--drug-disease-csv', type=str, default='data/processed/drug_disease_edges.csv')
    parser.add_argument('--disease', type=str, required=True, help='DiseaseID or disease name, e.g. MESH:D000544 or Alzheimer Disease')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--include-known', action='store_true', help='Include known drug-disease edges. Default: only novel candidates.')
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()

    graph = load_graph(Path(args.graph))
    checkpoint = torch.load(args.model, map_location='cpu')
    relation_edges, relation_weights = build_inference_relations(graph, checkpoint)
    model = build_model(graph, checkpoint)
    catalog = load_drug_catalog(Path(args.drug_catalog))
    disease_names = disease_name_map(graph, Path(args.drug_disease_csv))
    disease_id = find_disease_id(graph, Path(args.drug_disease_csv), args.disease)

    result = predict_for_disease(
        model=model,
        graph=graph,
        relation_edges=relation_edges,
        relation_weights=relation_weights,
        disease_id=disease_id,
        drug_catalog=catalog,
        disease_names=disease_names,
        top_k=args.top_k,
        exclude_known=not args.include_known,
    )

    display_cols = ['rank', 'drugbank_id', 'drug_name', 'disease_name', 'score', 'num_target_genes']
    print(result[display_cols].to_string(index=False))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
