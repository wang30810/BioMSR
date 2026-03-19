from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_utils import (
    ProjectPaths,
    build_disease_features,
    build_disease_gene_edges_if_available,
    build_disease_similarity_g,
    build_disease_similarity_h,
    build_disease_similarity_o,
    build_disease_texts,
    build_drug_catalog,
    build_drug_disease_edges,
    build_drug_features,
    build_drug_similarity_drsie,
    build_drug_similarity_gip,
    build_drug_similarity_morgan,
    build_dti_edges,
    build_final_graph,
    build_gene_features,
    build_gene_network_edges,
    discover_sources,
    integrate_reference_pt_results,
    integrate_reference_processed_results,
    validate_outputs,
    write_json,
)

DEFAULT_FOCUS_KEYWORDS = [
    "alzheimer",
    "parkinson",
    "schizophrenia",
    "depress",
    "bipolar",
    "dementia",
    "anxiety",
    "neurodegenerat",
    "huntington",
    "epilepsy",
]


def require_path(sources: dict, key: str) -> Path:
    value = sources.get(key)
    if not value:
        raise FileNotFoundError(f"Missing required source: {key}")
    return Path(value)


def outputs_exist(*paths: Path) -> bool:
    return all(path.exists() for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed files and final heterogeneous graph for the current project.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k neighbors kept in each similarity matrix.")
    parser.add_argument("--all-diseases", action="store_true", help="Use all diseases instead of only nervous-system-related targets.")
    parser.add_argument("--strict", action="store_true", help="Require optional OMIM / STRING / disease-gene sources.")
    parser.add_argument("--force", action="store_true", help="Rebuild outputs even if processed files already exist.")
    parser.add_argument("--skip-gene-network", action="store_true", help="Skip building STRING/HumanNet gene network edges.")
    parser.add_argument("--no-reference-merge", action="store_true", help="Do not merge processed results from MSRHGNN-Graph-Builder-main.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    paths = ProjectPaths(root)
    paths.ensure()

    sources = discover_sources(root)
    write_json(paths.reports / "discovered_sources.json", sources)

    missing_optional = [name for name in ["omim_texts", "disease_gene_edges"] if not sources.get(name)]
    if not (sources.get("string_links") or sources.get("humannet")):
        missing_optional.append("string_links_or_humannet")
    if args.strict and missing_optional:
        raise FileNotFoundError("Strict mode requires optional sources, but these are missing: " + ", ".join(missing_optional))

    focus_keywords = None if args.all_diseases else DEFAULT_FOCUS_KEYWORDS

    print("\n[1/10] Build drug catalog from DrugBank")
    if args.force or not outputs_exist(paths.interim / "drug_catalog.json"):
        build_drug_catalog(require_path(sources, "drugbank_xml"), paths.interim / "drug_catalog.json")
    else:
        print("  skip: drug_catalog.json already exists")

    print("\n[2/10] Build drug features")
    if args.force or not outputs_exist(paths.processed / "drug_features_dict.json"):
        build_drug_features(paths.interim / "drug_catalog.json", paths.processed / "drug_features_dict.json")
    else:
        print("  skip: drug_features_dict.json already exists")

    print("\n[3/10] Build drug-disease edges")
    if args.force or not outputs_exist(paths.processed / "drug_disease_edges.csv", paths.interim / "target_disease_ids.txt"):
        build_drug_disease_edges(
            require_path(sources, "ctd_chemicals_diseases"),
            paths.interim / "drug_catalog.json",
            paths.processed / "drug_disease_edges.csv",
            paths.interim / "target_disease_ids.txt",
            focus_keywords=focus_keywords,
        )
    else:
        print("  skip: drug_disease_edges.csv already exists")

    print("\n[4/10] Build disease texts and disease features")
    if args.force or not outputs_exist(paths.processed / "disease_texts.json", paths.processed / "disease_texts_meta.json"):
        build_disease_texts(
            paths.interim / "target_disease_ids.txt",
            require_path(sources, "ctd_diseases"),
            paths.processed / "disease_texts.json",
            paths.processed / "disease_texts_meta.json",
            Path(sources["omim_texts"]) if sources.get("omim_texts") else None,
        )
    else:
        print("  skip: disease_texts.json already exists")
    if args.force or not outputs_exist(paths.processed / "disease_features_dict.json", paths.reports / "disease_encoder_report.json"):
        build_disease_features(
            paths.processed / "disease_texts.json",
            paths.processed / "disease_features_dict.json",
            paths.reports / "disease_encoder_report.json",
            Path(sources["biobert_dir"]) if sources.get("biobert_dir") else None,
        )
    else:
        print("  skip: disease_features_dict.json already exists")

    print("\n[5/10] Build gene features")
    if args.force or not outputs_exist(paths.processed / "gene_features_dict.json"):
        build_gene_features(require_path(sources, "gene2vec"), paths.processed / "gene_features_dict.json")
    else:
        print("  skip: gene_features_dict.json already exists")

    print("\n[6/10] Build hetero relations")
    if args.force or not outputs_exist(paths.processed / "dti_edges.csv"):
        build_dti_edges(paths.interim / "drug_catalog.json", paths.processed / "dti_edges.csv")
    else:
        print("  skip: dti_edges.csv already exists")

    disease_gene_source = sources.get("disease_gene_edges") or sources.get("ctd_gene_disease")
    if disease_gene_source:
        if args.force or not outputs_exist(paths.processed / "disease_gene_edges.csv"):
            build_disease_gene_edges_if_available(Path(disease_gene_source), paths.processed / "disease_gene_edges.csv")
        else:
            print("  skip: disease_gene_edges.csv already exists")
    else:
        print("  warn: no disease-gene source detected")

    if args.skip_gene_network:
        print("  skip: gene network construction disabled by --skip-gene-network")
    elif sources.get("string_links") or sources.get("humannet"):
        if args.force or not outputs_exist(paths.processed / "gene_network_edges.csv"):
            build_gene_network_edges(
                save_path=paths.processed / "gene_network_edges.csv",
                gene_features_path=paths.processed / "gene_features_dict.json",
                string_links_path=Path(sources["string_links"]) if sources.get("string_links") else None,
                string_alias_path=Path(sources["string_aliases"]) if sources.get("string_aliases") else None,
                humannet_path=Path(sources["humannet"]) if sources.get("humannet") else None,
                gene_info_path=Path(sources["gene_info"]) if sources.get("gene_info") else None,
            )
        else:
            print("  skip: gene_network_edges.csv already exists")
    else:
        print("  warn: no STRING/HumanNet source detected")

    print("\n[7/10] Build drug similarity views")
    if args.force or not outputs_exist(paths.processed / "DrugSim_Morgan.json"):
        build_drug_similarity_morgan(paths.processed / "drug_features_dict.json", paths.processed / "DrugSim_Morgan.json", args.top_k)
    else:
        print("  skip: DrugSim_Morgan.json already exists")
    if args.force or not outputs_exist(paths.processed / "DrugSim_GIP.json"):
        build_drug_similarity_gip(
            paths.processed / "drug_features_dict.json",
            paths.processed / "drug_disease_edges.csv",
            paths.processed / "DrugSim_GIP.json",
            args.top_k,
        )
    else:
        print("  skip: DrugSim_GIP.json already exists")
    if args.force or not outputs_exist(paths.processed / "DrugSim_DRSIE.json"):
        build_drug_similarity_drsie(
            paths.processed / "drug_features_dict.json",
            paths.processed / "drug_disease_edges.csv",
            paths.processed / "DrugSim_DRSIE.json",
            args.top_k,
        )
    else:
        print("  skip: DrugSim_DRSIE.json already exists")

    print("\n[8/10] Build disease similarity views")
    if args.force or not outputs_exist(paths.processed / "DiSimNet_O.json"):
        build_disease_similarity_o(
            paths.processed / "disease_features_dict.json",
            paths.processed / "DiSimNet_O.json",
            args.top_k,
        )
    else:
        print("  skip: DiSimNet_O.json already exists")
    if args.force or not outputs_exist(paths.processed / "DiSimNet_H.json"):
        build_disease_similarity_h(
            require_path(sources, "phenotype_hpoa"),
            require_path(sources, "ctd_diseases"),
            paths.interim / "target_disease_ids.txt",
            paths.processed / "DiSimNet_H.json",
            args.top_k,
        )
    else:
        print("  skip: DiSimNet_H.json already exists")
    if args.force or not outputs_exist(paths.processed / "DiSimNet_G.json"):
        build_disease_similarity_g(
            paths.interim / "target_disease_ids.txt",
            paths.processed / "drug_disease_edges.csv",
            paths.processed / "DiSimNet_G.json",
            args.top_k,
            disease_gene_edges_path=paths.processed / "disease_gene_edges.csv" if (paths.processed / "disease_gene_edges.csv").exists() else None,
            gene_network_edges_path=paths.processed / "gene_network_edges.csv" if (paths.processed / "gene_network_edges.csv").exists() else None,
        )
    else:
        print("  skip: DiSimNet_G.json already exists")

    print("\n[8.5/10] Merge reference processed results")
    reference_report = {}
    reference_pt_report = {}
    if args.no_reference_merge:
        print("  skip: reference processed merge disabled by --no-reference-merge")
    else:
        reference_pt_report = integrate_reference_pt_results(paths, root, top_k=args.top_k)
        if reference_pt_report.get("reference_pt_found"):
            print("  checked reference pt results")
        reference_report = integrate_reference_processed_results(paths, root, top_k=args.top_k)
        if reference_report.get("reference_found"):
            print("  merged reference processed results")
        else:
            print("  no reference processed results found")

    print("\n[9/10] Build final heterogeneous graph")
    need_rebuild_final = args.force or bool(reference_report.get("integrated")) or not outputs_exist(paths.final / "final_graph_data.pt")
    if need_rebuild_final:
        build_final_graph(paths)
    else:
        print("  skip: final_graph_data.pt already exists")

    print("\n[10/10] Validate outputs")
    report = validate_outputs(paths)
    summary = {
        "processed_dir": str(paths.processed),
        "final_dir": str(paths.final),
        "reports_dir": str(paths.reports),
        "warnings": report.get("warnings", []),
        "reference_merge": reference_report,
        "reference_pt_merge": reference_pt_report,
    }
    write_json(paths.reports / "run_summary.json", summary)

    print("\nDone")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nKey outputs")
    print(f"- processed: {paths.processed}")
    print(f"- graph: {paths.final / 'final_graph_data.pt'}")
    print(f"- validation: {paths.reports / 'validation_report.json'}")


if __name__ == "__main__":
    main()
