from __future__ import annotations

import gzip
import io
import json
import math
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXCLUDED_DIRS = {
    "__MACOSX",
    ".git",
    ".ipynb_checkpoints",
    "__pycache__",
    ".venv",
    "venv",
}

RDLogger.DisableLog("rdApp.*")


@dataclass
class ProjectPaths:
    root: Path

    def __post_init__(self) -> None:
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.interim = self.data / "interim"
        self.processed = self.data / "processed"
        self.final = self.data / "final"
        self.reports = self.data / "reports"

    def ensure(self) -> None:
        for path in [
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.final,
            self.reports,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_dict(path: Path) -> Dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def discover_reference_processed(root: Path) -> Dict[str, Optional[str]]:
    base = root / "data" / "reference_graph_builder"
    found = {
        "base_dir": str(base) if base.exists() else None,
        "processed_dir": None,
        "processed_zip": None,
        "final_hetero_pt": None,
    }
    processed_dir = base / "processed"
    processed_zip = base / "processed.zip"
    final_hetero_pt = base / "final_hetero_data_raw.pt"
    if processed_dir.exists():
        found["processed_dir"] = str(processed_dir)
    if processed_zip.exists():
        found["processed_zip"] = str(processed_zip)
    if final_hetero_pt.exists():
        found["final_hetero_pt"] = str(final_hetero_pt)
    return found


def _reference_has_asset(reference: Dict[str, Optional[str]], name: str) -> bool:
    if reference.get("processed_dir"):
        if (Path(reference["processed_dir"]) / name).exists():
            return True
    if reference.get("processed_zip"):
        with zipfile.ZipFile(reference["processed_zip"]) as zf:
            return f"processed/{name}" in zf.namelist()
    return False


def _read_reference_csv(reference: Dict[str, Optional[str]], name: str, **kwargs: Any) -> pd.DataFrame:
    if reference.get("processed_dir"):
        path = Path(reference["processed_dir"]) / name
        if path.exists():
            return pd.read_csv(path, **kwargs)
    if reference.get("processed_zip"):
        with zipfile.ZipFile(reference["processed_zip"]) as zf:
            member = f"processed/{name}"
            if member in zf.namelist():
                with zf.open(member) as f:
                    return pd.read_csv(f, **kwargs)
    raise FileNotFoundError(f"Reference processed asset not found: {name}")


def _read_reference_npy(reference: Dict[str, Optional[str]], name: str) -> np.ndarray:
    if reference.get("processed_dir"):
        path = Path(reference["processed_dir"]) / name
        if path.exists():
            return np.load(path)
    if reference.get("processed_zip"):
        with zipfile.ZipFile(reference["processed_zip"]) as zf:
            member = f"processed/{name}"
            if member in zf.namelist():
                with zf.open(member) as f:
                    return np.load(io.BytesIO(f.read()))
    raise FileNotFoundError(f"Reference processed asset not found: {name}")


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
        if k >= n:
            idx = np.where(np.isfinite(row))[0]
        else:
            idx = np.argpartition(row, -k)[-k:]
        idx = [j for j in idx if j != i and np.isfinite(row[j]) and row[j] > 0]
        for j in idx:
            sparse[i, j] = matrix[i, j]
    if symmetric:
        sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def save_similarity(
    ids: List[str],
    matrix: np.ndarray,
    save_path: Path,
    top_k: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "ids": ids,
        "matrix": matrix.tolist(),
        "top_k": top_k,
        "metadata": metadata or {},
    }
    write_json(save_path, payload)


def align_reference_similarity(target_ids: List[str], reference_ids: List[str], reference_matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    aligned = np.zeros((len(target_ids), len(target_ids)), dtype=float)
    if reference_matrix.size == 0 or not target_ids or not reference_ids:
        return aligned, 0
    ref_index = {str(did): i for i, did in enumerate(reference_ids)}
    matched = [i for i, did in enumerate(target_ids) if str(did) in ref_index]
    for i in matched:
        ri = ref_index[str(target_ids[i])]
        for j in matched:
            rj = ref_index[str(target_ids[j])]
            aligned[i, j] = float(reference_matrix[ri, rj])
    return aligned, len(matched)


def merge_similarity_payload(
    ids: List[str],
    save_path: Path,
    top_k: int,
    reference_matrix: np.ndarray,
    metadata_updates: Dict[str, Any],
) -> Dict[str, Any]:
    current_matrix = np.zeros_like(reference_matrix)
    current_metadata: Dict[str, Any] = {}
    if save_path.exists():
        payload = read_json(save_path)
        payload_ids = [str(x) for x in payload.get("ids", ids)]
        payload_matrix = np.array(payload.get("matrix", []), dtype=float)
        if payload_matrix.shape == reference_matrix.shape and payload_ids == ids:
            current_matrix = payload_matrix
        current_metadata = payload.get("metadata", {})
    merged = np.maximum(current_matrix, reference_matrix)
    merged = normalize_matrix(merged)
    merged = sparsify_top_k(merged, k=top_k, symmetric=True)
    metadata = dict(current_metadata)
    metadata.update(metadata_updates)
    save_similarity(ids, merged, save_path, top_k, metadata)
    return metadata


def similarity_from_vectors(
    ids: List[str],
    vectors: np.ndarray,
    save_path: Path,
    top_k: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if len(ids) == 0:
        save_similarity([], np.zeros((0, 0)), save_path, top_k, metadata)
        return
    sim = cosine_similarity(vectors)
    sim = normalize_matrix(sim)
    sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
    save_similarity(ids, sparse, save_path, top_k, metadata)


def discover_sources(root: Path) -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {
        "drugbank_xml": None,
        "ctd_chemicals_diseases": None,
        "ctd_diseases": None,
        "ctd_gene_disease": None,
        "phenotype_hpoa": None,
        "gene2vec": None,
        "gene_info": None,
        "biobert_dir": None,
        "omim_texts": None,
        "humannet": None,
        "string_links": None,
        "string_aliases": None,
        "disease_gene_edges": None,
    }

    candidate_dirs: List[Path] = []
    for path in iter_files(root):
        lower = path.name.lower()
        full = str(path)
        if found["ctd_chemicals_diseases"] is None and lower == "ctd_chemicals_diseases.csv":
            found["ctd_chemicals_diseases"] = full
        elif found["ctd_diseases"] is None and lower == "ctd_diseases.csv":
            found["ctd_diseases"] = full
        elif found["phenotype_hpoa"] is None and lower == "phenotype.hpoa":
            found["phenotype_hpoa"] = full
        elif found["gene2vec"] is None and lower == "gene2vec_dim_200_iter_9_w2v.txt":
            found["gene2vec"] = full
        elif found["gene_info"] is None and lower == "gene_info":
            found["gene_info"] = full
        elif found["drugbank_xml"] is None and lower.endswith(".xml") and "full database" in lower:
            found["drugbank_xml"] = full
        elif found["omim_texts"] is None and "omim" in lower and path.suffix.lower() in {".json", ".csv", ".tsv", ".txt"}:
            found["omim_texts"] = full
        elif found["humannet"] is None and "humannet" in lower and path.suffix.lower() in {".csv", ".tsv", ".txt", ".gz"}:
            found["humannet"] = full
        elif found["string_links"] is None and (
            ("protein.links" in lower and "9606" in lower)
            or ("protein.physical.links" in lower)
            or (lower in {"protein.links.v12.0.txt.gz", "protein.physical.links.v12.0.txt.gz"})
        ) and path.suffix.lower() in {".txt", ".gz"}:
            found["string_links"] = full
        elif found["string_aliases"] is None and (
            ("protein.aliases" in lower and "9606" in lower)
            or (lower == "protein.aliases.v12.0.txt.gz")
            or ("protein.aliases" in lower)
        ) and path.suffix.lower() in {".txt", ".gz"}:
            found["string_aliases"] = full
        elif found["ctd_gene_disease"] is None and "genes_diseases" in lower and "ctd" in lower and path.suffix.lower() in {".csv", ".tsv", ".txt", ".gz"}:
            found["ctd_gene_disease"] = full
        elif found["disease_gene_edges"] is None and (
            lower in {"disease_gene_edges.csv", "disease_gene_edges.tsv"}
            or "ctd_genes_diseases" in lower
            or ("disease" in lower and "gene" in lower and path.suffix.lower() in {".csv", ".tsv", ".txt", ".gz"})
        ):
            found["disease_gene_edges"] = full
        candidate_dirs.append(path.parent)

    for directory in dict.fromkeys(candidate_dirs):
        if found["biobert_dir"] is None:
            names = {p.name for p in directory.iterdir()} if directory.exists() else set()
            if {"config.json", "vocab.txt", "tokenizer_config.json"}.issubset(names):
                found["biobert_dir"] = str(directory)

    return found


def load_ctd_csv(path: Path) -> pd.DataFrame:
    columns: Optional[List[str]] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("#") or "," not in stripped or "Fields:" in stripped:
                continue
            if "ChemicalName" not in stripped and "DiseaseName" not in stripped:
                continue
            if "DirectEvidence" not in stripped and "Definition" not in stripped:
                continue
            if stripped.startswith("#") and "," in stripped:
                body = stripped.lstrip("#").strip()
                parts = [x.strip() for x in body.split(",")]
                if len(parts) >= 2:
                    columns = parts
                    break
    if not columns:
        raise ValueError(f"Unable to detect CTD header in {path}")
    return pd.read_csv(path, comment="#", header=None, names=columns, low_memory=False)


def extract_omim_ids(alt_ids: Any) -> List[str]:
    if pd.isna(alt_ids):
        return []
    items = []
    for token in str(alt_ids).split("|"):
        token = token.strip()
        if token.upper().startswith("OMIM:"):
            items.append(token)
    return items


def load_target_ids(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def read_table_auto(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    compression = "gzip" if ".gz" in suffixes else None
    sep = kwargs.pop("sep", None)
    engine = kwargs.pop("engine", None)
    if sep is None:
        if any(ext in suffixes for ext in {".tsv", ".txt"}):
            sep = "\t"
        elif path.name.lower() in {"gene_info", "gene_info.txt"}:
            sep = "\t"
        else:
            sep = ","
    if engine is None and (sep is None or sep == r"\s+"):
        engine = "python"
    return pd.read_csv(path, sep=sep, compression=compression, low_memory=False, engine=engine, **kwargs)


def load_gene_info_map(path: Optional[Path]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if path is None or not path.exists():
        return {}, {}
    id_to_symbol: Dict[str, str] = {}
    chunk_iter = pd.read_csv(
        path,
        sep="\t",
        header=0,
        usecols=[0, 1, 2],
        low_memory=False,
        chunksize=500_000,
    )
    for chunk in chunk_iter:
        chunk.columns = ["#tax_id", "GeneID", "Symbol"]
        chunk = chunk[chunk["#tax_id"].astype(str) == "9606"].copy()
        if len(chunk) == 0:
            continue
        for gid, sym in zip(chunk["GeneID"], chunk["Symbol"]):
            if pd.isna(gid) or pd.isna(sym):
                continue
            gid_str = str(gid).strip()
            sym_str = str(sym).strip()
            if not gid_str or not sym_str or sym_str == "-":
                continue
            id_to_symbol[gid_str] = sym_str
    symbol_to_id = {symbol: gid for gid, symbol in id_to_symbol.items()}
    return id_to_symbol, symbol_to_id


def load_ctd_gene_disease_edges(
    path: Path,
    gene_universe: Optional[set[str]] = None,
    gene_info_path: Optional[Path] = None,
) -> pd.DataFrame:
    columns = [
        "GeneSymbol",
        "GeneID",
        "DiseaseName",
        "DiseaseID",
        "DirectEvidence",
        "InferenceChemicalName",
        "InferenceScore",
        "OmimIDs",
        "PubMedIDs",
    ]
    df = pd.read_csv(path, compression="gzip" if ".gz" in {s.lower() for s in path.suffixes} else None, comment="#", header=None, names=columns, low_memory=False)
    df = df[df["DiseaseID"].notna()].copy()
    df["DiseaseID"] = df["DiseaseID"].astype(str).str.strip()
    df = df[df["DiseaseID"] != ""]
    df = df[df["DirectEvidence"].astype(str).str.contains("marker|mechanism", case=False, na=False)].copy()

    df["GeneSymbol"] = df["GeneSymbol"].astype(str).str.strip()
    df["GeneID"] = df["GeneID"].astype(str).str.strip()
    if gene_universe:
        df["gene_id"] = np.where(
            df["GeneSymbol"].isin(gene_universe),
            df["GeneSymbol"],
            np.where(
                df["GeneID"].isin(gene_universe),
                df["GeneID"],
                "",
            ),
        )
    else:
        id_to_symbol, _ = load_gene_info_map(gene_info_path)
        mapped_symbol = df["GeneID"].map(id_to_symbol).fillna("")
        df["gene_id"] = np.where(mapped_symbol != "", mapped_symbol, np.where(df["GeneSymbol"] != "", df["GeneSymbol"], df["GeneID"]))
    out = df[["DiseaseID", "gene_id"]].copy()
    out.columns = ["disease_id", "gene_id"]
    out["gene_id"] = out["gene_id"].astype(str).str.strip()
    out = out[(out["gene_id"] != "") & (out["gene_id"] != "-")].drop_duplicates()
    return out


def standardize_disease_gene_edges(
    path: Path,
    gene_universe: Optional[set[str]] = None,
    gene_info_path: Optional[Path] = None,
) -> pd.DataFrame:
    lower = path.name.lower()
    if "ctd" in lower and "genes_diseases" in lower:
        return load_ctd_gene_disease_edges(path, gene_universe=gene_universe, gene_info_path=gene_info_path)

    df = read_table_auto(path)
    disease_col = find_column(df, ["disease_id", "DiseaseID", "database_id", "disease"])
    gene_candidates = ["gene_symbol", "GeneSymbol", "gene_id", "GeneID", "gene", "symbol"]
    gene_col = find_column(df, gene_candidates)
    if disease_col is None or gene_col is None:
        raise ValueError(f"Cannot detect disease/gene columns in {path}")
    out = df[[disease_col, gene_col]].copy()
    out.columns = ["disease_id", "gene_id"]
    out.dropna(inplace=True)
    out["disease_id"] = out["disease_id"].astype(str).str.strip()
    out["gene_id"] = out["gene_id"].astype(str).str.strip()
    if gene_universe:
        id_to_symbol, _ = load_gene_info_map(gene_info_path)
        out["gene_id"] = out["gene_id"].map(lambda x: id_to_symbol.get(x, x))
        out = out[out["gene_id"].isin(gene_universe)].copy()
    out = out[(out["disease_id"] != "") & (out["gene_id"] != "")].drop_duplicates()
    return out


def load_omim_text_source(path: Path) -> Dict[str, str]:
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if str(v).strip()}
        if isinstance(data, list):
            mapping = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                did = row.get("disease_id") or row.get("DiseaseID") or row.get("omim_id") or row.get("OMIM")
                text = row.get("text") or row.get("definition") or row.get("description") or row.get("disease_text")
                if did and text:
                    mapping[str(did)] = str(text)
            return mapping
        raise ValueError(f"Unsupported OMIM json format: {path}")

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    did_col = find_column(df, ["disease_id", "DiseaseID", "omim_id", "OMIM"])
    text_col = find_column(df, ["text", "definition", "description", "disease_text"])
    if did_col is None or text_col is None:
        raise ValueError(f"Cannot detect OMIM text columns in {path}")
    return {
        str(k): str(v)
        for k, v in zip(df[did_col], df[text_col])
        if pd.notna(k) and pd.notna(v) and str(v).strip()
    }


def build_drug_catalog(drugbank_xml: Path, save_path: Path) -> Dict[str, Any]:
    ns = "{http://www.drugbank.ca}"
    drugs: Dict[str, Any] = {}
    context = ET.iterparse(drugbank_xml, events=("end",))
    for _, elem in context:
        if elem.tag != f"{ns}drug":
            continue

        ids = [node.text for node in elem.findall(f"{ns}drugbank-id") if node.text]
        db_id = None
        for node in elem.findall(f"{ns}drugbank-id"):
            if node.attrib.get("primary") == "true" and node.text:
                db_id = node.text
                break
        if db_id is None:
            db_id = next((item for item in ids if str(item).startswith("DB")), None)
        if not db_id:
            elem.clear()
            continue

        groups = [g.text for g in elem.findall(f"{ns}groups/{ns}group") if g.text]
        if "approved" not in groups:
            elem.clear()
            continue

        name_node = elem.find(f"{ns}name")
        name = name_node.text if name_node is not None and name_node.text else db_id

        smiles = ""
        for prop in elem.findall(f"{ns}calculated-properties/{ns}property"):
            kind = prop.find(f"{ns}kind")
            value = prop.find(f"{ns}value")
            if kind is not None and kind.text == "SMILES" and value is not None and value.text:
                smiles = value.text
                break

        pairs: set[Tuple[str, str]] = set()
        for target in elem.findall(f"{ns}targets/{ns}target"):
            poly = target.find(f"{ns}polypeptide")
            if poly is None:
                continue
            organism = poly.find(f"{ns}organism")
            if organism is not None and organism.text and organism.text.lower() != "humans":
                continue
            uniprot_id = poly.attrib.get("id", "")
            gene_name_node = poly.find(f"{ns}gene-name")
            gene_symbol = gene_name_node.text.strip() if gene_name_node is not None and gene_name_node.text else ""
            if uniprot_id or gene_symbol:
                pairs.add((uniprot_id, gene_symbol))

        target_pairs = [{"uniprot_id": a, "gene_symbol": b} for a, b in sorted(pairs)]
        if smiles or target_pairs:
            drugs[db_id] = {
                "name": name,
                "smiles": smiles,
                "target_pairs": target_pairs,
                "targets": [x["uniprot_id"] for x in target_pairs if x["uniprot_id"]],
                "target_genes": [x["gene_symbol"] for x in target_pairs if x["gene_symbol"]],
            }

        elem.clear()

    write_json(save_path, drugs)
    return drugs


def build_drug_features(drug_json_path: Path, save_path: Path) -> Dict[str, List[int]]:
    drugs = load_json_dict(drug_json_path)
    features: Dict[str, List[int]] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    for db_id in sorted(drugs.keys()):
        smiles = drugs[db_id].get("smiles")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((1024,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features[db_id] = arr.astype(int).tolist()
    write_json(save_path, features)
    return features


def build_drug_disease_edges(
    ctd_chemicals_path: Path,
    drug_json_path: Path,
    edges_save_path: Path,
    target_ids_save_path: Path,
    focus_keywords: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    drugs = load_json_dict(drug_json_path)
    name_to_db: Dict[str, List[str]] = {}
    for db_id, info in drugs.items():
        name = str(info.get("name", "")).strip().lower()
        if name:
            name_to_db.setdefault(name, []).append(db_id)

    df = load_ctd_csv(ctd_chemicals_path)
    df = df[df["DirectEvidence"].astype(str).str.contains("therapeutic", case=False, na=False)].copy()
    df["chemical_name_lower"] = df["ChemicalName"].astype(str).str.lower()
    df = df[df["chemical_name_lower"].isin(name_to_db)].copy()
    df["drugbank_id"] = df["chemical_name_lower"].map(lambda x: sorted(name_to_db[x])[0])

    if focus_keywords:
        pattern = "|".join(focus_keywords)
        df = df[df["DiseaseName"].astype(str).str.lower().str.contains(pattern, na=False)].copy()

    edges = df[["drugbank_id", "ChemicalName", "ChemicalID", "DiseaseName", "DiseaseID"]].drop_duplicates().reset_index(drop=True)
    edges.to_csv(edges_save_path, index=False)
    target_ids = sorted(edges["DiseaseID"].dropna().astype(str).unique().tolist())
    with open(target_ids_save_path, "w", encoding="utf-8") as f:
        for did in target_ids:
            f.write(f"{did}\n")
    return edges, target_ids


def build_disease_texts(
    target_ids_path: Path,
    ctd_diseases_path: Path,
    save_path: Path,
    metadata_path: Path,
    omim_text_source: Optional[Path] = None,
) -> Dict[str, str]:
    target_ids = load_target_ids(target_ids_path)
    vocab = load_ctd_csv(ctd_diseases_path)
    vocab = vocab[vocab["DiseaseID"].isin(target_ids)].copy()
    omim_texts: Dict[str, str] = {}
    if omim_text_source and omim_text_source.exists():
        omim_texts = load_omim_text_source(omim_text_source)

    texts: Dict[str, str] = {}
    source_meta: Dict[str, str] = {}
    for _, row in vocab.iterrows():
        did = str(row["DiseaseID"])
        text = None
        source = None
        if omim_texts:
            if did in omim_texts:
                text = omim_texts[did]
                source = "omim_direct"
            else:
                for omim_id in extract_omim_ids(row.get("AltDiseaseIDs")):
                    if omim_id in omim_texts:
                        text = omim_texts[omim_id]
                        source = "omim_alt_id"
                        break
        if not text:
            definition = row.get("Definition")
            if pd.notna(definition) and str(definition).strip():
                text = str(definition)
                source = "ctd_definition"
        if not text:
            text = str(row.get("DiseaseName"))
            source = "disease_name_fallback"
        texts[did] = text
        source_meta[did] = source

    write_json(save_path, texts)
    write_json(
        metadata_path,
        {
            "source_per_disease": source_meta,
            "used_omim_source": str(omim_text_source) if omim_text_source else None,
        },
    )
    return texts


def build_disease_features(
    disease_texts_path: Path,
    save_path: Path,
    encoder_report_path: Path,
    biobert_dir: Optional[Path] = None,
) -> Dict[str, List[float]]:
    disease_texts = load_json_dict(disease_texts_path)
    ids = list(disease_texts.keys())
    texts = [str(disease_texts[k]) for k in ids]

    encoder = "tfidf_svd"
    fallback_reason: Optional[str] = None
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        if not biobert_dir or not biobert_dir.exists():
            raise ModuleNotFoundError("biobert_dir missing")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(biobert_dir))
        model = AutoModel.from_pretrained(str(biobert_dir)).to(device)
        model.eval()
        outputs: List[List[float]] = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                result = model(**inputs)
                hidden = result.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                outputs.append([round(float(x), 6) for x in pooled.squeeze().detach().cpu().tolist()])
        vectors = np.array(outputs, dtype=float)
        encoder = "biobert_mean_pooling"
    except Exception as exc:
        fallback_reason = repr(exc)
        tfidf = TfidfVectorizer(max_features=4096, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(texts)
        if tfidf_matrix.shape[0] > 1 and tfidf_matrix.shape[1] > 1:
            n_components = int(min(256, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1))
            if n_components >= 2:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                vectors = svd.fit_transform(tfidf_matrix)
            else:
                vectors = tfidf_matrix.toarray()
        else:
            vectors = tfidf_matrix.toarray()

    features = {did: [round(float(x), 6) for x in vec] for did, vec in zip(ids, vectors)}
    write_json(save_path, features)
    write_json(
        encoder_report_path,
        {
            "encoder": encoder,
            "dimension": int(vectors.shape[1]) if len(vectors.shape) == 2 else 0,
            "biobert_dir": str(biobert_dir) if biobert_dir else None,
            "fallback_reason": fallback_reason,
        },
    )
    return features


def build_gene_features(gene2vec_path: Path, save_path: Path) -> Dict[str, List[float]]:
    gene_features: Dict[str, List[float]] = {}
    with open(gene2vec_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            parts = line.strip().split()
            if line_no == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                continue
            if len(parts) < 201:
                continue
            gene = parts[0]
            gene_features[gene] = [float(x) for x in parts[1:201]]
    write_json(save_path, gene_features)
    return gene_features


def build_dti_edges(drug_json_path: Path, save_path: Path) -> pd.DataFrame:
    drugs = load_json_dict(drug_json_path)
    rows: List[Dict[str, str]] = []
    for drug_id, info in drugs.items():
        target_pairs = info.get("target_pairs", []) or []
        if target_pairs:
            for pair in target_pairs:
                gene_symbol = str(pair.get("gene_symbol", "")).strip()
                uniprot_id = str(pair.get("uniprot_id", "")).strip()
                target_id = gene_symbol or uniprot_id
                if target_id:
                    rows.append(
                        {
                            "drug_id": drug_id,
                            "target_id": target_id,
                            "target_uniprot_id": uniprot_id,
                        }
                    )
        else:
            for target_id in info.get("target_genes", []) or info.get("targets", []):
                rows.append({"drug_id": drug_id, "target_id": target_id, "target_uniprot_id": ""})
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(save_path, index=False)
    return df


def build_drug_similarity_morgan(drug_features_path: Path, save_path: Path, top_k: int) -> None:
    features = load_json_dict(drug_features_path)
    ids = list(features.keys())
    arr = np.array([features[i] for i in ids], dtype=float)
    n = len(ids)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = arr[i] > 0
        for j in range(i, n):
            b = arr[j] > 0
            union = np.logical_or(a, b).sum()
            score = float(np.logical_and(a, b).sum() / union) if union else 0.0
            sim[i, j] = sim[j, i] = score
    sim = normalize_matrix(sim)
    sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
    save_similarity(ids, sparse, save_path, top_k, {"method": "morgan_tanimoto"})


def build_drug_similarity_gip(drug_features_path: Path, drug_disease_edges_path: Path, save_path: Path, top_k: int) -> None:
    features = load_json_dict(drug_features_path)
    drug_ids = list(features.keys())
    edges = pd.read_csv(drug_disease_edges_path)
    disease_ids = sorted(edges["DiseaseID"].astype(str).unique().tolist()) if len(edges) else []
    dmap = {d: i for i, d in enumerate(drug_ids)}
    dis_map = {d: i for i, d in enumerate(disease_ids)}
    Y = np.zeros((len(drug_ids), len(disease_ids)), dtype=float)
    for _, row in edges.iterrows():
        drug_id = str(row["drugbank_id"])
        disease_id = str(row["DiseaseID"])
        if drug_id in dmap and disease_id in dis_map:
            Y[dmap[drug_id], dis_map[disease_id]] = 1.0
    row_norm = np.sum(Y**2, axis=1)
    gamma = 1.0 / np.mean(row_norm[row_norm > 0]) if np.any(row_norm > 0) else 1.0
    dist = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    sim = np.exp(-gamma * dist)
    sim = normalize_matrix(sim)
    sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
    save_similarity(ids=drug_ids, matrix=sparse, save_path=save_path, top_k=top_k, metadata={"method": "gip_kernel", "gamma": gamma})


def build_drug_similarity_drsie(drug_features_path: Path, drug_disease_edges_path: Path, save_path: Path, top_k: int) -> None:
    features = load_json_dict(drug_features_path)
    drug_ids = list(features.keys())
    edges = pd.read_csv(drug_disease_edges_path)
    disease_sets = edges.groupby("drugbank_id")["DiseaseID"].apply(lambda s: set(map(str, s))).to_dict() if len(edges) else {}
    disease_counts = edges["DiseaseID"].astype(str).value_counts().to_dict() if len(edges) else {}
    num_drugs = max(1, len(drug_ids))
    sim = np.zeros((len(drug_ids), len(drug_ids)), dtype=float)
    for i, di in enumerate(drug_ids):
        set_i = disease_sets.get(di, set())
        for j in range(i, len(drug_ids)):
            set_j = disease_sets.get(drug_ids[j], set())
            union = set_i | set_j
            if not union:
                score = 0.0
            else:
                common = set_i & set_j

                def weight(did: str) -> float:
                    return math.log(1.0 + num_drugs / max(1, disease_counts.get(did, 1)))

                common_w = sum(weight(did) for did in common)
                union_w = sum(weight(did) for did in union)
                score = common_w / union_w if union_w else 0.0
            sim[i, j] = sim[j, i] = score
    sim = normalize_matrix(sim)
    sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
    save_similarity(drug_ids, sparse, save_path, top_k, {"method": "weighted_disease_set_overlap_proxy"})


def build_disease_similarity_o(disease_features_path: Path, save_path: Path, top_k: int) -> None:
    features = load_json_dict(disease_features_path)
    ids = list(features.keys())
    vectors = np.array([features[i] for i in ids], dtype=float)
    similarity_from_vectors(ids, vectors, save_path, top_k, {"method": "text_cosine"})


def build_disease_similarity_h(
    phenotype_hpoa_path: Path,
    ctd_diseases_path: Path,
    target_ids_path: Path,
    save_path: Path,
    top_k: int,
) -> None:
    target_ids = load_target_ids(target_ids_path)
    vocab = load_ctd_csv(ctd_diseases_path)
    vocab = vocab[vocab["DiseaseID"].isin(target_ids)].copy()
    mesh_to_omim = {str(row["DiseaseID"]): extract_omim_ids(row.get("AltDiseaseIDs")) for _, row in vocab.iterrows()}

    df_hpoa = pd.read_csv(phenotype_hpoa_path, sep="\t", comment="#", low_memory=False)
    if "database_id" not in df_hpoa.columns or "hpo_id" not in df_hpoa.columns:
        raise ValueError("phenotype.hpoa must contain database_id and hpo_id columns")

    term_counts = df_hpoa["hpo_id"].value_counts()
    total = max(1, int(term_counts.sum()))
    ic = {term: -math.log(count / total) for term, count in term_counts.items()}
    disease_to_hpo = df_hpoa.groupby("database_id")["hpo_id"].apply(lambda s: set(map(str, s))).to_dict()

    sim = np.zeros((len(target_ids), len(target_ids)), dtype=float)
    for i, did_i in enumerate(target_ids):
        hpo_i: set[str] = set()
        for omim_id in mesh_to_omim.get(did_i, []):
            hpo_i |= disease_to_hpo.get(omim_id, set())
        for j in range(i, len(target_ids)):
            did_j = target_ids[j]
            hpo_j: set[str] = set()
            for omim_id in mesh_to_omim.get(did_j, []):
                hpo_j |= disease_to_hpo.get(omim_id, set())
            union = hpo_i | hpo_j
            if not union:
                score = 0.0
            else:
                common = hpo_i & hpo_j
                common_w = sum(ic.get(h, 0.0) for h in common)
                union_w = sum(ic.get(h, 0.0) for h in union)
                score = common_w / union_w if union_w else 0.0
            sim[i, j] = sim[j, i] = score
    sim = normalize_matrix(sim)
    sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
    save_similarity(target_ids, sparse, save_path, top_k, {"method": "hpo_ic_weighted_jaccard"})


def build_disease_gene_edges_if_available(source_path: Optional[Path], processed_save_path: Path) -> Optional[pd.DataFrame]:
    if source_path is None or not source_path.exists():
        return None
    gene_universe = None
    gene_info_path = None
    maybe_gene_json = processed_save_path.parent / "gene_features_dict.json"
    if maybe_gene_json.exists():
        gene_universe = set(load_json_dict(maybe_gene_json).keys())
    candidate_gene_info = source_path.parent / "gene_info"
    if candidate_gene_info.exists():
        gene_info_path = candidate_gene_info
    else:
        fallback_gene_info = processed_save_path.parent.parent / "other" / "gene_info"
        if fallback_gene_info.exists():
            gene_info_path = fallback_gene_info
    df = standardize_disease_gene_edges(source_path, gene_universe=gene_universe, gene_info_path=gene_info_path)
    df.to_csv(processed_save_path, index=False)
    return df


def load_gene_network_edges(
    path: Path,
    allowed_genes: Optional[set[str]] = None,
    string_alias_path: Optional[Path] = None,
    gene_info_path: Optional[Path] = None,
    min_confidence: float = 700.0,
) -> pd.DataFrame:
    lower = path.name.lower()
    if "protein.links" in lower or "protein.physical.links" in lower:
        if string_alias_path is None or not string_alias_path.exists():
            raise FileNotFoundError("STRING links require a matching aliases file.")
        string_to_gene: Dict[str, set[str]] = {}
        alias_iter = pd.read_csv(
            string_alias_path,
            sep="\t",
            compression="gzip" if ".gz" in {s.lower() for s in string_alias_path.suffixes} else None,
            header=0,
            comment="#",
            names=["string_id", "alias", "source"],
            chunksize=500_000,
            low_memory=False,
        )
        for chunk in alias_iter:
            chunk = chunk[chunk["string_id"].astype(str).str.startswith("9606.")].copy()
            if len(chunk) == 0:
                continue
            chunk["alias"] = chunk["alias"].astype(str).str.strip()
            if allowed_genes:
                chunk = chunk[chunk["alias"].isin(allowed_genes)].copy()
            if len(chunk) == 0:
                continue
            chunk = chunk.drop_duplicates(subset=["string_id", "alias"])
            for string_id, alias in chunk[["string_id", "alias"]].itertuples(index=False, name=None):
                string_to_gene.setdefault(str(string_id), set()).add(str(alias))

        if not string_to_gene:
            return pd.DataFrame(columns=["gene1", "gene2", "weight", "source"])

        valid_string_ids = set(string_to_gene.keys())
        edge_weights: Dict[Tuple[str, str], float] = {}
        link_iter = pd.read_csv(
            path,
            sep=r"\s+",
            compression="gzip" if ".gz" in {s.lower() for s in path.suffixes} else None,
            chunksize=300_000,
            low_memory=False,
        )
        protein1 = protein2 = score_col = None
        for chunk in link_iter:
            if protein1 is None or protein2 is None:
                protein1 = find_column(chunk, ["protein1"])
                protein2 = find_column(chunk, ["protein2"])
                score_col = find_column(chunk, ["combined_score", "score", "confidence"])
                if protein1 is None or protein2 is None:
                    raise ValueError(f"Cannot detect STRING protein columns in {path}")
            chunk = chunk[
                chunk[protein1].astype(str).isin(valid_string_ids)
                & chunk[protein2].astype(str).isin(valid_string_ids)
            ].copy()
            if len(chunk) == 0:
                continue
            if score_col is not None:
                chunk[score_col] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0.0)
                chunk = chunk[chunk[score_col] >= min_confidence].copy()
            if len(chunk) == 0:
                continue
            cols = [protein1, protein2] + ([score_col] if score_col else [])
            for row in chunk[cols].itertuples(index=False, name=None):
                p1 = str(row[0]).strip()
                p2 = str(row[1]).strip()
                genes1 = string_to_gene.get(p1)
                genes2 = string_to_gene.get(p2)
                if not genes1 or not genes2:
                    continue
                raw_score = float(row[2]) if score_col else 1000.0
                weight = raw_score / 1000.0 if raw_score > 1.0 else raw_score
                for g1 in genes1:
                    for g2 in genes2:
                        if g1 == g2:
                            continue
                        a, b = (g1, g2) if g1 <= g2 else (g2, g1)
                        prev = edge_weights.get((a, b), 0.0)
                        if weight > prev:
                            edge_weights[(a, b)] = weight
        if not edge_weights:
            return pd.DataFrame(columns=["gene1", "gene2", "weight", "source"])
        df = pd.DataFrame(
            [{"gene1": a, "gene2": b, "weight": w, "source": "string"} for (a, b), w in edge_weights.items()]
        )
    else:
        df = read_table_auto(path)
        g1 = find_column(df, ["gene1", "gene_a", "source_gene", "protein1"])
        g2 = find_column(df, ["gene2", "gene_b", "target_gene", "protein2"])
        weight_col = find_column(df, ["weight", "score", "confidence", "combined_score"])
        if g1 is None or g2 is None:
            raise ValueError(f"Cannot detect gene network columns in {path}")
        df = df[[g1, g2] + ([weight_col] if weight_col else [])].copy()
        cols = ["gene1", "gene2"] + (["weight"] if weight_col else [])
        df.columns = cols
        if "weight" not in df.columns:
            df["weight"] = 1.0
        df["source"] = "humannet"
        if gene_info_path and df["gene1"].astype(str).str.isnumeric().any():
            id_to_symbol, _ = load_gene_info_map(gene_info_path)
            df["gene1"] = df["gene1"].astype(str).map(lambda x: id_to_symbol.get(x, x))
            df["gene2"] = df["gene2"].astype(str).map(lambda x: id_to_symbol.get(x, x))

    df["gene1"] = df["gene1"].astype(str).str.strip()
    df["gene2"] = df["gene2"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[(df["gene1"] != "") & (df["gene2"] != "") & (df["gene1"] != df["gene2"]) & (df["weight"] > 0)].copy()
    if allowed_genes:
        df = df[df["gene1"].isin(allowed_genes) & df["gene2"].isin(allowed_genes)].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=["gene1", "gene2", "weight", "source"])
    gene_min = np.where(df["gene1"].values <= df["gene2"].values, df["gene1"].values, df["gene2"].values)
    gene_max = np.where(df["gene1"].values <= df["gene2"].values, df["gene2"].values, df["gene1"].values)
    df["gene1"] = gene_min
    df["gene2"] = gene_max
    df = df.groupby(["gene1", "gene2"], as_index=False).agg({"weight": "max", "source": "first"})
    return df


def build_gene_network_edges(
    save_path: Path,
    gene_features_path: Path,
    string_links_path: Optional[Path] = None,
    string_alias_path: Optional[Path] = None,
    humannet_path: Optional[Path] = None,
    gene_info_path: Optional[Path] = None,
    min_confidence: float = 700.0,
) -> Optional[pd.DataFrame]:
    allowed_genes = set(load_json_dict(gene_features_path).keys())
    source_path = string_links_path if string_links_path and string_links_path.exists() else humannet_path
    if source_path is None or not source_path.exists():
        return None
    df = load_gene_network_edges(
        source_path,
        allowed_genes=allowed_genes,
        string_alias_path=string_alias_path,
        gene_info_path=gene_info_path,
        min_confidence=min_confidence,
    )
    if df is None or len(df) == 0:
        return None
    df.to_csv(save_path, index=False)
    return df


def build_disease_similarity_g(
    target_ids_path: Path,
    drug_disease_edges_path: Path,
    save_path: Path,
    top_k: int,
    disease_gene_edges_path: Optional[Path] = None,
    gene_network_edges_path: Optional[Path] = None,
) -> None:
    ids = load_target_ids(target_ids_path)

    if disease_gene_edges_path and disease_gene_edges_path.exists():
        dg = standardize_disease_gene_edges(disease_gene_edges_path)
        dg = dg[dg["disease_id"].isin(ids)].copy()
        disease_to_gene = dg.groupby("disease_id")["gene_id"].apply(lambda s: set(map(str, s))).to_dict()

        if gene_network_edges_path and gene_network_edges_path.exists():
            hn = read_table_auto(gene_network_edges_path)
            g1 = find_column(hn, ["gene1"])
            g2 = find_column(hn, ["gene2"])
            weight_col = find_column(hn, ["weight", "score", "confidence"])
            if g1 and g2:
                edge_weights: Dict[Tuple[str, str], float] = {}
                for _, row in hn.iterrows():
                    a = str(row[g1]).strip()
                    b = str(row[g2]).strip()
                    if not a or not b:
                        continue
                    weight = float(row[weight_col]) if weight_col and pd.notna(row[weight_col]) else 1.0
                    edge_weights[(a, b)] = weight
                    edge_weights[(b, a)] = weight

                sim = np.zeros((len(ids), len(ids)), dtype=float)
                for i, di in enumerate(ids):
                    genes_i = disease_to_gene.get(di, set())
                    for j in range(i, len(ids)):
                        genes_j = disease_to_gene.get(ids[j], set())
                        if not genes_i or not genes_j:
                            score = 0.0
                        else:
                            overlap = len(genes_i & genes_j)
                            total = 0.0
                            for gi in genes_i:
                                best = 0.0
                                for gj in genes_j:
                                    best = max(best, edge_weights.get((gi, gj), 0.0))
                                total += best
                            reverse_total = 0.0
                            for gj in genes_j:
                                best = 0.0
                                for gi in genes_i:
                                    best = max(best, edge_weights.get((gj, gi), 0.0))
                                reverse_total += best
                            interaction = (total / max(len(genes_i), 1) + reverse_total / max(len(genes_j), 1)) / 2.0
                            score = (overlap + interaction) / 2.0
                        sim[i, j] = sim[j, i] = score
                sim = normalize_matrix(sim)
                sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
                method_name = "string_gene_function_similarity" if "string" in str(gene_network_edges_path).lower() else "humannet_gene_function_similarity"
                save_similarity(ids, sparse, save_path, top_k, {"method": method_name})
                return

        sim = np.zeros((len(ids), len(ids)), dtype=float)
        for i, di in enumerate(ids):
            set_i = disease_to_gene.get(di, set())
            for j in range(i, len(ids)):
                set_j = disease_to_gene.get(ids[j], set())
                union = set_i | set_j
                score = len(set_i & set_j) / len(union) if union else 0.0
                sim[i, j] = sim[j, i] = score
        sim = normalize_matrix(sim)
        sparse = sparsify_top_k(sim, k=top_k, symmetric=True)
        save_similarity(ids, sparse, save_path, top_k, {"method": "shared_gene_jaccard_fallback"})
        return

    # Do not fall back to full drug-disease edges here: that would encode
    # label information from the entire dataset into a disease similarity view.
    save_similarity(
        ids,
        np.zeros((len(ids), len(ids)), dtype=float),
        save_path,
        top_k,
        {
            "method": "disabled_missing_disease_gene_edges",
            "note": "Drug-disease fallback removed to prevent label leakage.",
        },
    )


def integrate_reference_processed_results(paths: ProjectPaths, root: Path, top_k: int = 5) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "reference_found": False,
        "assets": {},
        "integrated": {},
        "warnings": [],
    }
    reference = discover_reference_processed(root)
    if not (reference.get("processed_dir") or reference.get("processed_zip")):
        report["warnings"].append("reference processed results not found")
        write_json(paths.reports / "reference_processed_report.json", report)
        return report

    report["reference_found"] = True
    report["assets"] = reference

    target_ids_path = paths.interim / "target_disease_ids.txt"
    if not target_ids_path.exists():
        report["warnings"].append("target_disease_ids.txt not found; skip reference integration")
        write_json(paths.reports / "reference_processed_report.json", report)
        return report

    target_ids = load_target_ids(target_ids_path)

    if _reference_has_asset(reference, "selected_diseases.csv"):
        selected = _read_reference_csv(reference, "selected_diseases.csv")
        if "DiseaseID" in selected.columns:
            reference_ids = selected["DiseaseID"].astype(str).tolist()

            if _reference_has_asset(reference, "sim_hpo.npy"):
                ref_h = _read_reference_npy(reference, "sim_hpo.npy")
                aligned_h, matched_h = align_reference_similarity(target_ids, reference_ids, ref_h)
                meta_h = merge_similarity_payload(
                    target_ids,
                    paths.processed / "DiSimNet_H.json",
                    top_k,
                    aligned_h,
                    {
                        "method": "merged_current_and_reference_hpo",
                        "reference_source": reference.get("processed_zip") or reference.get("processed_dir"),
                        "reference_match_count": matched_h,
                    },
                )
                report["integrated"]["DiSimNet_H"] = meta_h

            if _reference_has_asset(reference, "sim_gene.npy"):
                ref_g = _read_reference_npy(reference, "sim_gene.npy")
                aligned_g, matched_g = align_reference_similarity(target_ids, reference_ids, ref_g)
                meta_g = merge_similarity_payload(
                    target_ids,
                    paths.processed / "DiSimNet_G.json",
                    top_k,
                    aligned_g,
                    {
                        "method": "merged_current_and_reference_gene",
                        "reference_source": reference.get("processed_zip") or reference.get("processed_dir"),
                        "reference_match_count": matched_g,
                    },
                )
                report["integrated"]["DiSimNet_G"] = meta_g
        else:
            report["warnings"].append("selected_diseases.csv found but DiseaseID column is missing")

    if _reference_has_asset(reference, "string_edges.csv"):
        df_string = _read_reference_csv(reference, "string_edges.csv")
        if not df_string.empty:
            g1 = find_column(df_string, ["gene1"])
            g2 = find_column(df_string, ["gene2"])
            w = find_column(df_string, ["weight", "score", "confidence"])
            if g1 and g2:
                out = df_string[[g1, g2] + ([w] if w else [])].copy()
                out.columns = ["gene1", "gene2"] + (["weight"] if w else [])
                if "weight" not in out.columns:
                    out["weight"] = 1.0
                out["source"] = "msrhgnn_graph_builder"
                out["gene1"] = out["gene1"].astype(str).str.strip()
                out["gene2"] = out["gene2"].astype(str).str.strip()
                out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
                out = out[(out["gene1"] != "") & (out["gene2"] != "") & (out["gene1"] != out["gene2"]) & (out["weight"] > 0)]
                if not out.empty:
                    out.to_csv(paths.processed / "gene_network_edges.csv", index=False)
                    report["integrated"]["gene_network_edges"] = {
                        "rows": int(len(out)),
                        "source": "string_edges.csv",
                    }
                else:
                    report["warnings"].append("reference string_edges.csv is present but empty after cleaning")
        else:
            report["warnings"].append("reference string_edges.csv is empty")

    write_json(paths.reports / "reference_processed_report.json", report)
    return report


def _matrix_from_edge_store(num_nodes: int, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> np.ndarray:
    matrix = np.zeros((num_nodes, num_nodes), dtype=float)
    if edge_index is None or edge_index.numel() == 0:
        return matrix
    weights = edge_attr.view(-1).detach().cpu().numpy() if edge_attr is not None and edge_attr.numel() else np.ones(edge_index.size(1), dtype=float)
    coords = edge_index.detach().cpu().numpy().T
    for (i, j), w in zip(coords, weights):
        matrix[int(i), int(j)] = max(matrix[int(i), int(j)], float(w))
    return matrix


def integrate_reference_pt_results(paths: ProjectPaths, root: Path, top_k: int = 5) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "reference_pt_found": False,
        "integrated": {},
        "warnings": [],
    }
    reference = discover_reference_processed(root)
    if not reference.get("final_hetero_pt"):
        report["warnings"].append("reference final_hetero_data_raw.pt not found")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report

    report["reference_pt_found"] = True
    pt_path = Path(reference["final_hetero_pt"])
    try:
        from torch_geometric.data import HeteroData  # type: ignore  # noqa: F401
    except Exception as exc:
        report["warnings"].append(f"torch_geometric unavailable, skip pt integration: {exc}")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report

    try:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        report["warnings"].append(f"failed to load reference pt: {exc}")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report

    if not hasattr(obj, "edge_types") or not hasattr(obj, "node_types"):
        report["warnings"].append("reference pt is not a supported HeteroData object")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report

    target_ids_path = paths.interim / "target_disease_ids.txt"
    if not target_ids_path.exists():
        report["warnings"].append("target_disease_ids.txt not found; skip pt integration")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report
    target_ids = load_target_ids(target_ids_path)

    reference_ids: List[str] = []
    if _reference_has_asset(reference, "selected_diseases.csv"):
        selected = _read_reference_csv(reference, "selected_diseases.csv")
        if "DiseaseID" in selected.columns:
            reference_ids = selected["DiseaseID"].astype(str).tolist()
    if not reference_ids:
        report["warnings"].append("selected_diseases.csv not available; cannot align disease nodes from pt")
        write_json(paths.reports / "reference_pt_report.json", report)
        return report

    disease_num_nodes = int(obj["disease"].num_nodes) if hasattr(obj["disease"], "num_nodes") and obj["disease"].num_nodes is not None else len(reference_ids)
    disease_edge_map = {
        ("disease", "sim_h", "disease"): ("DiSimNet_H.json", "merged_current_and_reference_pt_hpo"),
        ("disease", "sim_g", "disease"): ("DiSimNet_G.json", "merged_current_and_reference_pt_gene"),
        ("disease", "sim_t", "disease"): ("DiSimNet_O.json", "merged_current_and_reference_pt_text"),
    }

    for edge_type, (file_name, method_name) in disease_edge_map.items():
        if edge_type not in obj.edge_types:
            continue
        store = obj[edge_type]
        edge_index = getattr(store, "edge_index", None)
        edge_attr = getattr(store, "edge_attr", None)
        ref_matrix = _matrix_from_edge_store(disease_num_nodes, edge_index, edge_attr)
        aligned, matched = align_reference_similarity(target_ids, reference_ids, ref_matrix)
        meta = merge_similarity_payload(
            target_ids,
            paths.processed / file_name,
            top_k,
            aligned,
            {
                "method": method_name,
                "reference_pt_source": str(pt_path),
                "reference_match_count": matched,
            },
        )
        report["integrated"][file_name] = meta

    write_json(paths.reports / "reference_pt_report.json", report)
    return report


def adjacency_to_edges(matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    if matrix.size == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
    coords = np.argwhere(matrix > 0)
    if coords.size == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
    edge_index = torch.tensor(coords.T, dtype=torch.long)
    edge_weight = torch.tensor(matrix[coords[:, 0], coords[:, 1]], dtype=torch.float)
    return edge_index, edge_weight


def summarize_graph_coverage(graph: Dict[str, Any]) -> Dict[str, Any]:
    edge_index = graph.get("edge_index", {})
    edge_counts = {
        relation: int(edges.size(1)) if isinstance(edges, torch.Tensor) and edges.dim() == 2 else 0
        for relation, edges in edge_index.items()
    }

    def has_relation(name: str) -> bool:
        return edge_counts.get(name, 0) > 0

    has_dti = has_relation("drug__targets__gene")
    has_dg = has_relation("disease__associated_with__gene")
    has_gg = has_relation("gene__interacts__gene")
    has_dd = has_relation("drug__treats__disease")

    active_high_order_paths: List[str] = []
    if has_dti and has_dg:
        active_high_order_paths.append("Drug-Protein-Disease")
    if has_dg:
        active_high_order_paths.append("Disease-Protein-Disease")
    if has_dg and has_gg:
        active_high_order_paths.append("Disease-Protein-Protein-Disease")

    missing_prerequisites: List[str] = []
    if not has_dti:
        missing_prerequisites.append("drug__targets__gene")
    if not has_dg:
        missing_prerequisites.append("disease__associated_with__gene")
    if not has_gg:
        missing_prerequisites.append("gene__interacts__gene")

    return {
        "edge_counts": edge_counts,
        "low_order_views": {
            "required_relations": [
                "drug__targets__gene",
                "disease__associated_with__gene",
                "drug__treats__disease",
            ],
            "active_relations": [
                relation
                for relation in [
                    "drug__targets__gene",
                    "disease__associated_with__gene",
                    "drug__treats__disease",
                ]
                if has_relation(relation)
            ],
            "complete": has_dti and has_dg and has_dd,
        },
        "high_order_views": {
            "implemented_paths": [
                "Drug-Protein-Disease",
                "Disease-Protein-Disease",
                "Disease-Protein-Protein-Disease",
            ],
            "active_paths": active_high_order_paths,
            "complete": has_dti and has_dg and has_gg,
            "missing_prerequisites": missing_prerequisites,
            "note": (
                "Disease-Protein-Protein-Disease requires gene__interacts__gene. "
                "Without gene network edges, the model stays in partial high-order mode."
            ),
        },
    }


def build_final_graph(paths: ProjectPaths) -> Dict[str, Any]:
    drug_feat = load_json_dict(paths.processed / "drug_features_dict.json")
    disease_feat = load_json_dict(paths.processed / "disease_features_dict.json")
    gene_feat = load_json_dict(paths.processed / "gene_features_dict.json")

    drug_ids = list(drug_feat.keys())
    disease_ids = list(disease_feat.keys())
    gene_ids = list(gene_feat.keys())
    d_map = {x: i for i, x in enumerate(drug_ids)}
    dis_map = {x: i for i, x in enumerate(disease_ids)}
    g_map = {x: i for i, x in enumerate(gene_ids)}

    graph: Dict[str, Any] = {
        "node_ids": {"drug": drug_ids, "disease": disease_ids, "gene": gene_ids},
        "node_features": {
            "drug": torch.tensor([drug_feat[x] for x in drug_ids], dtype=torch.float),
            "disease": torch.tensor([disease_feat[x] for x in disease_ids], dtype=torch.float),
            "gene": torch.tensor([gene_feat[x] for x in gene_ids], dtype=torch.float),
        },
        "edge_index": {},
        "edge_weight": {},
        "metadata": {},
    }

    for name in ["DrugSim_Morgan", "DrugSim_GIP", "DrugSim_DRSIE"]:
        path = paths.processed / f"{name}.json"
        if path.exists():
            payload = read_json(path)
            relation = f"drug__{name.lower()}__drug"
            edge_index, edge_weight = adjacency_to_edges(np.array(payload["matrix"], dtype=float))
            graph["edge_index"][relation] = edge_index
            graph["edge_weight"][relation] = edge_weight
            graph["metadata"][relation] = payload.get("metadata", {})

    for name in ["DiSimNet_O", "DiSimNet_H", "DiSimNet_G"]:
        path = paths.processed / f"{name}.json"
        if path.exists():
            payload = read_json(path)
            relation = f"disease__{name.lower()}__disease"
            edge_index, edge_weight = adjacency_to_edges(np.array(payload["matrix"], dtype=float))
            graph["edge_index"][relation] = edge_index
            graph["edge_weight"][relation] = edge_weight
            graph["metadata"][relation] = payload.get("metadata", {})

    dd = pd.read_csv(paths.processed / "drug_disease_edges.csv")
    dd_pairs = [
        [d_map[str(r["drugbank_id"])], dis_map[str(r["DiseaseID"])]]
        for _, r in dd.iterrows()
        if str(r["drugbank_id"]) in d_map and str(r["DiseaseID"]) in dis_map
    ]
    graph["edge_index"]["drug__treats__disease"] = (
        torch.tensor(dd_pairs, dtype=torch.long).t().contiguous() if dd_pairs else torch.empty((2, 0), dtype=torch.long)
    )
    graph["edge_weight"]["drug__treats__disease"] = torch.ones((len(dd_pairs),), dtype=torch.float)

    dti = pd.read_csv(paths.processed / "dti_edges.csv")
    dti_pairs = [
        [d_map[str(r["drug_id"])], g_map[str(r["target_id"])]]
        for _, r in dti.iterrows()
        if str(r["drug_id"]) in d_map and str(r["target_id"]) in g_map
    ]
    graph["edge_index"]["drug__targets__gene"] = (
        torch.tensor(dti_pairs, dtype=torch.long).t().contiguous() if dti_pairs else torch.empty((2, 0), dtype=torch.long)
    )
    graph["edge_weight"]["drug__targets__gene"] = torch.ones((len(dti_pairs),), dtype=torch.float)

    dg_path = paths.processed / "disease_gene_edges.csv"
    if dg_path.exists():
        dg = pd.read_csv(dg_path)
        dg_pairs = [
            [dis_map[str(r["disease_id"])], g_map[str(r["gene_id"])]]
            for _, r in dg.iterrows()
            if str(r["disease_id"]) in dis_map and str(r["gene_id"]) in g_map
        ]
        graph["edge_index"]["disease__associated_with__gene"] = (
            torch.tensor(dg_pairs, dtype=torch.long).t().contiguous() if dg_pairs else torch.empty((2, 0), dtype=torch.long)
        )
        graph["edge_weight"]["disease__associated_with__gene"] = torch.ones((len(dg_pairs),), dtype=torch.float)

    gene_network_path = paths.processed / "gene_network_edges.csv"
    if gene_network_path.exists():
        gn = pd.read_csv(gene_network_path)
        gn_pairs = [
            [g_map[str(r["gene1"])], g_map[str(r["gene2"])]]
            for _, r in gn.iterrows()
            if str(r["gene1"]) in g_map and str(r["gene2"]) in g_map
        ]
        gn_weights = [
            float(r["weight"])
            for _, r in gn.iterrows()
            if str(r["gene1"]) in g_map and str(r["gene2"]) in g_map
        ]
        if gn_pairs:
            forward = torch.tensor(gn_pairs, dtype=torch.long).t().contiguous()
            reverse = torch.stack([forward[1], forward[0]], dim=0)
            graph["edge_index"]["gene__interacts__gene"] = torch.cat([forward, reverse], dim=1)
            weight_tensor = torch.tensor(gn_weights, dtype=torch.float)
            graph["edge_weight"]["gene__interacts__gene"] = torch.cat([weight_tensor, weight_tensor], dim=0)
            graph["metadata"]["gene__interacts__gene"] = {"source": "string_or_humannet"}

    try:
        from torch_geometric.data import HeteroData  # type: ignore

        data = HeteroData()
        data["drug"].x = graph["node_features"]["drug"]
        data["disease"].x = graph["node_features"]["disease"]
        data["gene"].x = graph["node_features"]["gene"]
        for relation, edge_index in graph["edge_index"].items():
            src, rel, dst = relation.split("__")
            data[src, rel, dst].edge_index = edge_index
            if relation in graph["edge_weight"]:
                data[src, rel, dst].edge_weight = graph["edge_weight"][relation]
        torch.save(data, paths.final / "final_hetero_data.pt")
        graph["metadata"]["pyg_export"] = "final_hetero_data.pt"
    except Exception as exc:
        graph["metadata"]["pyg_export"] = f"skipped: {exc}"

    graph["metadata"]["coverage"] = summarize_graph_coverage(graph)
    torch.save(graph, paths.final / "final_graph_data.pt")
    return graph


def validate_outputs(paths: ProjectPaths) -> Dict[str, Any]:
    report: Dict[str, Any] = {"files": {}, "warnings": []}
    expected = [
        "drug_features_dict.json",
        "disease_features_dict.json",
        "gene_features_dict.json",
        "drug_disease_edges.csv",
        "dti_edges.csv",
        "disease_gene_edges.csv",
        "DrugSim_Morgan.json",
        "DrugSim_GIP.json",
        "DrugSim_DRSIE.json",
        "DiSimNet_O.json",
        "DiSimNet_H.json",
        "DiSimNet_G.json",
        "final_graph_data.pt",
    ]
    for name in expected:
        base = paths.processed if name.endswith((".json", ".csv")) else paths.final
        path = base / name
        report["files"][name] = path.exists()
        if not path.exists():
            report["warnings"].append(f"missing: {name}")

    if not (paths.processed / "disease_gene_edges.csv").exists():
        report["warnings"].append(
            "disease_gene_edges.csv not found; DiSimNet_G is disabled to avoid label leakage and heterogeneous graph lacks disease-gene edges."
        )
    if not (paths.processed / "gene_network_edges.csv").exists():
        report["warnings"].append(
            "gene_network_edges.csv not found; STRING/HumanNet functional view and gene interaction meta-paths will fall back."
        )

    meta_path = paths.processed / "disease_texts_meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        if not meta.get("used_omim_source"):
            report["warnings"].append("OMIM text source not found; disease text view falls back to CTD definitions / disease names.")

    g_path = paths.processed / "DiSimNet_G.json"
    if g_path.exists():
        g_method = read_json(g_path).get("metadata", {}).get("method")
        report["DiSimNet_G_method"] = g_method
        if g_method == "shared_drug_jaccard_fallback":
            report["warnings"].append(
                "Unsafe DiSimNet_G detected: shared_drug_jaccard_fallback leaks labels via full drug-disease edges."
            )

    final_graph_path = paths.final / "final_graph_data.pt"
    if final_graph_path.exists():
        try:
            graph = torch.load(final_graph_path, map_location="cpu")
            graph_g_method = graph.get("metadata", {}).get("disease__disimnet_g__disease", {}).get("method")
            if graph_g_method == "shared_drug_jaccard_fallback":
                report["warnings"].append(
                    "Unsafe final_graph_data.pt detected: disease__disimnet_g__disease was built from shared_drug_jaccard_fallback."
                )
            coverage = graph.get("metadata", {}).get("coverage", {})
            if coverage:
                report["graph_coverage"] = coverage
                high_order = coverage.get("high_order_views", {})
                if not high_order.get("complete", False):
                    report["warnings"].append(
                        "High-order view is only partially active; complete Disease-Protein-Protein-Disease needs gene_network_edges.csv."
                    )
        except Exception as exc:
            report["warnings"].append(f"failed to inspect final_graph_data.pt: {exc}")

    write_json(paths.reports / "validation_report.json", report)
    return report
