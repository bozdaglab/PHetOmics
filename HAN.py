"""
PyTorch Geometric HANConv baseline for plant phenotype prediction.
================================================================

Uses the same heterogeneous graph as our_model.py:
  - plant, microbe, and metabolite node features
  - all edge types passed directly to HANConv via edge_index_dict

Run: python HAN.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HANConv

    PYG_AVAILABLE = True
except Exception as e:  # pragma: no cover
    PYG_AVAILABLE = False
    PYG_IMPORT_ERROR = e


# ============================================================
# Config kept aligned with your baseline GNN settings
# ============================================================
BASE_DIR = Path(".")
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
MODEL_NAME = "pyg_han"
OUTPUT_DIR = BASE_DIR / "han_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]

ID_COLS = ["sample_id", "variety", "varietyID", "block", "plot"]
ID_COL_SET = set(ID_COLS)

TARGET_TRAITS = ["ShootDryWeight_g"]

OMICS_TO_USE = ["genotype", "metabolome", "microbiome"]

# Same plant-plant threshold as your homogeneous GNN baseline.
PLANT_SIMILARITY_THRESHOLD = 0.70
USE_ABSOLUTE_CORRELATION = False
FORCE_RECOMPUTE_PLANT_EDGES = True

# Use the threshold file created by your metabolome graph script.
METABOLITE_SIMILARITY_THRESHOLD = 0.80

# Your preferred validation-tuning grid from the baseline GNN.
HYPERPARAM_GRID = [
    {"hidden_channels": 64,  "num_layers": 2, "dropout": 0.2, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 128, "num_layers": 2, "dropout": 0.2, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 128, "num_layers": 3, "dropout": 0.3, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 64,  "num_layers": 2, "dropout": 0.3, "lr": 0.0005, "weight_decay": 1e-4},
]

MAX_EPOCHS = 500
PATIENCE = 60

SELECTION_METRIC = "mae"
SELECTION_MODE = "min"

SCALE_TARGET = True
SCALE_PLANT_NODE_FEATURES_FOR_GNN = False
SCALE_NON_PLANT_NODE_FEATURES = False

# Fixed HAN settings. These are not tuned, so the baseline tuning protocol remains the same.
HAN_HEADS = 2
ATTENTION_EXPORT_TOP_N_PER_EDGE_TYPE = 500

# Plant-to-omic observation edges are necessary if you want the microbe/metabolite
# graph to influence plant phenotype predictions.
ADD_PLANT_OMIC_OBSERVATION_EDGES = True

# Options:
#   "nonzero"  : add every plant-feature edge where abs(value) > OBS_MIN_ABS_VALUE
#   "topk_abs" : add only the top absolute-valued features per plant
OBSERVATION_EDGE_MODE = "topk_positive"
OBS_MIN_ABS_VALUE = 1e-8
OBS_TOP_K_MICROBES = 50
OBS_TOP_K_METABOLITES = 50

# Biological microbe-metabolite edge file generated from KEGG/reaction logic.
BIOLOGICAL_EDGE_PATH = PREPROCESSED_DIR / "microbe_metabolite_biological_edges.csv"


# ============================================================
# Safety check
# ============================================================
def check_pyg_available():
    if not PYG_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric import failed. Install PyTorch and PyTorch Geometric first.\n"
            "Typical install depends on your CUDA/PyTorch version. Start with:\n"
            "  pip install torch\n"
            "  pip install torch-geometric\n\n"
            f"Original import error: {PYG_IMPORT_ERROR}"
        )


# ============================================================
# Basic helpers
# ============================================================
def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def as_float32_frame(df: pd.DataFrame, id_cols: Sequence[str]) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in set(id_cols)]
    out = df[feature_cols].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def strip_duplicate_suffix(value: object) -> str:
    """Remove pandas duplicate suffixes such as .1, .2, .18 from taxonomy names."""
    return re.sub(r"\.\d+$", "", str(value).strip())


def strip_known_prefix(value: str, prefix: str) -> str:
    return value[len(prefix):] if value.startswith(prefix) else value


def normalize_compound_key(value: object) -> str:
    """
    Robust metabolite-name normalization.

    Handles examples like:
      L.Glutamic.acid
      mtb_L.Glutamic.acid
      metabolome__mtb_L.Glutamic.acid
      L Glutamic acid
    """
    s = str(value).strip()
    s = strip_known_prefix(s, "metabolome__")
    s = strip_known_prefix(s, "mtb_")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def safe_edge_key(edge_type: Tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def edge_type_from_key(key: str) -> Tuple[str, str, str]:
    a, b, c = key.split("__", 2)
    return (a, b, c)


def make_undirected_id_edges(
    rows: Iterable[Tuple[str, str, float]],
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "edge_weight",
) -> pd.DataFrame:
    """Return directed rows for both directions, de-duplicated."""
    seen = set()
    out = []
    for s, t, w in rows:
        if pd.isna(s) or pd.isna(t) or s == t:
            continue
        for a, b in [(s, t), (t, s)]:
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out.append({source_col: a, target_col: b, weight_col: float(w)})
    return pd.DataFrame(out, columns=[source_col, target_col, weight_col])


# ============================================================
# Load plant plot node data, same as baseline GNN
# ============================================================
def assert_same_ids(y_df: pd.DataFrame, x_df: pd.DataFrame, split_name: str, omic_name: str):
    y_ids = y_df[ID_COLS].astype(str).reset_index(drop=True)
    x_ids = x_df[ID_COLS].astype(str).reset_index(drop=True)

    if not y_ids.equals(x_ids):
        raise ValueError(
            f"ID/order mismatch for split={split_name}, omic={omic_name}. "
            f"Check y_{split_name}.csv and {omic_name}_{split_name}.csv."
        )


def load_split_from_seed(seed_dir: Path, split_name: str):
    y_df = read_csv_checked(seed_dir / f"y_{split_name}.csv")

    missing_traits = [t for t in TARGET_TRAITS if t not in y_df.columns]
    if missing_traits:
        raise ValueError(f"Missing TARGET_TRAITS in {seed_dir}/y_{split_name}.csv: {missing_traits}")

    x_parts = []

    for omic in OMICS_TO_USE:
        x_df = read_csv_checked(seed_dir / f"{omic}_{split_name}.csv")
        assert_same_ids(y_df, x_df, split_name, omic)

        feature_cols = [c for c in x_df.columns if c not in ID_COLS]
        x_part = x_df[feature_cols].copy()
        x_part.columns = [
            f"{omic}__{c}" if not c.startswith(f"{omic}__") else c
            for c in x_part.columns
        ]
        x_parts.append(x_part)

    X = pd.concat(x_parts, axis=1)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    ids = y_df[ID_COLS].copy()
    ids.insert(1, "split", split_name)
    y = y_df[TARGET_TRAITS].copy()

    return ids, X, y


def load_seed_plant_data(seed: int):
    seed_dir = PREPROCESSED_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed folder: {seed_dir}")

    all_ids, all_X, all_y = [], [], []
    for split_name in ["train", "val", "test"]:
        ids, X, y = load_split_from_seed(seed_dir, split_name)
        all_ids.append(ids)
        all_X.append(X)
        all_y.append(y)

    plant_table = pd.concat(all_ids, ignore_index=True)
    X_plant = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    if y_all[TARGET_TRAITS].isna().sum().sum() > 0:
        raise ValueError(f"Target y contains NaN values for {seed_dir.name}.")

    return seed_dir, plant_table, X_plant, y_all


# ============================================================
# Plant-plant similarity edges, same logic as baseline GNN
# ============================================================
def compute_rowwise_pearson(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    Z = Xc / norms
    sim = Z @ Z.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def plant_edge_filename(threshold: float):
    thr = str(threshold).replace(".", "p")
    abs_tag = "_abs" if USE_ABSOLUTE_CORRELATION else ""
    return f"hetero_han_plant_edges_pearson_thr_{thr}{abs_tag}.csv"


def load_or_compute_plant_edges(seed_dir: Path, plant_table: pd.DataFrame, X_plant: pd.DataFrame) -> pd.DataFrame:
    edge_path = seed_dir / plant_edge_filename(PLANT_SIMILARITY_THRESHOLD)

    if edge_path.exists() and not FORCE_RECOMPUTE_PLANT_EDGES:
        return pd.read_csv(edge_path)

    sim = compute_rowwise_pearson(X_plant.to_numpy(dtype=np.float32))
    n = sim.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    scores = sim[upper_i, upper_j]

    if USE_ABSOLUTE_CORRELATION:
        keep = np.abs(scores) >= PLANT_SIMILARITY_THRESHOLD
    else:
        keep = scores >= PLANT_SIMILARITY_THRESHOLD

    edges = pd.DataFrame({
        "source": upper_i[keep].astype(int),
        "target": upper_j[keep].astype(int),
        "edge_weight": scores[keep].astype(float),
        "similarity": scores[keep].astype(float),
        "method": "pearson",
        "threshold": PLANT_SIMILARITY_THRESHOLD,
        "source_sample_id": plant_table.loc[upper_i[keep], "sample_id"].to_numpy(),
        "target_sample_id": plant_table.loc[upper_j[keep], "sample_id"].to_numpy(),
        "source_split": plant_table.loc[upper_i[keep], "split"].to_numpy(),
        "target_split": plant_table.loc[upper_j[keep], "split"].to_numpy(),
    })
    edges.to_csv(edge_path, index=False)
    return edges


# ============================================================
# Load microbe and metabolite node tables
# ============================================================
def load_microbe_nodes(seed_dir: Path):
    graph_dir = seed_dir / "graph_microbiome"
    numeric_path = graph_dir / "microbiome_node_features_numeric.csv"
    audit_path = graph_dir / "microbiome_node_features_audit.csv"

    numeric = read_csv_checked(numeric_path)
    audit = read_csv_checked(audit_path)

    if "microbe_id" not in numeric.columns:
        raise ValueError(f"Missing microbe_id in {numeric_path}")

    microbe_ids = numeric["microbe_id"].astype(str).tolist()
    X_microbe = as_float32_frame(numeric, id_cols=["microbe_id"])
    feature_names = X_microbe.columns.tolist()

    microbe_table = pd.DataFrame({
        "microbe_id": microbe_ids,
        "node_type": "microbe",
    })
    if "raw_taxa_column" in audit.columns:
        microbe_table = microbe_table.merge(
            audit[[c for c in ["microbe_id", "raw_taxa_column", "taxa_base", "genus", "species"] if c in audit.columns]],
            on="microbe_id",
            how="left",
        )

    return microbe_table, X_microbe, feature_names, audit


def load_metabolite_nodes(seed_dir: Path):
    graph_dir = seed_dir / "graph_metabolome"
    node_path = graph_dir / "metabolome_node_features.csv"
    nodes = read_csv_checked(node_path)

    if "node_id" not in nodes.columns:
        raise ValueError(f"Missing node_id in {node_path}")

    metabolite_ids = nodes["node_id"].astype(str).tolist()
    non_numeric = {"node_id", "node_type", "compound_clean", "smiles", "compound_col"}
    X_met = as_float32_frame(nodes, id_cols=list(non_numeric))
    feature_names = X_met.columns.tolist()

    met_table = nodes[[c for c in ["node_id", "node_type", "compound_clean", "smiles"] if c in nodes.columns]].copy()
    if "node_type" not in met_table.columns:
        met_table["node_type"] = "metabolite"

    return met_table, X_met, feature_names


# ============================================================
# Edge loading and mapping
# ============================================================
def tensor_edge_index_and_attr(
    edges: pd.DataFrame,
    src_col: str,
    dst_col: str,
    weight_col: str,
    src_index: Dict[str, int],
    dst_index: Dict[str, int],
    device,
):
    if edges.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
        return edge_index, edge_attr

    src = edges[src_col].astype(str).map(src_index)
    dst = edges[dst_col].astype(str).map(dst_index)
    keep = src.notna() & dst.notna()
    edges = edges.loc[keep].copy()
    src = src.loc[keep].astype(int).to_numpy()
    dst = dst.loc[keep].astype(int).to_numpy()

    w = pd.to_numeric(edges[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    edge_attr = torch.tensor(w.reshape(-1, 1), dtype=torch.float32, device=device)
    return edge_index, edge_attr


def add_edge_store(
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
):
    if edge_index.numel() == 0:
        return
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_attr = edge_attr


def load_microbe_microbe_edges(seed_dir: Path, microbe_index: Dict[str, int], device):
    path = seed_dir / "graph_microbiome" / "microbiome_taxonomy_edges.csv"
    df = read_csv_checked(path)

    if df.empty:
        return {}

    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"Microbiome edge file missing source/target columns: {path}")

    if "edge_weight" not in df.columns:
        df["edge_weight"] = 1.0

    # The generator usually saves both directions. This keeps only mapped rows.
    edge_index, edge_attr = tensor_edge_index_and_attr(
        df, "source", "target", "edge_weight", microbe_index, microbe_index, device
    )
    return {("microbe", "same_taxonomy", "microbe"): (edge_index, edge_attr, len(df))}


def load_metabolite_metabolite_edges(seed_dir: Path, metabolite_index: Dict[str, int], device):
    graph_dir = seed_dir / "graph_metabolome"
    preferred = graph_dir / f"metabolome_edges_similarity_ge_{METABOLITE_SIMILARITY_THRESHOLD:.2f}.csv"
    if preferred.exists():
        path = preferred
    else:
        candidates = sorted(graph_dir.glob("metabolome_edges_similarity_ge_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No metabolite similarity edge file found in {graph_dir}")
        path = candidates[0]

    df = read_csv_checked(path)
    if df.empty:
        return {}

    if "similarity" in df.columns:
        df["edge_weight"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(1.0)
    elif "avg_similarity" in df.columns:
        df["edge_weight"] = pd.to_numeric(df["avg_similarity"], errors="coerce").fillna(1.0)
    elif "edge_weight" not in df.columns:
        df["edge_weight"] = 1.0

    # Similarity file is usually upper triangle only, so add reverse direction.
    undirected = make_undirected_id_edges(
        [(r["source"], r["target"], r["edge_weight"]) for _, r in df.iterrows()],
        source_col="source",
        target_col="target",
        weight_col="edge_weight",
    )

    edge_index, edge_attr = tensor_edge_index_and_attr(
        undirected, "source", "target", "edge_weight", metabolite_index, metabolite_index, device
    )
    return {("metabolite", "similar_to", "metabolite"): (edge_index, edge_attr, len(undirected))}


def build_microbe_source_resolver(microbe_audit: pd.DataFrame):
    """Return a function that maps raw biological edge source strings to mb_* IDs."""
    audit = microbe_audit.copy()
    audit["microbe_id"] = audit["microbe_id"].astype(str)

    exact_map = {}
    if "raw_taxa_column" in audit.columns:
        for _, r in audit.iterrows():
            exact_map[str(r["raw_taxa_column"]).strip()] = r["microbe_id"]

    base_groups: Dict[str, List[str]] = {}
    base_col = "taxa_base" if "taxa_base" in audit.columns else None
    if base_col:
        for _, r in audit.iterrows():
            key = str(r[base_col]).strip()
            base_groups.setdefault(key, []).append(r["microbe_id"])

    def resolve(source_value: object) -> Tuple[Optional[str], str]:
        s = str(source_value).strip()
        if s in exact_map:
            return exact_map[s], "exact_raw_taxa_column"

        base = strip_duplicate_suffix(s)
        ids = base_groups.get(base, [])
        if len(ids) == 1:
            return ids[0], "unique_taxa_base"
        if len(ids) > 1:
            return None, "ambiguous_taxa_base"
        return None, "unmapped_microbe"

    return resolve


def build_metabolite_resolver(metabolite_table: pd.DataFrame):
    """Return a function that maps biological edge destination names to metabolite node IDs."""
    candidate_pairs: List[Tuple[str, str]] = []

    for _, r in metabolite_table.iterrows():
        node_id = str(r["node_id"]).strip()
        values = [node_id, strip_known_prefix(node_id, "mtb_")]
        for extra_col in ["compound_clean", "compound_col"]:
            if extra_col in metabolite_table.columns and pd.notna(r.get(extra_col)):
                values.append(str(r[extra_col]).strip())
                values.append(strip_known_prefix(str(r[extra_col]).strip(), "mtb_"))
        for v in values:
            if v:
                candidate_pairs.append((normalize_compound_key(v), node_id))

    groups: Dict[str, List[str]] = {}
    for key, node_id in candidate_pairs:
        groups.setdefault(key, [])
        if node_id not in groups[key]:
            groups[key].append(node_id)

    def resolve(destination_value: object) -> Tuple[Optional[str], str]:
        raw = str(destination_value).strip()
        direct_variants = [raw, f"mtb_{raw}"]
        for v in direct_variants:
            key = normalize_compound_key(v)
            ids = groups.get(key, [])
            if len(ids) == 1:
                return ids[0], "normalized_name"
            if len(ids) > 1:
                return None, "ambiguous_metabolite_name"
        return None, "unmapped_metabolite"

    return resolve


def map_biological_microbe_metabolite_edges(
    seed_dir: Path,
    microbe_audit: pd.DataFrame,
    metabolite_table: pd.DataFrame,
):
    bio = read_csv_checked(BIOLOGICAL_EDGE_PATH)
    required = {"source", "destination", "edge"}
    if not required.issubset(bio.columns):
        raise ValueError(f"{BIOLOGICAL_EDGE_PATH} must contain columns: {sorted(required)}")

    resolve_microbe = build_microbe_source_resolver(microbe_audit)
    resolve_metabolite = build_metabolite_resolver(metabolite_table)

    rows = []
    audit_rows = []
    for _, r in bio.iterrows():
        microbe_id, microbe_status = resolve_microbe(r["source"])
        metabolite_id, metabolite_status = resolve_metabolite(r["destination"])
        edge_label = str(r["edge"]).strip().lower()
        if edge_label not in {"uses", "produces"}:
            edge_label = "biological"

        mapped = microbe_id is not None and metabolite_id is not None

        audit_row = {
            "raw_microbe_source": r.get("source"),
            "raw_metabolite_destination": r.get("destination"),
            "source_kegg": r.get("source_kegg"),
            "destination_kegg": r.get("destination_kegg"),
            "edge": edge_label,
            "microbe_id": microbe_id,
            "metabolite_id": metabolite_id,
            "microbe_mapping_status": microbe_status,
            "metabolite_mapping_status": metabolite_status,
            "mapped": mapped,
        }
        audit_rows.append(audit_row)

        if mapped:
            rows.append({
                "microbe_id": microbe_id,
                "metabolite_id": metabolite_id,
                "edge": edge_label,
                "edge_weight": 1.0,
                "source_kegg": r.get("source_kegg"),
                "destination_kegg": r.get("destination_kegg"),
            })

    audit_df = pd.DataFrame(audit_rows)
    mapped_df = pd.DataFrame(rows).drop_duplicates(
        ["microbe_id", "metabolite_id", "edge"]
    ).reset_index(drop=True)

    out_dir = OUTPUT_DIR / seed_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_dir / "biological_edge_mapping_audit.csv", index=False)
    mapped_df.to_csv(out_dir / "mapped_microbe_metabolite_edges.csv", index=False)

    return mapped_df, audit_df


def load_biological_edges_for_data(
    mapped_df: pd.DataFrame,
    microbe_index: Dict[str, int],
    metabolite_index: Dict[str, int],
    device,
):
    result = {}
    if mapped_df.empty:
        return result

    for edge_label in sorted(mapped_df["edge"].dropna().unique()):
        sub = mapped_df[mapped_df["edge"] == edge_label].copy()
        edge_index, edge_attr = tensor_edge_index_and_attr(
            sub,
            "microbe_id",
            "metabolite_id",
            "edge_weight",
            microbe_index,
            metabolite_index,
            device,
        )
        result[("microbe", edge_label, "metabolite")] = (edge_index, edge_attr, len(sub))

        # Reverse relation so metabolites can also send messages back to microbes.
        rev = sub.rename(columns={"microbe_id": "metabolite_id_tmp", "metabolite_id": "microbe_id_tmp"}).copy()
        rev["metabolite_id"] = rev["metabolite_id_tmp"]
        rev["microbe_id"] = rev["microbe_id_tmp"]
        rev_edge_index, rev_edge_attr = tensor_edge_index_and_attr(
            rev,
            "metabolite_id",
            "microbe_id",
            "edge_weight",
            metabolite_index,
            microbe_index,
            device,
        )
        result[("metabolite", f"rev_{edge_label}", "microbe")] = (rev_edge_index, rev_edge_attr, len(sub))

    return result


# ============================================================
# Plant-to-omic observation edges
# ============================================================
def build_node_id_resolver_from_index(ids: Sequence[str]):
    """Resolve feature-column raw names to graph node IDs using exact and normalized names."""
    exact = {str(x): str(x) for x in ids}
    normalized: Dict[str, List[str]] = {}
    for x in ids:
        sx = str(x)
        variants = [sx, strip_known_prefix(sx, "mtb_")]
        for v in variants:
            normalized.setdefault(normalize_compound_key(v), [])
            if sx not in normalized[normalize_compound_key(v)]:
                normalized[normalize_compound_key(v)].append(sx)

    def resolve(raw_value: str) -> Optional[str]:
        v = str(raw_value)
        if v in exact:
            return exact[v]
        key = normalize_compound_key(v)
        matches = normalized.get(key, [])
        if len(matches) == 1:
            return matches[0]
        return None

    return resolve


def build_plant_to_microbe_edges(
    X_plant: pd.DataFrame,
    microbe_ids: Sequence[str],
) -> pd.DataFrame:
    microbe_set = set(map(str, microbe_ids))
    microbe_cols = [c for c in X_plant.columns if c.startswith("microbiome__")]
    rows = []

    for plant_idx, (_, row) in enumerate(X_plant[microbe_cols].iterrows()):
        pairs = []
        for col, value in row.items():
            microbe_id = col.replace("microbiome__", "", 1)
            if microbe_id not in microbe_set:
                continue
            value = float(value)
            if abs(value) <= OBS_MIN_ABS_VALUE:
                continue
            pairs.append((microbe_id, value))

        if OBSERVATION_EDGE_MODE == "topk_abs":
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:OBS_TOP_K_MICROBES]
        elif OBSERVATION_EDGE_MODE == "topk_positive":
            pairs = [(node_id, value) for node_id, value in pairs if value > 0]
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:OBS_TOP_K_MICROBES]


        for microbe_id, value in pairs:
            rows.append({
                "plant_idx": plant_idx,
                "microbe_id": microbe_id,
                "edge_weight": value,
                "abs_value": abs(value),
            })

    return pd.DataFrame(rows)


def build_plant_to_metabolite_edges(
    X_plant: pd.DataFrame,
    metabolite_ids: Sequence[str],
) -> pd.DataFrame:
    resolve_met = build_node_id_resolver_from_index(metabolite_ids)
    metabolite_cols = [c for c in X_plant.columns if c.startswith("metabolome__")]
    rows = []

    for plant_idx, (_, row) in enumerate(X_plant[metabolite_cols].iterrows()):
        pairs = []
        for col, value in row.items():
            raw_feature = col.replace("metabolome__", "", 1)
            metabolite_id = resolve_met(raw_feature)
            if metabolite_id is None:
                continue
            value = float(value)
            if abs(value) <= OBS_MIN_ABS_VALUE:
                continue
            pairs.append((metabolite_id, value))

        if OBSERVATION_EDGE_MODE == "topk_abs":
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:OBS_TOP_K_METABOLITES]
        elif OBSERVATION_EDGE_MODE == "topk_positive":
            pairs = [(node_id, value) for node_id, value in pairs if value > 0]
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:OBS_TOP_K_METABOLITES]


        for metabolite_id, value in pairs:
            rows.append({
                "plant_idx": plant_idx,
                "metabolite_id": metabolite_id,
                "edge_weight": value,
                "abs_value": abs(value),
            })

    return pd.DataFrame(rows)


def add_observation_edges_to_data(
    data: HeteroData,
    X_plant: pd.DataFrame,
    microbe_table: pd.DataFrame,
    metabolite_table: pd.DataFrame,
    microbe_index: Dict[str, int],
    metabolite_index: Dict[str, int],
    device,
) -> Dict[str, int]:
    counts = {}
    if not ADD_PLANT_OMIC_OBSERVATION_EDGES:
        return counts

    plant_index = {str(i): i for i in range(X_plant.shape[0])}

    pm = build_plant_to_microbe_edges(X_plant, microbe_table["microbe_id"].tolist())
    if not pm.empty:
        pm2 = pm.copy()
        pm2["plant_idx_str"] = pm2["plant_idx"].astype(str)
        edge_index, edge_attr = tensor_edge_index_and_attr(
            pm2, "plant_idx_str", "microbe_id", "edge_weight", plant_index, microbe_index, device
        )
        add_edge_store(data, ("plant", "has_microbe", "microbe"), edge_index, edge_attr)
        counts["plant_has_microbe"] = int(edge_index.shape[1])

        rev = pm2.copy()
        edge_index, edge_attr = tensor_edge_index_and_attr(
            rev, "microbe_id", "plant_idx_str", "edge_weight", microbe_index, plant_index, device
        )
        add_edge_store(data, ("microbe", "rev_has_microbe", "plant"), edge_index, edge_attr)
        counts["microbe_rev_has_microbe_plant"] = int(edge_index.shape[1])

    pmet = build_plant_to_metabolite_edges(X_plant, metabolite_table["node_id"].tolist())
    if not pmet.empty:
        pmet2 = pmet.copy()
        pmet2["plant_idx_str"] = pmet2["plant_idx"].astype(str)
        edge_index, edge_attr = tensor_edge_index_and_attr(
            pmet2, "plant_idx_str", "metabolite_id", "edge_weight", plant_index, metabolite_index, device
        )
        add_edge_store(data, ("plant", "has_metabolite", "metabolite"), edge_index, edge_attr)
        counts["plant_has_metabolite"] = int(edge_index.shape[1])

        edge_index, edge_attr = tensor_edge_index_and_attr(
            pmet2, "metabolite_id", "plant_idx_str", "edge_weight", metabolite_index, plant_index, device
        )
        add_edge_store(data, ("metabolite", "rev_has_metabolite", "plant"), edge_index, edge_attr)
        counts["metabolite_rev_has_metabolite_plant"] = int(edge_index.shape[1])

    return counts



# ============================================================
# Target and feature scaling
# ============================================================
def prepare_target_tensors(plant_table: pd.DataFrame, y_all: pd.DataFrame, device):
    train_mask_np = plant_table["split"].to_numpy() == "train"
    y_np = y_all[TARGET_TRAITS].to_numpy(dtype=np.float32)

    if SCALE_TARGET:
        y_scaler = StandardScaler()
        y_scaled = y_np.copy()
        y_scaled[train_mask_np] = y_scaler.fit_transform(y_np[train_mask_np])
        y_scaled[~train_mask_np] = y_scaler.transform(y_np[~train_mask_np])
    else:
        y_scaler = None
        y_scaled = y_np

    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)
    train_mask = torch.tensor(plant_table["split"].to_numpy() == "train", dtype=torch.bool, device=device)
    val_mask = torch.tensor(plant_table["split"].to_numpy() == "val", dtype=torch.bool, device=device)
    test_mask = torch.tensor(plant_table["split"].to_numpy() == "test", dtype=torch.bool, device=device)
    return y_tensor, train_mask, val_mask, test_mask, y_scaler


def inverse_transform_predictions(arr: np.ndarray, y_scaler):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if y_scaler is not None:
        return y_scaler.inverse_transform(arr)
    return arr


def scale_feature_frame(X: pd.DataFrame, fit_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    arr = X.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    if fit_mask is None:
        arr2 = scaler.fit_transform(arr)
    else:
        arr2 = arr.copy()
        arr2[fit_mask] = scaler.fit_transform(arr[fit_mask])
        arr2[~fit_mask] = scaler.transform(arr[~fit_mask])
    return pd.DataFrame(arr2, columns=X.columns, index=X.index).astype(np.float32)


# ============================================================
# Build HeteroData
# ============================================================
@dataclass
class SeedGraphPayload:
    seed_dir: Path
    data: HeteroData
    plant_table: pd.DataFrame
    microbe_table: pd.DataFrame
    metabolite_table: pd.DataFrame
    feature_names: Dict[str, List[str]]
    y_scaler: Optional[StandardScaler]
    y_all: pd.DataFrame
    edge_count_summary: pd.DataFrame
    biological_audit: pd.DataFrame


def build_hetero_data_for_seed(seed: int, device) -> SeedGraphPayload:
    seed_dir, plant_table, X_plant, y_all = load_seed_plant_data(seed)
    microbe_table, X_microbe, microbe_feature_names, microbe_audit = load_microbe_nodes(seed_dir)
    metabolite_table, X_met, metabolite_feature_names = load_metabolite_nodes(seed_dir)

    # Optional scaling. Plant scaling is disabled by default to match your GNN baseline.
    if SCALE_PLANT_NODE_FEATURES_FOR_GNN:
        train_mask_np = plant_table["split"].to_numpy() == "train"
        X_plant = scale_feature_frame(X_plant, fit_mask=train_mask_np)
    if SCALE_NON_PLANT_NODE_FEATURES:
        X_microbe = scale_feature_frame(X_microbe)
        X_met = scale_feature_frame(X_met)

    y_tensor, train_mask, val_mask, test_mask, y_scaler = prepare_target_tensors(
        plant_table=plant_table,
        y_all=y_all,
        device=device,
    )

    data = HeteroData()
    data["plant"].x = torch.tensor(X_plant.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    data["plant"].y = y_tensor
    data["plant"].train_mask = train_mask
    data["plant"].val_mask = val_mask
    data["plant"].test_mask = test_mask

    data["microbe"].x = torch.tensor(X_microbe.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    data["metabolite"].x = torch.tensor(X_met.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    microbe_index = {m: i for i, m in enumerate(microbe_table["microbe_id"].astype(str).tolist())}
    metabolite_index = {m: i for i, m in enumerate(metabolite_table["node_id"].astype(str).tolist())}

    edge_counts = []

    # Plant-plant edges.
    plant_edges = load_or_compute_plant_edges(seed_dir, plant_table, X_plant)
    if not plant_edges.empty:
        pp_rows = []
        for _, r in plant_edges.iterrows():
            pp_rows.append((str(int(r["source"])), str(int(r["target"])), float(r["edge_weight"])))
        pp_df = make_undirected_id_edges(pp_rows, "source", "target", "edge_weight")
        plant_index = {str(i): i for i in range(len(plant_table))}
        edge_index, edge_attr = tensor_edge_index_and_attr(
            pp_df, "source", "target", "edge_weight", plant_index, plant_index, device
        )
        add_edge_store(data, ("plant", "similar_to", "plant"), edge_index, edge_attr)
        edge_counts.append({"edge_type": "plant__similar_to__plant", "num_edges": int(edge_index.shape[1])})

    # Microbe-microbe edges.
    for etype, (ei, ea, n_source_rows) in load_microbe_microbe_edges(seed_dir, microbe_index, device).items():
        add_edge_store(data, etype, ei, ea)
        edge_counts.append({"edge_type": safe_edge_key(etype), "num_edges": int(ei.shape[1]), "source_rows": n_source_rows})

    # Metabolite-metabolite edges.
    for etype, (ei, ea, n_source_rows) in load_metabolite_metabolite_edges(seed_dir, metabolite_index, device).items():
        add_edge_store(data, etype, ei, ea)
        edge_counts.append({"edge_type": safe_edge_key(etype), "num_edges": int(ei.shape[1]), "source_rows": n_source_rows})

    # Microbe-metabolite biological edges.
    mapped_bio, biological_audit = map_biological_microbe_metabolite_edges(
        seed_dir=seed_dir,
        microbe_audit=microbe_audit,
        metabolite_table=metabolite_table,
    )
    for etype, (ei, ea, n_source_rows) in load_biological_edges_for_data(
        mapped_bio, microbe_index, metabolite_index, device
    ).items():
        add_edge_store(data, etype, ei, ea)
        edge_counts.append({"edge_type": safe_edge_key(etype), "num_edges": int(ei.shape[1]), "source_rows": n_source_rows})

    # Plant-to-omic observation edges.
    obs_counts = add_observation_edges_to_data(
        data=data,
        X_plant=X_plant,
        microbe_table=microbe_table,
        metabolite_table=metabolite_table,
        microbe_index=microbe_index,
        metabolite_index=metabolite_index,
        device=device,
    )
    for name, count in obs_counts.items():
        edge_counts.append({"edge_type": name, "num_edges": int(count)})

    edge_count_summary = pd.DataFrame(edge_counts)
    seed_out = OUTPUT_DIR / seed_dir.name
    seed_out.mkdir(parents=True, exist_ok=True)
    edge_count_summary.to_csv(seed_out / "edge_count_summary.csv", index=False)

    feature_names = {
        "plant": X_plant.columns.tolist(),
        "microbe": microbe_feature_names,
        "metabolite": metabolite_feature_names,
    }

    # Helpful audit copies.
    pd.concat([plant_table.reset_index(drop=True), X_plant.reset_index(drop=True)], axis=1).to_csv(
        seed_out / "hetero_han_plant_node_features_used.csv", index=False
    )
    microbe_table.to_csv(seed_out / "hetero_han_microbe_nodes_used.csv", index=False)
    metabolite_table.to_csv(seed_out / "hetero_han_metabolite_nodes_used.csv", index=False)

    return SeedGraphPayload(
        seed_dir=seed_dir,
        data=data,
        plant_table=plant_table,
        microbe_table=microbe_table,
        metabolite_table=metabolite_table,
        feature_names=feature_names,
        y_scaler=y_scaler,
        y_all=y_all,
        edge_count_summary=edge_count_summary,
        biological_audit=biological_audit,
    )


# ============================================================
# PyTorch Geometric HANConv
# ============================================================
class PyGHANRegressor(nn.Module):
    """Pass all node features and edge types directly into PyG HANConv."""

    def __init__(
        self,
        metadata,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        heads: int = HAN_HEADS,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(
            HANConv(
                in_channels=in_channels_dict,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                dropout=dropout,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                HANConv(
                    in_channels={ntype: hidden_channels for ntype in metadata[0]},
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                    dropout=dropout,
                )
            )

        self.out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        collect_attention: bool = False,
    ):
        semantic_attention_records = []
        h_dict = x_dict

        for layer_idx, conv in enumerate(self.convs):
            if collect_attention:
                h_dict, semantic_attn = conv(
                    h_dict,
                    edge_index_dict,
                    return_semantic_attention_weights=True,
                )
                for node_type, weights in semantic_attn.items():
                    if weights is None:
                        continue
                    for i, value in enumerate(weights.detach().cpu().numpy().tolist()):
                        semantic_attention_records.append({
                            "layer": layer_idx,
                            "node_type": node_type,
                            "incoming_relation": f"semantic_relation_{i}",
                            "semantic_attention": float(value),
                        })
            else:
                h_dict = conv(h_dict, edge_index_dict)

            h_dict = {
                node_type: F.elu(x) if x is not None else None
                for node_type, x in h_dict.items()
            }
            h_dict = {
                node_type: F.dropout(x, p=self.dropout, training=self.training) if x is not None else None
                for node_type, x in h_dict.items()
            }

        pred = self.out(h_dict["plant"])
        if collect_attention:
            return pred, semantic_attention_records, {}
        return pred

    def feature_gate_importance(self, feature_names: Dict[str, List[str]]):
        return pd.DataFrame()

# ============================================================
# Metrics
# ============================================================
def safe_corr(y_true, y_pred, method="pearson"):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    if len(np.unique(y_true)) <= 1:
        return np.nan
    if len(np.unique(y_pred)) <= 1:
        return np.nan
    return pd.Series(y_true).corr(pd.Series(y_pred), method=method)


def evaluate_predictions(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    seed: int,
    model_name: str,
    split_name: str,
):
    rows = []
    for trait in TARGET_TRAITS:
        y_true = y_true_df[trait].to_numpy()
        y_pred = y_pred_df[trait].to_numpy()
        rows.append({
            "seed": seed,
            "model": model_name,
            "split": split_name,
            "trait": trait,
            "n": len(y_true),
            "r2": r2_score(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "mae": mean_absolute_error(y_true, y_pred),
            "pearson": safe_corr(y_true, y_pred, method="pearson"),
            "spearman": safe_corr(y_true, y_pred, method="spearman"),
        })

    metric_df = pd.DataFrame(rows)
    rows.append({
        "seed": seed,
        "model": model_name,
        "split": split_name,
        "trait": "AVERAGE",
        "n": int(metric_df["n"].mean()),
        "r2": metric_df["r2"].mean(),
        "rmse": metric_df["rmse"].mean(),
        "mae": metric_df["mae"].mean(),
        "pearson": metric_df["pearson"].mean(),
        "spearman": metric_df["spearman"].mean(),
    })
    return rows


def get_average_row(rows):
    return [r for r in rows if r["trait"] == "AVERAGE"][0]


def selection_score_from_val_rows(val_rows, best_val_loss):
    avg = get_average_row(val_rows)
    if SELECTION_METRIC == "val_loss":
        return float(best_val_loss)
    if SELECTION_METRIC not in avg:
        raise ValueError(f"Unknown SELECTION_METRIC: {SELECTION_METRIC}")
    return float(avg[SELECTION_METRIC])


def is_better_score(new_score, best_score):
    if best_score is None:
        return True
    if SELECTION_MODE == "min":
        return new_score < best_score
    if SELECTION_MODE == "max":
        return new_score > best_score
    raise ValueError("SELECTION_MODE must be 'min' or 'max'.")


def add_config_to_rows(rows, config_id, config, best_val_loss, selected, selection_score):
    for row in rows:
        row["config_id"] = config_id
        row["selected_config"] = bool(selected)
        row["best_val_loss_scaled"] = float(best_val_loss)
        row["selection_metric"] = SELECTION_METRIC
        row["selection_mode"] = SELECTION_MODE
        row["selection_score"] = float(selection_score)
        for k, v in config.items():
            row[k] = v
    return rows


# ============================================================
# Training and prediction
# ============================================================
def edge_attr_dict_from_data(data: HeteroData):
    return {
        etype: data[etype].edge_attr
        for etype in data.edge_types
        if hasattr(data[etype], "edge_attr")
    }


def build_model_for_data(data: HeteroData, config: dict, device):
    in_channels_dict = {ntype: int(data[ntype].x.shape[1]) for ntype in data.node_types}
    model = PyGHANRegressor(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=config["hidden_channels"],
        out_channels=len(TARGET_TRAITS),
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        heads=HAN_HEADS,
    )
    return model.to(device)


def train_one_model(model, data: HeteroData, lr: float, weight_decay: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_loss = float("inf")
    patience_count = 0
    edge_attr_dict = edge_attr_dict_from_data(data)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
        train_loss = F.mse_loss(pred[data["plant"].train_mask], data["plant"].y[data["plant"].train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            val_loss = F.mse_loss(pred[data["plant"].val_mask], data["plant"].y[data["plant"].val_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 100 == 0:
            print(f"      epoch={epoch:04d} train_loss={train_loss.item():.4f} val_loss={val_loss:.4f}")

        if patience_count >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def predict_plant_split(model, data: HeteroData, mask, y_scaler, target_index):
    model.eval()
    edge_attr_dict = edge_attr_dict_from_data(data)
    with torch.no_grad():
        pred_scaled = model(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict).detach().cpu().numpy()
    pred_selected = pred_scaled[mask.detach().cpu().numpy()]
    pred_original = inverse_transform_predictions(pred_selected, y_scaler)
    return pd.DataFrame(pred_original, columns=TARGET_TRAITS, index=target_index)


def save_prediction_file(seed: int, split_name: str, ids: pd.DataFrame, y_true: pd.DataFrame, y_pred: pd.DataFrame):
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = y_true[TARGET_TRAITS].copy()
    actual.columns = [f"actual_{c}" for c in actual.columns]
    pred = y_pred[TARGET_TRAITS].copy()
    pred.columns = [f"pred_{c}" for c in pred.columns]

    out = pd.concat([ids.reset_index(drop=True), actual.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out.to_csv(out_dir / f"han_{split_name}_predictions.csv", index=False)


def print_average_metrics(val_rows, test_rows):
    val_avg = get_average_row(val_rows)
    test_avg = get_average_row(test_rows)
    print(
        f"    SELECTED VAL  | R2={val_avg['r2']:.3f} RMSE={val_avg['rmse']:.3f} "
        f"MAE={val_avg['mae']:.3f} Pearson={val_avg['pearson']:.3f} Spearman={val_avg['spearman']:.3f}"
    )
    print(
        f"    SELECTED TEST | R2={test_avg['r2']:.3f} RMSE={test_avg['rmse']:.3f} "
        f"MAE={test_avg['mae']:.3f} Pearson={test_avg['pearson']:.3f} Spearman={test_avg['spearman']:.3f}"
    )


def tune_model_on_validation(seed: int, payload: SeedGraphPayload, device):
    data = payload.data
    val_idx = payload.plant_table.index[payload.plant_table["split"] == "val"]
    y_val_true = payload.y_all.loc[val_idx, TARGET_TRAITS].reset_index(drop=True)

    tuning_rows = []
    best_score = None
    best_payload = None

    for config_id, config in enumerate(HYPERPARAM_GRID):
        print(f"    config {config_id}: {config}")

        torch.manual_seed(seed + config_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + config_id)

        model = build_model_for_data(data, config, device)
        model, best_val_loss = train_one_model(
            model=model,
            data=data,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        val_pred_df = predict_plant_split(
            model=model,
            data=data,
            mask=data["plant"].val_mask,
            y_scaler=payload.y_scaler,
            target_index=y_val_true.index,
        )

        val_rows = evaluate_predictions(
            y_true_df=y_val_true,
            y_pred_df=val_pred_df,
            seed=seed,
            model_name=MODEL_NAME,
            split_name="val",
        )
        score = selection_score_from_val_rows(val_rows, best_val_loss)
        val_avg = get_average_row(val_rows)
        print(
            f"      val {SELECTION_METRIC}={score:.4f} | "
            f"R2={val_avg['r2']:.3f} RMSE={val_avg['rmse']:.3f} "
            f"Pearson={val_avg['pearson']:.3f} Spearman={val_avg['spearman']:.3f}"
        )

        config_rows = add_config_to_rows(
            rows=copy.deepcopy(val_rows),
            config_id=config_id,
            config=config,
            best_val_loss=best_val_loss,
            selected=False,
            selection_score=score,
        )
        tuning_rows.extend(config_rows)

        if is_better_score(score, best_score):
            best_score = score
            best_payload = {
                "config_id": config_id,
                "config": dict(config),
                "score": score,
                "best_val_loss": best_val_loss,
                "model_state": copy.deepcopy(model.state_dict()),
                "val_rows": val_rows,
                "val_pred_df": val_pred_df.copy(),
            }

    if best_payload is None:
        raise RuntimeError("No best PyG-HAN config selected.")

    for row in tuning_rows:
        if row["config_id"] == best_payload["config_id"]:
            row["selected_config"] = True

    return best_payload, tuning_rows


# ============================================================
# Attention and feature importance export
# ============================================================
def node_display_ids(payload: SeedGraphPayload):
    return {
        "plant": payload.plant_table["sample_id"].astype(str).tolist(),
        "microbe": payload.microbe_table["microbe_id"].astype(str).tolist(),
        "metabolite": payload.metabolite_table["node_id"].astype(str).tolist(),
    }


def export_attention_tables(seed: int, model, payload: SeedGraphPayload):
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = payload.data
    edge_attr_dict = edge_attr_dict_from_data(data)

    model.eval()
    with torch.no_grad():
        _, semantic_records, edge_attention = model(
            data.x_dict,
            data.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            collect_attention=True,
        )

    semantic_df = pd.DataFrame(semantic_records)
    if not semantic_df.empty:
        semantic_df.insert(0, "seed", seed)
        semantic_df.to_csv(out_dir / "selected_attention_summary.csv", index=False)

    top_df = pd.DataFrame()

    return semantic_df, top_df

def export_feature_gate_importance(seed: int, model, payload: SeedGraphPayload):
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    gate_df = model.feature_gate_importance(payload.feature_names)
    if not gate_df.empty:
        gate_df.insert(0, "seed", seed)
        gate_df.to_csv(out_dir / "selected_feature_gate_importance.csv", index=False)
    return gate_df


def export_gradient_feature_importance(seed: int, model, payload: SeedGraphPayload):
    """
    Export gradient x input attribution using validation loss.

    This is usually more defensible as feature attribution than attention alone.
    """
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    data = payload.data
    edge_attr_dict = edge_attr_dict_from_data(data)

    model.eval()
    x_dict = {
        ntype: data[ntype].x.detach().clone().requires_grad_(True)
        for ntype in data.node_types
    }
    pred = model(x_dict, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
    loss = F.mse_loss(pred[data["plant"].val_mask], data["plant"].y[data["plant"].val_mask])
    model.zero_grad(set_to_none=True)
    loss.backward()

    rows = []
    for ntype, x in x_dict.items():
        if x.grad is None:
            continue
        importance = (x.grad.detach() * x.detach()).abs().mean(dim=0).cpu().numpy()
        for fname, score in zip(payload.feature_names[ntype], importance):
            rows.append({
                "seed": seed,
                "node_type": ntype,
                "feature": fname,
                "gradient_x_input_importance": float(score),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["node_type", "gradient_x_input_importance"], ascending=[True, False]).reset_index(drop=True)
        df.to_csv(out_dir / "selected_gradient_feature_importance.csv", index=False)
    return df


# ============================================================
# Seed-level run
# ============================================================
def run_seed(seed: int, device):
    print("\n==============================")
    print(f"Seed {seed}")
    print("==============================")

    payload = build_hetero_data_for_seed(seed, device)
    data = payload.data

    print(f"Node counts: plant={data['plant'].num_nodes}, microbe={data['microbe'].num_nodes}, metabolite={data['metabolite'].num_nodes}")
    print(f"Plant features: {data['plant'].x.shape[1]}")
    print(f"Microbe features: {data['microbe'].x.shape[1]}")
    print(f"Metabolite features: {data['metabolite'].x.shape[1]}")
    print(f"Edge types: {len(data.edge_types)}")
    print(payload.edge_count_summary.to_string(index=False))
    print(f"Targets: {TARGET_TRAITS}")
    print(f"Selection rule: {SELECTION_METRIC} ({SELECTION_MODE})")

    best_payload, tuning_rows = tune_model_on_validation(seed, payload, device)
    best_config = best_payload["config"]

    print(
        f"  Best config for {MODEL_NAME}: config_id={best_payload['config_id']}, "
        f"{SELECTION_METRIC}={best_payload['score']:.4f}, params={best_config}"
    )

    best_model = build_model_for_data(data, best_config, device)
    best_model.load_state_dict(best_payload["model_state"])

    val_idx = payload.plant_table.index[payload.plant_table["split"] == "val"]
    test_idx = payload.plant_table.index[payload.plant_table["split"] == "test"]

    ids_val = payload.plant_table.loc[val_idx, ID_COLS + ["split"]].reset_index(drop=True)
    ids_test = payload.plant_table.loc[test_idx, ID_COLS + ["split"]].reset_index(drop=True)

    y_val_true = payload.y_all.loc[val_idx, TARGET_TRAITS].reset_index(drop=True)
    y_test_true = payload.y_all.loc[test_idx, TARGET_TRAITS].reset_index(drop=True)

    val_pred_df = best_payload["val_pred_df"]
    test_pred_df = predict_plant_split(
        model=best_model,
        data=data,
        mask=data["plant"].test_mask,
        y_scaler=payload.y_scaler,
        target_index=y_test_true.index,
    )

    val_rows = add_config_to_rows(
        rows=copy.deepcopy(best_payload["val_rows"]),
        config_id=best_payload["config_id"],
        config=best_config,
        best_val_loss=best_payload["best_val_loss"],
        selected=True,
        selection_score=best_payload["score"],
    )

    test_rows = evaluate_predictions(
        y_true_df=y_test_true,
        y_pred_df=test_pred_df,
        seed=seed,
        model_name=MODEL_NAME,
        split_name="test",
    )
    test_rows = add_config_to_rows(
        rows=test_rows,
        config_id=best_payload["config_id"],
        config=best_config,
        best_val_loss=best_payload["best_val_loss"],
        selected=True,
        selection_score=best_payload["score"],
    )

    save_prediction_file(seed, "val", ids_val, y_val_true, val_pred_df)
    save_prediction_file(seed, "test", ids_test, y_test_true, test_pred_df)

    seed_out = OUTPUT_DIR / f"seed_{seed}"
    selected_metrics = pd.DataFrame(val_rows + test_rows)
    tuning_metrics = pd.DataFrame(tuning_rows)
    best_configs = pd.DataFrame([{
        "seed": seed,
        "model": MODEL_NAME,
        "config_id": best_payload["config_id"],
        "selection_metric": SELECTION_METRIC,
        "selection_mode": SELECTION_MODE,
        "selection_score": best_payload["score"],
        "best_val_loss_scaled": best_payload["best_val_loss"],
        "num_node_types": len(data.node_types),
        "num_edge_types": len(data.edge_types),
        **best_config,
    }])

    selected_metrics.to_csv(seed_out / "metrics.csv", index=False)
    tuning_metrics.to_csv(seed_out / "tuning_validation_metrics_all_configs.csv", index=False)
    best_configs.to_csv(seed_out / "best_hyperparameters.csv", index=False)

    semantic_df, edge_top_df = export_attention_tables(seed, best_model, payload)
    gate_df = export_feature_gate_importance(seed, best_model, payload)
    grad_df = export_gradient_feature_importance(seed, best_model, payload)

    print_average_metrics(val_rows, test_rows)

    return selected_metrics, tuning_metrics, best_configs, semantic_df, edge_top_df, gate_df, grad_df


# ============================================================
# Summary
# ============================================================
def summarize_results(
    all_metrics: pd.DataFrame,
    all_tuning_metrics: pd.DataFrame,
    all_best_configs: pd.DataFrame,
    all_semantic: pd.DataFrame,
    all_edge_attention_top: pd.DataFrame,
    all_gate_importance: pd.DataFrame,
    all_grad_importance: pd.DataFrame,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(OUTPUT_DIR / "all_seed_metrics_long.csv", index=False)
    all_tuning_metrics.to_csv(OUTPUT_DIR / "all_tuning_validation_metrics.csv", index=False)
    all_best_configs.to_csv(OUTPUT_DIR / "best_hyperparameters.csv", index=False)

    if not all_semantic.empty:
        all_semantic.to_csv(OUTPUT_DIR / "all_attention_summary.csv", index=False)
    if not all_edge_attention_top.empty:
        all_edge_attention_top.to_csv(OUTPUT_DIR / "all_edge_attention_top.csv", index=False)
    if not all_gate_importance.empty:
        all_gate_importance.to_csv(OUTPUT_DIR / "all_feature_gate_importance.csv", index=False)
    if not all_grad_importance.empty:
        all_grad_importance.to_csv(OUTPUT_DIR / "all_gradient_feature_importance.csv", index=False)

    summary_by_trait = (
        all_metrics
        .groupby(["model", "split", "trait"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
    )
    summary_by_trait.to_csv(OUTPUT_DIR / "summary_by_trait.csv", index=False)

    ranking = (
        all_metrics[all_metrics["trait"] == "AVERAGE"]
        .groupby(["model", "split"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
        .sort_values(["split", "r2_mean"], ascending=[True, False])
    )
    ranking.to_csv(OUTPUT_DIR / "model_ranking_average_traits.csv", index=False)

    print("\n==============================")
    print("Best hyperparameters selected by validation")
    print("==============================")
    print(all_best_configs.to_string(index=False))

    print("\n==============================")
    print("Model ranking, average across traits")
    print("==============================")
    print(ranking.round(4).to_string(index=False))

    print(f"\nSaved feature-complete HAN results to: {OUTPUT_DIR}/")

    return summary_by_trait, ranking


# ============================================================
# Main
# ============================================================
def get_device(preferred_gpu: int = 0):
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch. "
            "This usually means PyTorch was installed as CPU-only, "
            "CUDA_VISIBLE_DEVICES is hiding the GPUs, or the job/session was not assigned a GPU."
        )

    if preferred_gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{preferred_gpu}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )

    torch.cuda.set_device(preferred_gpu)
    device = torch.device(f"cuda:{preferred_gpu}")

    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(device))

    return device


def main():
    check_pyg_available()

    if len(TARGET_TRAITS) == 0:
        raise ValueError("TARGET_TRAITS cannot be empty.")
    if SELECTION_METRIC in {"rmse", "mae", "val_loss"} and SELECTION_MODE != "min":
        raise ValueError(f"For {SELECTION_METRIC}, use SELECTION_MODE='min'.")
    if SELECTION_METRIC in {"r2", "pearson", "spearman"} and SELECTION_MODE != "max":
        raise ValueError(f"For {SELECTION_METRIC}, use SELECTION_MODE='max'.")

    device = get_device(preferred_gpu=3)
    print(f"Using device: {device}")

    all_seed_metrics = []
    all_tuning_metrics = []
    all_best_configs = []
    all_semantic = []
    all_edge_attention_top = []
    all_gate_importance = []
    all_grad_importance = []

    for seed in SEEDS:
        selected_metrics, tuning_metrics, best_configs, semantic_df, edge_top_df, gate_df, grad_df = run_seed(seed, device)
        all_seed_metrics.append(selected_metrics)
        all_tuning_metrics.append(tuning_metrics)
        all_best_configs.append(best_configs)
        if semantic_df is not None and not semantic_df.empty:
            all_semantic.append(semantic_df)
        if edge_top_df is not None and not edge_top_df.empty:
            all_edge_attention_top.append(edge_top_df)
        if gate_df is not None and not gate_df.empty:
            all_gate_importance.append(gate_df)
        if grad_df is not None and not grad_df.empty:
            all_grad_importance.append(grad_df)

    all_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    all_tuning = pd.concat(all_tuning_metrics, ignore_index=True)
    all_best = pd.concat(all_best_configs, ignore_index=True)
    all_semantic_df = pd.concat(all_semantic, ignore_index=True) if all_semantic else pd.DataFrame()
    all_edge_attention_df = pd.concat(all_edge_attention_top, ignore_index=True) if all_edge_attention_top else pd.DataFrame()
    all_gate_df = pd.concat(all_gate_importance, ignore_index=True) if all_gate_importance else pd.DataFrame()
    all_grad_df = pd.concat(all_grad_importance, ignore_index=True) if all_grad_importance else pd.DataFrame()

    summarize_results(
        all_metrics=all_metrics,
        all_tuning_metrics=all_tuning,
        all_best_configs=all_best,
        all_semantic=all_semantic_df,
        all_edge_attention_top=all_edge_attention_df,
        all_gate_importance=all_gate_df,
        all_grad_importance=all_grad_df,
    )


if __name__ == "__main__":
    main()
