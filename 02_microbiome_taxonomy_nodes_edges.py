"""
Create microbiome taxonomy node features and taxonomy edges for each preprocessed seed.

Why this version exists
-----------------------
Your new preprocessing saves microbiome columns as mb_0, mb_1, ... and the
train-fitted prevalence filter can retain a different subset of microbes for
each seed. This script maps those retained mb_* columns back to the original
raw taxonomy strings from raw_dataset/microbiome.csv, then creates:

1) microbiome_node_features_audit.csv
   Human-readable taxonomy table.

2) microbiome_node_features_numeric.csv
   Numeric taxonomy-only node features for GNNs. No abundance values are used.

3) microbiome_taxonomy_edges.csv
   Microbe-microbe edges from exact same genus/species only.

Expected project structure
--------------------------
preprocessed/
  seed_42/
    microbiome_train.csv
    microbiome_val.csv
    microbiome_test.csv
  seed_43/
    ...
raw_dataset/
  microbiome.csv

Run:
  python 02_microbiome_taxonomy_nodes_edges_new.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
BASE_DIR = Path(".")
RAW_MICROBIOME_PATH = BASE_DIR / "raw_dataset" / "microbiome.csv"
PREPROCESSED_ROOT = BASE_DIR / "preprocessed"
GLOBAL_OUT_ROOT = BASE_DIR / "graph_output" / "microbiome_taxonomy"
SEED_GRAPH_DIR_NAME = "graph_microbiome"

SEED_DIR_GLOB = "seed_*"
MICROBIOME_TRAIN_FILE = "microbiome_train.csv"

# These columns are sample/plot identifiers, not microbe/taxon nodes.
ID_COLS = {"sample_id", "variety", "varietyID", "block", "plot"}

RANK_PREFIX = {
    "domain": "d__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}
RANK_ORDER = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# Node feature vocabulary. Domain is usually not useful but harmless; keeping it
# makes the numeric feature matrix fully represent the available taxonomy.
FEATURE_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# Exact taxonomy edges. Keep this specific. Family/order/class cliques can become
# very dense and less biologically specific.
EXACT_EDGE_RANKS = ["species", "genus"]

# Safety controls.
MAKE_UNDIRECTED = True
MAX_EXACT_GROUP_SIZE = 500       # Skip exact clique for groups larger than this.


# ============================================================
# Taxonomy parsing helpers
# ============================================================
def strip_duplicate_suffix(col: str) -> str:
    """Remove pandas duplicate suffixes such as .1, .2, .3."""
    return re.sub(r"\.\d+$", "", str(col))


def parse_taxonomy(col: str) -> Dict[str, Optional[str]]:
    """Extract taxonomy ranks from a raw microbiome taxonomy column."""
    base_col = strip_duplicate_suffix(col)
    parts = [p.strip() for p in base_col.split(";")]
    result = {rank: None for rank in RANK_ORDER}

    for part in parts:
        for rank, prefix in RANK_PREFIX.items():
            if part.startswith(prefix):
                value = part.replace(prefix, "", 1).strip()
                result[rank] = value if value else None
    return result


def is_meaningful_taxon_name(value: object, rank: str) -> bool:
    """Filter vague taxonomy labels before using them as features/edges."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return False

    value_l = str(value).lower().strip()

    bad_terms = [
        "uncultured",
        "unclassified",
        "unknown",
        "unidentified",
        "metagenome",
        "unassigned",
        "ambiguous",
        "environmental",
        "sample",
    ]
    if any(term in value_l for term in bad_terms):
        return False

    if rank == "species":
        generic_species_terms = [
            "bacterium",
            "bacteria",
            "archaeon",
            "organism",
            "prokaryote",
            "endosymbiont",
            "symbiont",
        ]
        if any(term in value_l for term in generic_species_terms):
            return False

    return True


def deepest_rank(row: pd.Series, named_only: bool = False) -> str:
    """Return deepest available taxonomy rank."""
    for rank in reversed(RANK_ORDER):
        value = row.get(rank)
        if named_only:
            if is_meaningful_taxon_name(value, rank):
                return rank
        else:
            if value is not None and pd.notna(value) and str(value).strip() != "":
                return rank
    return "none"


def taxonomy_tokens(row: pd.Series, ranks: Sequence[str]) -> Set[str]:
    """Return meaningful rank=value tokens for one node."""
    tokens = set()
    for rank in ranks:
        value = row.get(rank)
        if is_meaningful_taxon_name(value, rank):
            tokens.add(f"{rank}={value}")
    return tokens




# ============================================================
# Mapping and node feature construction
# ============================================================
def build_raw_taxonomy_mapping(raw_microbiome_path: Path) -> pd.DataFrame:
    """
    Create mb_i -> raw taxonomy mapping.

    In 00_preprocess.py, the raw microbiome CSV is read with index_col=0 and
    then columns are renamed in order to mb_0, mb_1, ... . This recreates the
    same mapping from the raw header.
    """
    if not raw_microbiome_path.exists():
        raise FileNotFoundError(
            f"Raw microbiome file not found: {raw_microbiome_path}. "
            "Update RAW_MICROBIOME_PATH at the top of this script."
        )

    header = pd.read_csv(raw_microbiome_path, nrows=0).columns.tolist()
    if len(header) <= 1:
        raise ValueError("Raw microbiome file does not appear to contain taxa columns.")

    raw_taxa_cols = header[1:]  # first column is the sample/index column
    records = []
    for i, raw_col in enumerate(raw_taxa_cols):
        parsed = parse_taxonomy(raw_col)
        row = {
            "microbe_id": f"mb_{i}",
            "raw_taxa_column": raw_col,
            "taxa_base": strip_duplicate_suffix(raw_col),
        }
        row.update(parsed)
        records.append(row)

    mapping = pd.DataFrame(records)
    mapping["deepest_rank_any"] = mapping.apply(lambda r: deepest_rank(r, named_only=False), axis=1)
    mapping["deepest_rank_named"] = mapping.apply(lambda r: deepest_rank(r, named_only=True), axis=1)
    for rank in RANK_ORDER:
        mapping[f"has_{rank}"] = mapping[rank].notna()
        mapping[f"has_named_{rank}"] = mapping[rank].apply(lambda x: is_meaningful_taxon_name(x, rank))
    return mapping


def make_global_feature_vocab(mapping: pd.DataFrame) -> List[str]:
    """Build stable taxonomy-token vocabulary from the full raw microbiome file."""
    vocab = set()
    for _, row in mapping.iterrows():
        vocab.update(taxonomy_tokens(row, FEATURE_RANKS))
    return sorted(vocab)


def make_node_tables(mapping: pd.DataFrame, retained_microbe_ids: List[str], vocab: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create audit and numeric node feature tables for retained microbes."""
    node_audit = (
        mapping[mapping["microbe_id"].isin(retained_microbe_ids)]
        .set_index("microbe_id")
        .loc[retained_microbe_ids]
        .reset_index()
        .copy()
    )

    vocab_index = {tok: j for j, tok in enumerate(vocab)}
    X = np.zeros((len(node_audit), len(vocab)), dtype=np.uint8)

    named_rank_count = []
    deepest_rank_named = []
    for i, (_, row) in enumerate(node_audit.iterrows()):
        toks = taxonomy_tokens(row, FEATURE_RANKS)
        for tok in toks:
            if tok in vocab_index:
                X[i, vocab_index[tok]] = 1
        named_rank_count.append(len(toks))
        deepest_rank_named.append(deepest_rank(row, named_only=True))

    numeric = pd.DataFrame(X, columns=[f"tax__{tok}" for tok in vocab])
    numeric.insert(0, "microbe_id", node_audit["microbe_id"].values)

    # Compact numeric quality/depth features.
    numeric["tax__named_rank_count"] = named_rank_count
    for rank in RANK_ORDER + ["none"]:
        numeric[f"tax__deepest_named_is_{rank}"] = [1 if r == rank else 0 for r in deepest_rank_named]

    return node_audit, numeric


# ============================================================
# Edge construction
# ============================================================
def add_edge(
    edge_rows: List[dict],
    seen_pairs: Set[Tuple[str, str, str]],
    source: str,
    target: str,
    edge_type: str,
    edge_weight: float,
    shared_rank: str,
    shared_value: str,
    jaccard_value: float,
):
    """Add one edge, optionally adding reverse direction too."""
    if source == target:
        return

    def _one(s: str, t: str):
        key = (s, t, edge_type)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        edge_rows.append({
            "source": s,
            "target": t,
            "edge_type": edge_type,
            "edge_weight": round(float(edge_weight), 6),
            "shared_rank": shared_rank,
            "shared_value": shared_value,
            "jaccard": round(float(jaccard_value), 6),
        })

    _one(source, target)
    if MAKE_UNDIRECTED:
        _one(target, source)


def create_exact_rank_edges(node_audit: pd.DataFrame) -> Tuple[List[dict], Set[Tuple[str, str]]]:
    """Create same-species and same-genus edges, deepest rank first."""
    edge_rows: List[dict] = []
    seen_edge_keys: Set[Tuple[str, str, str]] = set()
    pair_already_connected: Set[Tuple[str, str]] = set()

    for rank in EXACT_EDGE_RANKS:
        valid = node_audit[node_audit[rank].apply(lambda x: is_meaningful_taxon_name(x, rank))]
        for value, group in valid.groupby(rank, dropna=True):
            ids = group["microbe_id"].tolist()
            if len(ids) < 2:
                continue
            if len(ids) > MAX_EXACT_GROUP_SIZE:
                print(f"Skipping exact {rank} clique for value={value!r}; group size={len(ids)}")
                continue

            for source, target in combinations(ids, 2):
                pair_key = tuple(sorted((source, target)))
                # If already connected by a deeper exact rank, do not add a duplicate weaker exact edge.
                if pair_key in pair_already_connected:
                    continue
                add_edge(
                    edge_rows=edge_rows,
                    seen_pairs=seen_edge_keys,
                    source=source,
                    target=target,
                    edge_type=f"same_{rank}",
                    edge_weight=1.0,
                    shared_rank=rank,
                    shared_value=str(value),
                    jaccard_value=1.0,
                )
                pair_already_connected.add(pair_key)

    return edge_rows, pair_already_connected



def create_edges(node_audit: pd.DataFrame) -> pd.DataFrame:
    """Create only exact same-species and same-genus microbiome taxonomy edges."""
    exact_edges, _ = create_exact_rank_edges(node_audit)
    edges = pd.DataFrame(exact_edges)

    if edges.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "edge_type",
                "edge_weight",
                "shared_rank",
                "shared_value",
                "jaccard",
            ]
        )

    return edges.sort_values(["edge_type", "source", "target"]).reset_index(drop=True)


# ============================================================
# Seed processing
# ============================================================
def retained_microbes_from_seed(seed_dir: Path) -> List[str]:
    """Get retained mb_* columns from one seed's microbiome_train.csv."""
    path = seed_dir / MICROBIOME_TRAIN_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    microbes = [c for c in cols if c not in ID_COLS and c.startswith("mb_")]
    if not microbes:
        raise ValueError(f"No mb_* microbe columns found in {path}")
    return microbes


def save_summary(seed_out: Path, seed_name: str, node_audit: pd.DataFrame, numeric: pd.DataFrame, edges: pd.DataFrame):
    edge_counts = edges["edge_type"].value_counts().to_dict() if not edges.empty else {}
    deepest_counts = node_audit["deepest_rank_named"].value_counts().to_dict()
    summary = {
        "seed": seed_name,
        "num_microbe_nodes": int(len(node_audit)),
        "num_numeric_feature_columns_excluding_id": int(numeric.shape[1] - 1),
        "num_edges": int(len(edges)),
        "make_undirected": MAKE_UNDIRECTED,
        "edge_counts_by_type": {str(k): int(v) for k, v in edge_counts.items()},
        "deepest_named_rank_counts": {str(k): int(v) for k, v in deepest_counts.items()},
        "config": {
            "exact_edge_ranks": EXACT_EDGE_RANKS,
            "edge_mode": "exact_same_species_and_same_genus_only",
            "max_exact_group_size": MAX_EXACT_GROUP_SIZE,
        },
    }
    with open(seed_out / "microbiome_taxonomy_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def process_seed(seed_dir: Path, mapping: pd.DataFrame, vocab: List[str]) -> dict:
    seed_name = seed_dir.name

    # Save seed-specific microbiome graph directly inside each seed folder
    seed_out = seed_dir / SEED_GRAPH_DIR_NAME
    seed_out.mkdir(parents=True, exist_ok=True)

    retained = retained_microbes_from_seed(seed_dir)

    missing = sorted(set(retained) - set(mapping["microbe_id"]))
    if missing:
        raise ValueError(
            f"{seed_name}: retained microbes are not present in raw mapping. "
            f"First few missing: {missing[:10]}"
        )

    node_audit, numeric = make_node_tables(mapping, retained, vocab)
    edges = create_edges(node_audit)

    audit_path = seed_out / "microbiome_node_features_audit.csv"
    numeric_path = seed_out / "microbiome_node_features_numeric.csv"
    edges_path = seed_out / "microbiome_taxonomy_edges.csv"

    node_audit.to_csv(audit_path, index=False)
    numeric.to_csv(numeric_path, index=False)
    edges.to_csv(edges_path, index=False)

    summary = save_summary(seed_out, seed_name, node_audit, numeric, edges)

    print(f"\n[{seed_name}]")
    print(f"  nodes: {summary['num_microbe_nodes']}")
    print(f"  numeric feature cols: {summary['num_numeric_feature_columns_excluding_id']}")
    print(f"  edges: {summary['num_edges']} | {summary['edge_counts_by_type']}")
    print(f"  saved: {seed_out}")
    return summary


def main():
    GLOBAL_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mapping = build_raw_taxonomy_mapping(RAW_MICROBIOME_PATH)
    vocab = make_global_feature_vocab(mapping)

    mapping_path = GLOBAL_OUT_ROOT / "microbiome_mb_to_taxonomy_mapping.csv"
    vocab_path = GLOBAL_OUT_ROOT / "microbiome_taxonomy_feature_vocab.csv"
    mapping.to_csv(mapping_path, index=False)
    pd.DataFrame({"feature_token": vocab, "feature_column": [f"tax__{v}" for v in vocab]}).to_csv(vocab_path, index=False)

    seed_dirs = sorted([p for p in PREPROCESSED_ROOT.glob(SEED_DIR_GLOB) if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed folders found under {PREPROCESSED_ROOT}/{SEED_DIR_GLOB}. "
            "Run 00_preprocess.py first, or update PREPROCESSED_ROOT."
        )

    all_summaries = []
    for seed_dir in seed_dirs:
        all_summaries.append(process_seed(seed_dir, mapping, vocab))

    rows = []
    for s in all_summaries:
        row = {k: v for k, v in s.items() if k not in {"edge_counts_by_type", "deepest_named_rank_counts", "config"}}
        for et, count in s["edge_counts_by_type"].items():
            row[f"edges__{et}"] = count
        rows.append(row)

    pd.DataFrame(rows).to_csv(
    GLOBAL_OUT_ROOT / "all_seed_microbiome_taxonomy_summary.csv",
    index=False
    )

    print(f"\nDone. Seed-specific outputs saved under: preprocessed/seed_*/{SEED_GRAPH_DIR_NAME}/")
    print(f"Global summary outputs saved under: {GLOBAL_OUT_ROOT}")
    print(f"Global mapping: {mapping_path}")
    print(f"Global feature vocab: {vocab_path}")


if __name__ == "__main__":
    main()
