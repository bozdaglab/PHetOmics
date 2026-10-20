from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


from metabolome_seed_helpers import (
    get_global_metabolite_columns,
    get_seed_metabolite_columns,
    clean_compound_name,
    ensure_mtb_prefix,
    PREPROCESSED_DIR,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(".")
PREPROCESSED_DIR = BASE_DIR / "preprocessed"

OUT_DIR = BASE_DIR / "graph_output"
MET_DIR = OUT_DIR / "metabolome_global"
SMILE_DIR = OUT_DIR / "compound_smiles_mapping"

SMILES_PATH = SMILE_DIR / "compound_smiles_final.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# User settings
# ============================================================
# TOP_K = 10
SIMILARITY_THRESHOLD = 0.50
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
BATCH_SIZE = 16

CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
MOLFORMER_MODEL = "ibm/MoLFormer-XL-both-10pct"


# ============================================================
# Helpers
# ============================================================

def create_smiles_template(compound_cols: list[str]):
    template = pd.DataFrame({
        "compound_col": compound_cols,
        "compound_clean": [clean_compound_name(c) for c in compound_cols],
        "smiles": "",
    })
    out_path = MET_DIR / "compound_smiles_template.csv"
    template.to_csv(out_path, index=False)

    print("\nSMILES file not found.")
    print(f"Created template: {out_path}")
    print("\nFill the 'smiles' column, then save it as:")
    print(f"{SMILES_PATH}")
    print("\nThen rerun this script.")
    raise SystemExit


def load_smiles(compound_cols: list[str]) -> pd.DataFrame:
    if not SMILES_PATH.exists():
        create_smiles_template(compound_cols)

    smiles_df = pd.read_csv(SMILES_PATH)
    

    required_cols = {"compound_col", "smiles"}
    if not required_cols.issubset(smiles_df.columns):
        raise ValueError(
            "compound_smiles.csv must contain columns: compound_col, smiles"
        )
    smiles_df["compound_col"] = smiles_df["compound_col"].apply(ensure_mtb_prefix)
    smiles_df["smiles"] = smiles_df["smiles"].astype(str).str.strip()
    smiles_df = smiles_df[smiles_df["smiles"].notna()]
    smiles_df = smiles_df[smiles_df["smiles"] != ""]
    smiles_df = smiles_df[smiles_df["smiles"].str.lower() != "nan"]

    missing = sorted(set(compound_cols) - set(smiles_df["compound_col"]))
    if missing:
        print(f"\nWarning: {len(missing)} compounds do not have SMILES.")
        print("They will be removed from chemical-embedding graph construction.")
        pd.DataFrame({"missing_compound_col": missing}).to_csv(
            MET_DIR / "missing_smiles_compounds.csv",
            index=False
        )

    smiles_df = smiles_df[smiles_df["compound_col"].isin(compound_cols)].copy()
    smiles_df = smiles_df.drop_duplicates("compound_col").reset_index(drop=True)

    return smiles_df


def plot_hist(values, title, xlabel, out_path, bins=50):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def audit_embedding_matrix(name: str, X: np.ndarray) -> pd.DataFrame:
    flat = X.reshape(-1)
    row_norms = np.linalg.norm(X, axis=1)

    summary = pd.DataFrame([{
        "embedding_name": name,
        "num_nodes": X.shape[0],
        "num_features": X.shape[1],
        "value_min": np.min(flat),
        "value_max": np.max(flat),
        "value_mean": np.mean(flat),
        "value_std": np.std(flat),
        "value_median": np.median(flat),
        "norm_min": np.min(row_norms),
        "norm_max": np.max(row_norms),
        "norm_mean": np.mean(row_norms),
        "norm_std": np.std(row_norms),
        "norm_median": np.median(row_norms),
    }])

    plot_hist(
        flat,
        f"{name} raw embedding value distribution",
        "Embedding values",
        MET_DIR / f"{name}_embedding_value_distribution.png"
    )

    plot_hist(
        row_norms,
        f"{name} embedding L2 norm distribution",
        "L2 norm",
        MET_DIR / f"{name}_embedding_norm_distribution.png"
    )

    return summary


def cosine_to_0_1(S: np.ndarray) -> np.ndarray:
    return (S + 1.0) / 2.0


def similarity_pairs(compounds: list[str], S: np.ndarray, score_name: str) -> pd.DataFrame:
    rows = []
    n = len(compounds)

    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "source": compounds[i],
                "target": compounds[j],
                score_name: S[i, j],
            })

    return pd.DataFrame(rows)


def threshold_edges_from_similarity(
    compounds: list[str],
    S: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Create all undirected metabolite-metabolite edges whose similarity
    is greater than or equal to the threshold.

    This does NOT use top-k.
    """
    rows = []
    n = len(compounds)

    if n < 2:
        return pd.DataFrame(
            columns=["source", "target", "edge_type", "similarity"]
        )

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(S[i, j])

            if sim >= threshold:
                rows.append({
                    "source": compounds[i],
                    "target": compounds[j],
                    "edge_type": "metabolome_metabolome_similarity",
                    "similarity": sim,
                })

    edges = pd.DataFrame(rows)

    if edges.empty:
        return pd.DataFrame(
            columns=["source", "target", "edge_type", "similarity"]
        )

    edges = edges.sort_values(
        "similarity",
        ascending=False
    ).reset_index(drop=True)

    return edges

# ============================================================
# RDKit features
# ============================================================
def compute_rdkit_and_morgan(smiles_list: list[str]):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    descriptor_funcs = Descriptors.descList
    descriptor_names = [d[0] for d in descriptor_funcs]

    rdkit_rows = []
    morgan_rows = []
    valid_idx = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        desc_vals = []
        for _, func in descriptor_funcs:
            try:
                value = func(mol)
            except Exception:
                value = np.nan
            desc_vals.append(value)

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=MORGAN_RADIUS,
            nBits=MORGAN_N_BITS
        )
        arr = np.zeros((MORGAN_N_BITS,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)

        rdkit_rows.append(desc_vals)
        morgan_rows.append(arr)
        valid_idx.append(idx)

    rdkit_df = pd.DataFrame(rdkit_rows, columns=descriptor_names)
    rdkit_df = rdkit_df.replace([np.inf, -np.inf], np.nan)
    rdkit_df = rdkit_df.fillna(rdkit_df.median(numeric_only=True))
    rdkit_df = rdkit_df.fillna(0)

    # Scale RDKit descriptors because different descriptors have very different units.
    rdkit_scaled = StandardScaler().fit_transform(rdkit_df)

    morgan_arr = np.vstack(morgan_rows).astype(float)

    return valid_idx, rdkit_scaled, morgan_arr, rdkit_df


# ============================================================
# Transformer embeddings
# ============================================================
def mean_pooling(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def compute_transformer_embeddings(smiles_list: list[str], model_name: str, name: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading {name}: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
    
)
    model = model.to(device)
    model.eval()

    all_embeddings = []

    for start in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

            if hasattr(output, "last_hidden_state"):
                emb = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
            else:
                emb = output[0][:, 0, :]

        all_embeddings.append(emb.detach().cpu().numpy())

    return np.vstack(all_embeddings)

def save_seed_specific_metabolome_graphs(
    global_node_features: pd.DataFrame,
    global_similarity: np.ndarray,
    global_compounds: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
):
    """
    Save metabolite node features and threshold-based metabolite-metabolite edges
    for every seed folder.

    Expensive chemical features are computed once globally.
    Here we only subset to the metabolites present in each seed and keep
    all metabolite pairs with similarity >= threshold.
    """
    compound_to_idx = {
        compound: idx for idx, compound in enumerate(global_compounds)
    }

    seed_dirs = sorted(PREPROCESSED_DIR.glob("seed_*"))

    if not seed_dirs:
        raise FileNotFoundError(f"No seed folders found under {PREPROCESSED_DIR}")

    summary_rows = []

    for seed_dir in seed_dirs:
        seed_compounds_all = get_seed_metabolite_columns(seed_dir)

        seed_compounds = [
            compound for compound in seed_compounds_all
            if compound in compound_to_idx
        ]

        missing_from_global = sorted(set(seed_compounds_all) - set(seed_compounds))

        seed_out_dir = seed_dir / "graph_metabolome"
        seed_out_dir.mkdir(parents=True, exist_ok=True)

        if missing_from_global:
            pd.DataFrame({
                "missing_metabolite_node": missing_from_global
            }).to_csv(
                seed_out_dir / "metabolites_missing_global_features_or_smiles.csv",
                index=False
            )

        if len(seed_compounds) == 0:
            print(f"{seed_dir.name}: no metabolite nodes with global features.")
            continue

        # Save node features in the same order as metabolome columns.
        seed_node_features = (
            global_node_features
            .set_index("node_id")
            .loc[seed_compounds]
            .reset_index()
        )

        seed_node_features.to_csv(
            seed_out_dir / "metabolome_node_features.csv",
            index=False
        )

        # Subset similarity matrix to this seed's available metabolites.
        idx = [compound_to_idx[compound] for compound in seed_compounds]
        seed_similarity = global_similarity[np.ix_(idx, idx)]

        # Optional audit file: all pairwise similarities for this seed.
        seed_all_pairs = similarity_pairs(
            seed_compounds,
            seed_similarity,
            "avg_similarity"
        )

        seed_all_pairs.to_csv(
            seed_out_dir / "metabolome_similarity_pairs.csv",
            index=False
        )

        # Actual graph edges: all pairs above threshold.
        seed_edges = threshold_edges_from_similarity(
            seed_compounds,
            seed_similarity,
            threshold=threshold
        )

        seed_edges.to_csv(
            seed_out_dir / f"metabolome_edges_similarity_ge_{threshold:.2f}.csv",
            index=False
        )

        summary_rows.append({
            "seed": seed_dir.name,
            "num_metabolite_nodes": len(seed_node_features),
            "num_threshold_edges": len(seed_edges),
            "threshold": threshold,
            "num_missing_global_features_or_smiles": len(missing_from_global),
        })

        print(
            f"{seed_dir.name}: saved "
            f"{len(seed_node_features)} metabolite nodes and "
            f"{len(seed_edges)} edges with similarity >= {threshold}"
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUT_DIR / "metabolome_seed_threshold_edge_summary.csv",
        index=False
    )

    print("\nSeed-specific metabolome threshold edge summary:")
    print(summary_df.to_string(index=False))
# ============================================================
# Main
# ============================================================
def main():
    print("\nCollecting metabolite columns from all seed folders...")

    compound_cols = get_global_metabolite_columns()

    print(f"Number of unique metabolite columns/nodes across all seeds: {len(compound_cols)}")

    smiles_df = load_smiles(compound_cols)

    compounds = smiles_df["compound_col"].tolist()
    compound_clean = [clean_compound_name(c) for c in compounds]
    smiles = smiles_df["smiles"].tolist()

    print(f"Compounds with SMILES: {len(compounds)}")

    # ------------------------------------------------------------
    # Node table
    # ------------------------------------------------------------
    node_df = pd.DataFrame({
        "node_id": compounds,
        "node_type": "metabolite",
        "compound_col": compounds,
        "compound_clean": compound_clean,
        "smiles": smiles,
    })

    node_df.to_csv(OUT_DIR / "metabolome_node_features.csv", index=False)

    # ------------------------------------------------------------
    # RDKit and Morgan
    # ------------------------------------------------------------
    print("\nComputing RDKit descriptors and Morgan fingerprints...")
    valid_idx, rdkit_X, morgan_X, rdkit_raw_df = compute_rdkit_and_morgan(smiles)

    node_df = node_df.iloc[valid_idx].reset_index(drop=True)
    compounds = node_df["node_id"].tolist()
    smiles = node_df["smiles"].tolist()

    rdkit_raw_df = rdkit_raw_df.reset_index(drop=True)
    rdkit_raw_df.insert(0, "node_id", compounds)
    rdkit_raw_df.to_csv(MET_DIR / "rdkit_descriptors_raw.csv", index=False)

    # ------------------------------------------------------------
    # Transformer embeddings
    # ------------------------------------------------------------
    print("\nComputing ChemBERTa embeddings...")
    chemberta_X = compute_transformer_embeddings(
        smiles,
        CHEMBERTA_MODEL,
        "chemberta"
    )

    print("\nComputing MolFormer embeddings...")
    molformer_X = compute_transformer_embeddings(
        smiles,
        MOLFORMER_MODEL,
        "molformer"
    )

    # ------------------------------------------------------------
    # Audit embedding ranges
    # ------------------------------------------------------------
    audit_tables = []

    audit_tables.append(audit_embedding_matrix("chemberta", chemberta_X))
    audit_tables.append(audit_embedding_matrix("molformer", molformer_X))
    audit_tables.append(audit_embedding_matrix("rdkit_scaled", rdkit_X))
    audit_tables.append(audit_embedding_matrix("morgan", morgan_X))

    audit_df = pd.concat(audit_tables, ignore_index=True)
    audit_df.to_csv(MET_DIR / "embedding_scale_audit.csv", index=False)

    print("\nEmbedding scale audit:")
    print(audit_df.to_string(index=False))

    # ------------------------------------------------------------
    # Save node features
    # ------------------------------------------------------------
    chemberta_df = pd.DataFrame(
        chemberta_X,
        columns=[f"chemberta_{i}" for i in range(chemberta_X.shape[1])]
    )

    molformer_df = pd.DataFrame(
        molformer_X,
        columns=[f"molformer_{i}" for i in range(molformer_X.shape[1])]
    )

    rdkit_df = pd.DataFrame(
        rdkit_X,
        columns=[f"rdkit_{i}" for i in range(rdkit_X.shape[1])]
    )

    morgan_df = pd.DataFrame(
        morgan_X,
        columns=[f"morgan_{i}" for i in range(morgan_X.shape[1])]
    )

    node_features = pd.concat(
        [
            node_df[["node_id", "node_type", "compound_clean", "smiles"]],
            chemberta_df,
            molformer_df,
            rdkit_df,
            morgan_df,
        ],
        axis=1
    )

    # node_features.to_csv(OUT_DIR / "metabolome_node_features.csv", index=False)

    node_features.to_csv(MET_DIR / "metabolome_node_features.csv", index=False)

    # Optional backward-compatible copy
    node_features.to_csv(OUT_DIR / "metabolome_node_features.csv", index=False)

    # ------------------------------------------------------------
    # Similarity matrices
    # ------------------------------------------------------------
    print("\nComputing similarity matrices...")

    chemberta_sim = cosine_to_0_1(cosine_similarity(chemberta_X))
    molformer_sim = cosine_to_0_1(cosine_similarity(molformer_X))
    rdkit_sim = cosine_to_0_1(cosine_similarity(rdkit_X))

    # For Morgan binary fingerprints, cosine is okay, but Tanimoto is more chemical-standard.
    from rdkit import DataStructs
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=MORGAN_RADIUS,
            nBits=MORGAN_N_BITS
        )
        fps.append(fp)

    n = len(fps)
    morgan_sim = np.eye(n)

    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        morgan_sim[i, :] = np.array(sims)

    avg_sim = np.mean(
        np.stack(
            [
                chemberta_sim,
                molformer_sim,
                rdkit_sim,
                morgan_sim,
            ],
            axis=0
        ),
        axis=0
    )

    np.save(MET_DIR / "chemberta_similarity.npy", chemberta_sim)
    np.save(MET_DIR / "molformer_similarity.npy", molformer_sim)
    np.save(MET_DIR / "rdkit_similarity.npy", rdkit_sim)
    np.save(MET_DIR / "morgan_similarity.npy", morgan_sim)
    np.save(MET_DIR / "average_similarity.npy", avg_sim)
    pd.DataFrame({
    "matrix_index": range(len(compounds)),
    "node_id": compounds,
    }).to_csv(MET_DIR / "metabolite_similarity_matrix_node_order.csv", index=False)

    # ------------------------------------------------------------
    # Similarity pair table
    # ------------------------------------------------------------
    pair_df = similarity_pairs(compounds, avg_sim, "avg_similarity")

    pair_df["chemberta_similarity"] = similarity_pairs(
        compounds, chemberta_sim, "chemberta_similarity"
    )["chemberta_similarity"]

    pair_df["molformer_similarity"] = similarity_pairs(
        compounds, molformer_sim, "molformer_similarity"
    )["molformer_similarity"]

    pair_df["rdkit_similarity"] = similarity_pairs(
        compounds, rdkit_sim, "rdkit_similarity"
    )["rdkit_similarity"]

    pair_df["morgan_similarity"] = similarity_pairs(
        compounds, morgan_sim, "morgan_similarity"
    )["morgan_similarity"]

    pair_df = pair_df.sort_values("avg_similarity", ascending=False).reset_index(drop=True)
    pair_df.to_csv(MET_DIR / "metabolome_similarity_pairs.csv", index=False)

    # ------------------------------------------------------------
    # Similarity distribution plots
    # ------------------------------------------------------------
    upper = np.triu_indices_from(avg_sim, k=1)

    sim_summary = []

    for name, S in {
        "chemberta": chemberta_sim,
        "molformer": molformer_sim,
        "rdkit": rdkit_sim,
        "morgan": morgan_sim,
        "average": avg_sim,
    }.items():
        vals = S[upper]

        sim_summary.append({
            "similarity_name": name,
            "min": vals.min(),
            "max": vals.max(),
            "mean": vals.mean(),
            "std": vals.std(),
            "median": np.median(vals),
            "p75": np.percentile(vals, 75),
            "p90": np.percentile(vals, 90),
            "p95": np.percentile(vals, 95),
            "p99": np.percentile(vals, 99),
        })

        plot_hist(
            vals,
            f"{name} similarity distribution",
            "Similarity score",
            MET_DIR / f"{name}_similarity_distribution.png"
        )

    sim_summary_df = pd.DataFrame(sim_summary)
    sim_summary_df.to_csv(MET_DIR / "similarity_distribution_summary.csv", index=False)

    print("\nSimilarity distribution summary:")
    print(sim_summary_df.to_string(index=False))

    # ------------------------------------------------------------
    # Threshold edges
    # ------------------------------------------------------------
    edges = threshold_edges_from_similarity(
        compounds,
        avg_sim,
        threshold=SIMILARITY_THRESHOLD,
    )

    edges.to_csv(
        MET_DIR / f"metabolome_edges_similarity_ge_{SIMILARITY_THRESHOLD:.2f}.csv",
        index=False
    )

    print(
        f"\nGlobal threshold edges saved: {len(edges)} "
        f"with avg_similarity >= {SIMILARITY_THRESHOLD}"
    )

    save_seed_specific_metabolome_graphs(
        global_node_features=node_features,
        global_similarity=avg_sim,
        global_compounds=compounds,
        threshold=SIMILARITY_THRESHOLD,
    )

    print("\nTop 20 threshold metabolome edges:")
    print(edges.head(20).to_string(index=False))

    print("\nSaved outputs:")
    print(f" - Global node features: {MET_DIR / 'metabolome_node_features.csv'}")
    print(f" - Global all-pair similarities: {MET_DIR / 'metabolome_similarity_pairs.csv'}")
    print(f" - Global threshold edges: {MET_DIR / f'metabolome_edges_similarity_ge_{SIMILARITY_THRESHOLD:.2f}.csv'}")
    print(f" - Similarity matrix order: {MET_DIR / 'metabolite_similarity_matrix_node_order.csv'}")
    print(f" - Seed edge summary: {OUT_DIR / 'metabolome_seed_threshold_edge_summary.csv'}")
    print(f" - Seed-specific files saved under: preprocessed/seed_*/graph_metabolome/")


if __name__ == "__main__":
    main()