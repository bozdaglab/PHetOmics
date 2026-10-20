
"""
Train tuned GNN models on preprocessed multi-omics plot-level files.
===================================================================

Models:
  - GCN
  - GraphSAGE
  - GIN

Design:
  1. Each plot is one node.
  2. Node features = genotype + metabolome + microbiome features concatenated.
  3. Edges are built from node-node Pearson correlation similarity.
  4. Pearson edges are selected by target average degree k=5 and cached inside each preprocessed seed folder.
  5. If an edge file already exists, it is loaded instead of recomputed.
  6. Supports either one target trait or multiple phenotype traits.
  7. Hyperparameter tuning is done on validation set only.
  8. Test set is evaluated only for the best validation-selected config.
  9. Reports R2, RMSE, MAE, Pearson, and Spearman.

Expected input folder:
preprocessed/
  seed_42/
    genotype_train.csv
    metabolome_train.csv
    microbiome_train.csv
    y_train.csv
    genotype_val.csv
    metabolome_val.csv
    microbiome_val.csv
    y_val.csv
    genotype_test.csv
    metabolome_test.csv
    microbiome_test.csv
    y_test.csv

Outputs:
preprocessed/seed_42/
  merged_node_features.csv
  merged_y_targets.csv
  node_table.csv
  edges_pearson_avgdeg_5p0_minsim_0p0.csv

gnn_results_tuned/
  all_seed_metrics_long.csv
  summary_by_trait.csv
  model_ranking_average_traits.csv
  all_tuning_validation_metrics.csv
  best_hyperparameters.csv
  seed_42/
    metrics.csv
    tuning_validation_metrics_all_configs.csv
    best_hyperparameters.csv
    gcn_pearson_val_predictions.csv
    gcn_pearson_test_predictions.csv
    ...
"""

from pathlib import Path
import copy
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Imports for ML and GNN
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv, GINConv
    from torch_geometric.utils import add_self_loops

    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    PYG_IMPORT_ERROR = e


# ============================================================
# Config
# ============================================================

PREPROCESSED_DIR = Path("preprocessed")
OUTPUT_DIR = Path("gnn_results_tuned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

ID_COLS = ["sample_id", "variety", "varietyID", "block", "plot"]

# Use one trait:
TARGET_TRAITS = ["ShootDryWeight_g"]

# Or use multiple traits:
# TARGET_TRAITS = [
#     "ShootDryWeight_g",
#     "ShootFreshWeight_g",
#     "GrowthStage",
#     "NumberOfNodes",
#     "NumberOfTillers",
#     "PlantHeight_cm",
#     "TillerLength_cm",
# ]

OMICS_TO_USE = ["genotype", "metabolome", "microbiome"]

# Build and train one graph per method.
EDGE_METHODS = ["pearson"]

# Edge construction settings
#
# We keep Pearson as the plant-plant similarity metric, but instead of choosing
# an arbitrary fixed threshold, we retain the strongest Pearson edges required
# to obtain the requested average degree k.
#
# For an undirected graph:
#     average_degree = 2 * num_edges / num_nodes
# Therefore, for target k=5, the code keeps approximately:
#     num_edges = num_nodes * 5 / 2
EDGE_SELECTION_MODE = "avg_degree"   # Options: "avg_degree" or "threshold"
TARGET_AVERAGE_DEGREE = 5.0

# Used only when EDGE_SELECTION_MODE == "threshold".
SIMILARITY_THRESHOLD = 0.70

# False means rank/keep positive Pearson similarity.
# True means rank/keep abs(Pearson), including strong negative correlations.
USE_ABSOLUTE_CORRELATION = False

# Used only when EDGE_SELECTION_MODE == "avg_degree".
# This avoids selecting negative Pearson edges if the graph is very sparse.
# Set to -1.0 if you want the top-k edges even when some are negative.
MIN_SIMILARITY_FOR_AVG_DEGREE = 0.0

# If True, cached edge files are ignored and recomputed.
FORCE_RECOMPUTE_EDGES = True

GNN_MODELS = ["gcn", "graphsage", "gin"]

# Your preferred validation-tuning grid.
HYPERPARAM_GRID = [
    {"hidden_channels": 64,  "num_layers": 2, "dropout": 0.2, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 128, "num_layers": 2, "dropout": 0.2, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 128, "num_layers": 3, "dropout": 0.3, "lr": 0.001,  "weight_decay": 1e-4},
    {"hidden_channels": 64,  "num_layers": 2, "dropout": 0.3, "lr": 0.0005, "weight_decay": 1e-4},
]

MAX_EPOCHS = 500
PATIENCE = 60

# Hyperparameter selection rule.
# Options: "rmse", "mae", "r2", "pearson", "spearman", or "val_loss"
SELECTION_METRIC = "mae"

# For rmse/mae/val_loss use "min"; for r2/pearson/spearman use "max".
SELECTION_MODE = "min"

# Target scaling is fit on train nodes only and inverse-transformed for metrics.
SCALE_TARGET = True

# Optional: scale merged node features again using train nodes only.
# Your metabolome and microbiome are already train-fitted/scaled from preprocessing.
# Genotype is not scaled in your preprocessing. Keep False unless you want all merged X on the same scale.
SCALE_NODE_FEATURES_FOR_GNN = False


# ============================================================
# Safety check for PyG
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
# File loading and alignment
# ============================================================

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def assert_same_ids(y_df: pd.DataFrame, x_df: pd.DataFrame, split_name: str, omic_name: str):
    """Make sure X and y are same samples in same row order."""
    y_ids = y_df[ID_COLS].astype(str).reset_index(drop=True)
    x_ids = x_df[ID_COLS].astype(str).reset_index(drop=True)

    if not y_ids.equals(x_ids):
        raise ValueError(
            f"ID/order mismatch for split={split_name}, omic={omic_name}. "
            f"Check y_{split_name}.csv and {omic_name}_{split_name}.csv."
        )


def load_split_from_seed(seed_dir: Path, split_name: str):
    """Load one split and concatenate all omics into one X matrix."""
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

        # Prefix features to avoid duplicate column names after concatenation.
        x_part.columns = [
            f"{omic}__{c}" if not c.startswith(f"{omic}__") else c
            for c in x_part.columns
        ]

        x_parts.append(x_part)

    X = pd.concat(x_parts, axis=1)
    ids = y_df[ID_COLS].copy()
    y = y_df[TARGET_TRAITS].copy()

    ids.insert(1, "split", split_name)

    return ids, X, y


def load_seed_node_data(seed: int):
    """Load train, val, test nodes for one seed."""
    seed_dir = PREPROCESSED_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed folder: {seed_dir}")

    all_ids = []
    all_X = []
    all_y = []

    for split_name in ["train", "val", "test"]:
        ids, X, y = load_split_from_seed(seed_dir, split_name)
        all_ids.append(ids)
        all_X.append(X)
        all_y.append(y)

    node_table = pd.concat(all_ids, ignore_index=True)
    X_all = pd.concat(all_X, ignore_index=True)
    y_all = pd.concat(all_y, ignore_index=True)

    if X_all.isna().sum().sum() > 0:
        raise ValueError("Merged node features contain NaN values. Check preprocessed omic files.")

    if y_all[TARGET_TRAITS].isna().sum().sum() > 0:
        raise ValueError("Target y contains NaN values. Check preprocessing complete-case filtering.")

    return seed_dir, node_table, X_all, y_all


def save_merged_node_files(seed_dir: Path, node_table: pd.DataFrame, X_all: pd.DataFrame, y_all: pd.DataFrame):
    """Save merged node features and node table for reproducibility."""
    node_table.to_csv(seed_dir / "node_table.csv", index=False)

    merged = pd.concat(
        [node_table.reset_index(drop=True), X_all.reset_index(drop=True)],
        axis=1,
    )
    merged.to_csv(seed_dir / "merged_node_features.csv", index=False)

    y_merged = pd.concat(
        [node_table.reset_index(drop=True), y_all.reset_index(drop=True)],
        axis=1,
    )
    y_merged.to_csv(seed_dir / "merged_y_targets.csv", index=False)


# ============================================================
# Edge computation and caching
# ============================================================

def _safe_float_tag(value: float) -> str:
    """Create a filename-safe tag from a numeric value."""
    return str(value).replace("-", "m").replace(".", "p")


def edge_filename(method: str, threshold: float):
    """Return the cache filename for the selected edge construction mode."""
    abs_tag = "_abs" if USE_ABSOLUTE_CORRELATION else ""

    if EDGE_SELECTION_MODE == "avg_degree":
        k_tag = _safe_float_tag(TARGET_AVERAGE_DEGREE)
        min_tag = _safe_float_tag(MIN_SIMILARITY_FOR_AVG_DEGREE)
        return f"edges_{method}_avgdeg_{k_tag}_minsim_{min_tag}{abs_tag}.csv"

    if EDGE_SELECTION_MODE == "threshold":
        thr = _safe_float_tag(threshold)
        return f"edges_{method}_thr_{thr}{abs_tag}.csv"

    raise ValueError(
        f"Unknown EDGE_SELECTION_MODE: {EDGE_SELECTION_MODE}. "
        "Use 'avg_degree' or 'threshold'."
    )


def compute_rowwise_pearson(X: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation between rows of X."""
    X = np.asarray(X, dtype=np.float64)

    Xc = X - X.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = np.nan

    Z = Xc / norms
    sim = Z @ Z.T

    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(sim, 1.0)

    return sim


def compute_rowwise_spearman(X: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation between rows of X by rank-transforming each row."""
    X_df = pd.DataFrame(X)
    X_ranked = X_df.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    return compute_rowwise_pearson(X_ranked)


def _format_edge_dataframe(
    src: np.ndarray,
    dst: np.ndarray,
    kept_scores: np.ndarray,
    node_table: pd.DataFrame,
    method: str,
    selected_similarity_cutoff: float,
) -> pd.DataFrame:
    """Create the final edge DataFrame with IDs, splits, and audit columns."""
    num_nodes = len(node_table)
    num_edges = len(src)
    actual_avg_degree = (2.0 * num_edges / num_nodes) if num_nodes > 0 else 0.0

    edges = pd.DataFrame({
        "source": src.astype(int),
        "target": dst.astype(int),
        "similarity": kept_scores.astype(float),
        "method": method,
        "edge_selection_mode": EDGE_SELECTION_MODE,
        "target_average_degree": TARGET_AVERAGE_DEGREE if EDGE_SELECTION_MODE == "avg_degree" else np.nan,
        "actual_average_degree": actual_avg_degree,
        "selected_similarity_cutoff": selected_similarity_cutoff,
        # Keep the old column name for backward compatibility with older outputs.
        # For avg_degree mode, this stores the final selected similarity cutoff.
        "threshold": selected_similarity_cutoff,
        "source_sample_id": node_table.loc[src, "sample_id"].to_numpy(),
        "target_sample_id": node_table.loc[dst, "sample_id"].to_numpy(),
        "source_split": node_table.loc[src, "split"].to_numpy(),
        "target_split": node_table.loc[dst, "split"].to_numpy(),
    })

    return edges


def build_threshold_edges_from_similarity(
    sim: np.ndarray,
    node_table: pd.DataFrame,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    """Create undirected edges using a fixed similarity threshold."""
    n = sim.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    scores = sim[upper_i, upper_j]

    if USE_ABSOLUTE_CORRELATION:
        keep = np.abs(scores) >= threshold
    else:
        keep = scores >= threshold

    src = upper_i[keep]
    dst = upper_j[keep]
    kept_scores = scores[keep]

    return _format_edge_dataframe(
        src=src,
        dst=dst,
        kept_scores=kept_scores,
        node_table=node_table,
        method=method,
        selected_similarity_cutoff=float(threshold),
    )


def build_avg_degree_edges_from_similarity(
    sim: np.ndarray,
    node_table: pd.DataFrame,
    method: str,
    target_average_degree: float = TARGET_AVERAGE_DEGREE,
    min_similarity: float = MIN_SIMILARITY_FOR_AVG_DEGREE,
) -> pd.DataFrame:
    """
    Create undirected edges by keeping the strongest similarities needed to
    reach the requested average degree.

    For an undirected graph:
        average_degree = 2 * num_edges / num_nodes
    so the target edge count is:
        num_edges = num_nodes * target_average_degree / 2

    This gives a comparable graph density across seeds even when the raw
    Pearson similarity distribution shifts slightly from seed to seed.
    """
    n = sim.shape[0]

    if n < 2:
        return _format_edge_dataframe(
            src=np.array([], dtype=int),
            dst=np.array([], dtype=int),
            kept_scores=np.array([], dtype=float),
            node_table=node_table,
            method=method,
            selected_similarity_cutoff=np.nan,
        )

    upper_i, upper_j = np.triu_indices(n, k=1)
    scores = sim[upper_i, upper_j]

    # Ranking score controls which pairs are considered strongest.
    # By default, we rank by positive Pearson similarity only.
    ranking_scores = np.abs(scores) if USE_ABSOLUTE_CORRELATION else scores

    if USE_ABSOLUTE_CORRELATION:
        candidate_mask = np.abs(scores) >= min_similarity
    else:
        candidate_mask = scores >= min_similarity

    candidate_idx = np.where(candidate_mask)[0]

    if len(candidate_idx) == 0:
        return _format_edge_dataframe(
            src=np.array([], dtype=int),
            dst=np.array([], dtype=int),
            kept_scores=np.array([], dtype=float),
            node_table=node_table,
            method=method,
            selected_similarity_cutoff=np.nan,
        )

    target_num_edges = int(np.ceil(n * target_average_degree / 2.0))
    target_num_edges = max(1, min(target_num_edges, len(candidate_idx)))

    ranked_candidate_idx = candidate_idx[
        np.argsort(ranking_scores[candidate_idx])[::-1]
    ]
    selected_idx = ranked_candidate_idx[:target_num_edges]

    src = upper_i[selected_idx]
    dst = upper_j[selected_idx]
    kept_scores = scores[selected_idx]

    # The selected cutoff is the weakest selected edge.
    if USE_ABSOLUTE_CORRELATION:
        selected_similarity_cutoff = float(np.min(np.abs(kept_scores)))
    else:
        selected_similarity_cutoff = float(np.min(kept_scores))

    edges = _format_edge_dataframe(
        src=src,
        dst=dst,
        kept_scores=kept_scores,
        node_table=node_table,
        method=method,
        selected_similarity_cutoff=selected_similarity_cutoff,
    )

    # Sort for easier inspection and reproducibility.
    sort_col = "abs_similarity_for_sort" if USE_ABSOLUTE_CORRELATION else "similarity"
    if USE_ABSOLUTE_CORRELATION:
        edges["abs_similarity_for_sort"] = edges["similarity"].abs()

    edges = edges.sort_values(sort_col, ascending=False).reset_index(drop=True)

    if USE_ABSOLUTE_CORRELATION:
        edges = edges.drop(columns=["abs_similarity_for_sort"])

    return edges


def build_edges_from_similarity(
    sim: np.ndarray,
    node_table: pd.DataFrame,
    threshold: float,
    method: str,
) -> pd.DataFrame:
    """Create undirected edge list using the selected edge construction mode."""
    if EDGE_SELECTION_MODE == "avg_degree":
        return build_avg_degree_edges_from_similarity(
            sim=sim,
            node_table=node_table,
            method=method,
            target_average_degree=TARGET_AVERAGE_DEGREE,
            min_similarity=MIN_SIMILARITY_FOR_AVG_DEGREE,
        )

    if EDGE_SELECTION_MODE == "threshold":
        return build_threshold_edges_from_similarity(
            sim=sim,
            node_table=node_table,
            threshold=threshold,
            method=method,
        )

    raise ValueError(
        f"Unknown EDGE_SELECTION_MODE: {EDGE_SELECTION_MODE}. "
        "Use 'avg_degree' or 'threshold'."
    )


def load_or_compute_edges(
    seed_dir: Path,
    node_table: pd.DataFrame,
    X_all: pd.DataFrame,
    method: str,
    threshold: float,
):
    """Load cached edges if present, otherwise compute and save."""
    edge_path = seed_dir / edge_filename(method, threshold)

    if edge_path.exists() and not FORCE_RECOMPUTE_EDGES:
        print(f"Loading cached edges: {edge_path}")
        return pd.read_csv(edge_path)

    if EDGE_SELECTION_MODE == "avg_degree":
        print(
            f"Computing {method} edges for {seed_dir.name} "
            f"with target average degree k={TARGET_AVERAGE_DEGREE}..."
        )
    else:
        print(f"Computing {method} edges for {seed_dir.name} with threshold {threshold}...")

    X_np = X_all.to_numpy(dtype=np.float32)

    if method == "pearson":
        sim = compute_rowwise_pearson(X_np)
    elif method == "spearman":
        sim = compute_rowwise_spearman(X_np)
    else:
        raise ValueError(f"Unknown edge method: {method}")

    edges = build_edges_from_similarity(
        sim=sim,
        node_table=node_table,
        threshold=threshold,
        method=method,
    )

    edges.to_csv(edge_path, index=False)

    actual_avg_degree = (2.0 * len(edges) / len(node_table)) if len(node_table) > 0 else 0.0
    selected_cutoff = (
        float(edges["selected_similarity_cutoff"].iloc[0])
        if len(edges) > 0 and "selected_similarity_cutoff" in edges.columns
        else np.nan
    )

    print(
        f"Saved {len(edges)} undirected edges to {edge_path}. "
        f"Nodes={len(node_table)}, "
        f"possible undirected pairs={len(node_table) * (len(node_table) - 1) // 2}, "
        f"actual average degree={actual_avg_degree:.3f}, "
        f"selected similarity cutoff={selected_cutoff:.4f}"
    )

    return edges


def edges_to_edge_index(edges: pd.DataFrame, num_nodes: int, device):
    """Convert undirected edge CSV to PyG edge_index with both directions and self-loops."""
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        src = torch.tensor(edges["source"].to_numpy(), dtype=torch.long, device=device)
        dst = torch.tensor(edges["target"].to_numpy(), dtype=torch.long, device=device)

        edge_index = torch.stack(
            [
                torch.cat([src, dst]),
                torch.cat([dst, src]),
            ],
            dim=0,
        )

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


# ============================================================
# Target and feature scaling
# ============================================================

def get_masks(node_table: pd.DataFrame, device):
    split = node_table["split"].to_numpy()

    train_mask = torch.tensor(split == "train", dtype=torch.bool, device=device)
    val_mask = torch.tensor(split == "val", dtype=torch.bool, device=device)
    test_mask = torch.tensor(split == "test", dtype=torch.bool, device=device)

    return train_mask, val_mask, test_mask


def prepare_tensors(node_table: pd.DataFrame, X_all: pd.DataFrame, y_all: pd.DataFrame, device):
    """Prepare X, scaled y, masks, and y scaler."""
    train_mask_np = node_table["split"].to_numpy() == "train"

    X_np = X_all.to_numpy(dtype=np.float32)

    if SCALE_NODE_FEATURES_FOR_GNN:
        x_scaler = StandardScaler()
        X_np[train_mask_np] = x_scaler.fit_transform(X_np[train_mask_np])
        X_np[~train_mask_np] = x_scaler.transform(X_np[~train_mask_np])
    else:
        x_scaler = None

    y_np = y_all[TARGET_TRAITS].to_numpy(dtype=np.float32)

    if SCALE_TARGET:
        y_scaler = StandardScaler()
        y_np_scaled = y_np.copy()
        y_np_scaled[train_mask_np] = y_scaler.fit_transform(y_np[train_mask_np])
        y_np_scaled[~train_mask_np] = y_scaler.transform(y_np[~train_mask_np])
    else:
        y_scaler = None
        y_np_scaled = y_np

    x = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np_scaled, dtype=torch.float32, device=device)

    train_mask, val_mask, test_mask = get_masks(node_table, device)

    return x, y, train_mask, val_mask, test_mask, y_scaler, x_scaler


def inverse_transform_predictions(arr: np.ndarray, y_scaler):
    """Convert predictions back to original target scale."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if y_scaler is not None:
        return y_scaler.inverse_transform(arr)

    return arr


# ============================================================
# GNN models
# ============================================================

class GCNRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class GraphSAGERegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class GINRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
            )
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
            )

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def build_gnn_model(
    model_name: str,
    in_channels: int,
    out_channels: int,
    device,
    hidden_channels: int,
    num_layers: int,
    dropout: float,
):
    if model_name == "gcn":
        model = GCNRegressor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "graphsage":
        model = GraphSAGERegressor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "gin":
        model = GINRegressor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown GNN model: {model_name}")

    return model.to(device)


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
    edge_method: str,
    split_name: str,
    n_edges: int,
    selected_similarity_cutoff: float = np.nan,
):
    rows = []

    for trait in TARGET_TRAITS:
        y_true = y_true_df[trait].to_numpy()
        y_pred = y_pred_df[trait].to_numpy()

        rows.append({
            "seed": seed,
            "model": model_name,
            "edge_method": edge_method,
            # For avg_degree mode, this is the weakest selected edge.
            # For threshold mode, this is the configured fixed threshold.
            "similarity_threshold": selected_similarity_cutoff if EDGE_SELECTION_MODE == "avg_degree" else SIMILARITY_THRESHOLD,
            "edge_selection_mode": EDGE_SELECTION_MODE,
            "target_average_degree": TARGET_AVERAGE_DEGREE if EDGE_SELECTION_MODE == "avg_degree" else np.nan,
            "selected_similarity_cutoff": selected_similarity_cutoff,
            "n_edges_undirected": n_edges,
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
        "edge_method": edge_method,
        "similarity_threshold": selected_similarity_cutoff if EDGE_SELECTION_MODE == "avg_degree" else SIMILARITY_THRESHOLD,
        "edge_selection_mode": EDGE_SELECTION_MODE,
        "target_average_degree": TARGET_AVERAGE_DEGREE if EDGE_SELECTION_MODE == "avg_degree" else np.nan,
        "selected_similarity_cutoff": selected_similarity_cutoff,
        "n_edges_undirected": n_edges,
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
        raise ValueError(
            f"Unknown SELECTION_METRIC: {SELECTION_METRIC}. "
            "Use rmse, mae, r2, pearson, spearman, or val_loss."
        )

    return float(avg[SELECTION_METRIC])


def is_better_score(new_score, best_score):
    if best_score is None:
        return True

    if SELECTION_MODE == "min":
        return new_score < best_score

    if SELECTION_MODE == "max":
        return new_score > best_score

    raise ValueError("SELECTION_MODE must be 'min' or 'max'.")


def add_config_to_rows(rows, config_id, config, best_val_loss, selected):
    for row in rows:
        row["config_id"] = config_id
        row["selected_config"] = bool(selected)
        row["best_val_loss_scaled"] = float(best_val_loss)
        for k, v in config.items():
            row[k] = v
    return rows


# ============================================================
# Training and prediction
# ============================================================

def train_one_gnn(model, data: Data, train_mask, val_mask, lr, weight_decay):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        train_loss = F.mse_loss(out[train_mask], data.y[train_mask])

        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = F.mse_loss(out[val_mask], data.y[val_mask]).item()

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


def predict_split(model, data: Data, mask, y_scaler, target_index):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(data.x, data.edge_index).detach().cpu().numpy()

    pred_selected = pred_scaled[mask.detach().cpu().numpy()]
    pred_original = inverse_transform_predictions(pred_selected, y_scaler)

    pred_df = pd.DataFrame(pred_original, columns=TARGET_TRAITS, index=target_index)
    return pred_df


def save_prediction_file(
    seed: int,
    model_name: str,
    edge_method: str,
    split_name: str,
    ids: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
):
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = y_true[TARGET_TRAITS].copy()
    actual.columns = [f"actual_{c}" for c in actual.columns]

    pred = y_pred[TARGET_TRAITS].copy()
    pred.columns = [f"pred_{c}" for c in pred.columns]

    out = pd.concat(
        [
            ids.reset_index(drop=True),
            actual.reset_index(drop=True),
            pred.reset_index(drop=True),
        ],
        axis=1,
    )

    out_path = out_dir / f"{model_name}_{edge_method}_{split_name}_predictions.csv"
    out.to_csv(out_path, index=False)


def print_average_metrics(rows_val, rows_test):
    val_avg = get_average_row(rows_val)
    test_avg = get_average_row(rows_test)

    print(
        f"    SELECTED VAL  | R2={val_avg['r2']:.3f} RMSE={val_avg['rmse']:.3f} "
        f"MAE={val_avg['mae']:.3f} Pearson={val_avg['pearson']:.3f} Spearman={val_avg['spearman']:.3f}"
    )
    print(
        f"    SELECTED TEST | R2={test_avg['r2']:.3f} RMSE={test_avg['rmse']:.3f} "
        f"MAE={test_avg['mae']:.3f} Pearson={test_avg['pearson']:.3f} Spearman={test_avg['spearman']:.3f}"
    )


# ============================================================
# Hyperparameter tuning for one model and one graph
# ============================================================

def tune_model_on_validation(
    seed: int,
    model_name: str,
    edge_method: str,
    data: Data,
    train_mask,
    val_mask,
    y_scaler,
    y_val_true: pd.DataFrame,
    n_edges: int,
    selected_similarity_cutoff: float,
    device,
):
    tuning_rows = []

    best_score = None
    best_payload = None

    for config_id, config in enumerate(HYPERPARAM_GRID):
        print(f"    config {config_id}: {config}")

        torch.manual_seed(seed + config_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + config_id)

        model = build_gnn_model(
            model_name=model_name,
            in_channels=data.x.shape[1],
            out_channels=len(TARGET_TRAITS),
            device=device,
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

        model, best_val_loss = train_one_gnn(
            model=model,
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        val_pred_df = predict_split(
            model=model,
            data=data,
            mask=val_mask,
            y_scaler=y_scaler,
            target_index=y_val_true.index,
        )

        val_rows = evaluate_predictions(
            y_true_df=y_val_true,
            y_pred_df=val_pred_df,
            seed=seed,
            model_name=model_name,
            edge_method=edge_method,
            split_name="val",
            n_edges=n_edges,
            selected_similarity_cutoff=selected_similarity_cutoff,
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
        )
        for row in config_rows:
            row["selection_metric"] = SELECTION_METRIC
            row["selection_mode"] = SELECTION_MODE
            row["selection_score"] = score
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
        raise RuntimeError(f"No best model selected for {model_name} / {edge_method}")

    # Mark selected config rows in tuning table.
    for row in tuning_rows:
        if row["config_id"] == best_payload["config_id"]:
            row["selected_config"] = True

    return best_payload, tuning_rows


# ============================================================
# Seed-level run
# ============================================================

def run_seed(seed: int, device):
    print("\n==============================")
    print(f"Seed {seed}")
    print("==============================")

    seed_dir, node_table, X_all, y_all = load_seed_node_data(seed)
    save_merged_node_files(seed_dir, node_table, X_all, y_all)

    print(f"Nodes: {len(node_table)}")
    print(f"Features: {X_all.shape[1]}")
    print(f"Targets: {TARGET_TRAITS}")
    print(f"Selection rule: {SELECTION_METRIC} ({SELECTION_MODE})")
    print(f"Edge selection mode: {EDGE_SELECTION_MODE}")
    if EDGE_SELECTION_MODE == "avg_degree":
        print(f"Target plant-plant average degree: {TARGET_AVERAGE_DEGREE}")

    x, y_scaled, train_mask, val_mask, test_mask, y_scaler, x_scaler = prepare_tensors(
        node_table=node_table,
        X_all=X_all,
        y_all=y_all,
        device=device,
    )

    val_idx = node_table.index[node_table["split"] == "val"]
    test_idx = node_table.index[node_table["split"] == "test"]

    ids_val = node_table.loc[val_idx, ID_COLS + ["split"]].reset_index(drop=True)
    ids_test = node_table.loc[test_idx, ID_COLS + ["split"]].reset_index(drop=True)

    y_val_true = y_all.loc[val_idx, TARGET_TRAITS].reset_index(drop=True)
    y_test_true = y_all.loc[test_idx, TARGET_TRAITS].reset_index(drop=True)

    all_selected_metric_rows = []
    all_tuning_rows = []
    best_config_rows = []

    for edge_method in EDGE_METHODS:
        edges = load_or_compute_edges(
            seed_dir=seed_dir,
            node_table=node_table,
            X_all=X_all,
            method=edge_method,
            threshold=SIMILARITY_THRESHOLD,
        )

        selected_similarity_cutoff = (
            float(edges["selected_similarity_cutoff"].iloc[0])
            if len(edges) > 0 and "selected_similarity_cutoff" in edges.columns
            else np.nan
        )
        actual_average_degree = (2.0 * len(edges) / len(node_table)) if len(node_table) > 0 else 0.0

        edge_index = edges_to_edge_index(
            edges=edges,
            num_nodes=len(node_table),
            device=device,
        )

        graph_data = Data(x=x, edge_index=edge_index, y=y_scaled)

        print(f"\nEdge method: {edge_method}")
        print(f"Undirected cached edges without self-loops: {len(edges)}")
        print(f"Directed PyG edges with self-loops: {graph_data.edge_index.shape[1]}")

        for model_name in GNN_MODELS:
            print(f"\n  Tuning {model_name} on {edge_method} graph")

            best_payload, tuning_rows = tune_model_on_validation(
                seed=seed,
                model_name=model_name,
                edge_method=edge_method,
                data=graph_data,
                train_mask=train_mask,
                val_mask=val_mask,
                y_scaler=y_scaler,
                y_val_true=y_val_true,
                n_edges=len(edges),
                selected_similarity_cutoff=selected_similarity_cutoff,
                device=device,
            )

            all_tuning_rows.extend(tuning_rows)

            best_config = best_payload["config"]
            print(
                f"  Best config for {model_name}/{edge_method}: "
                f"config_id={best_payload['config_id']}, "
                f"{SELECTION_METRIC}={best_payload['score']:.4f}, "
                f"params={best_config}"
            )

            # Rebuild model with best config and load best weights.
            best_model = build_gnn_model(
                model_name=model_name,
                in_channels=graph_data.x.shape[1],
                out_channels=len(TARGET_TRAITS),
                device=device,
                hidden_channels=best_config["hidden_channels"],
                num_layers=best_config["num_layers"],
                dropout=best_config["dropout"],
            )
            best_model.load_state_dict(best_payload["model_state"])

            val_pred_df = best_payload["val_pred_df"]

            test_pred_df = predict_split(
                model=best_model,
                data=graph_data,
                mask=test_mask,
                y_scaler=y_scaler,
                target_index=y_test_true.index,
            )

            val_rows = add_config_to_rows(
                rows=copy.deepcopy(best_payload["val_rows"]),
                config_id=best_payload["config_id"],
                config=best_config,
                best_val_loss=best_payload["best_val_loss"],
                selected=True,
            )

            test_rows = evaluate_predictions(
                y_true_df=y_test_true,
                y_pred_df=test_pred_df,
                seed=seed,
                model_name=model_name,
                edge_method=edge_method,
                split_name="test",
                n_edges=len(edges),
                selected_similarity_cutoff=selected_similarity_cutoff,
            )
            test_rows = add_config_to_rows(
                rows=test_rows,
                config_id=best_payload["config_id"],
                config=best_config,
                best_val_loss=best_payload["best_val_loss"],
                selected=True,
            )

            for row in val_rows + test_rows:
                row["selection_metric"] = SELECTION_METRIC
                row["selection_mode"] = SELECTION_MODE
                row["selection_score"] = best_payload["score"]

            all_selected_metric_rows.extend(val_rows)
            all_selected_metric_rows.extend(test_rows)

            save_prediction_file(
                seed=seed,
                model_name=model_name,
                edge_method=edge_method,
                split_name="val",
                ids=ids_val,
                y_true=y_val_true,
                y_pred=val_pred_df,
            )

            save_prediction_file(
                seed=seed,
                model_name=model_name,
                edge_method=edge_method,
                split_name="test",
                ids=ids_test,
                y_true=y_test_true,
                y_pred=test_pred_df,
            )

            best_config_rows.append({
                "seed": seed,
                "model": model_name,
                "edge_method": edge_method,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "edge_selection_mode": EDGE_SELECTION_MODE,
                "target_average_degree": TARGET_AVERAGE_DEGREE if EDGE_SELECTION_MODE == "avg_degree" else np.nan,
                "selected_similarity_cutoff": selected_similarity_cutoff,
                "actual_average_degree": actual_average_degree,
                "n_edges_undirected": len(edges),
                "config_id": best_payload["config_id"],
                "selection_metric": SELECTION_METRIC,
                "selection_mode": SELECTION_MODE,
                "selection_score": best_payload["score"],
                "best_val_loss_scaled": best_payload["best_val_loss"],
                **best_config,
            })

            print_average_metrics(val_rows, test_rows)

    seed_out_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)

    selected_metrics = pd.DataFrame(all_selected_metric_rows)
    tuning_metrics = pd.DataFrame(all_tuning_rows)
    best_configs = pd.DataFrame(best_config_rows)

    selected_metrics.to_csv(seed_out_dir / "metrics.csv", index=False)
    tuning_metrics.to_csv(seed_out_dir / "tuning_validation_metrics_all_configs.csv", index=False)
    best_configs.to_csv(seed_out_dir / "best_hyperparameters.csv", index=False)

    return selected_metrics, tuning_metrics, best_configs


# ============================================================
# Summary
# ============================================================

def summarize_results(
    all_metrics: pd.DataFrame,
    all_tuning_metrics: pd.DataFrame,
    all_best_configs: pd.DataFrame,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(OUTPUT_DIR / "all_seed_metrics_long.csv", index=False)
    all_tuning_metrics.to_csv(OUTPUT_DIR / "all_tuning_validation_metrics.csv", index=False)
    all_best_configs.to_csv(OUTPUT_DIR / "best_hyperparameters.csv", index=False)

    summary_by_trait = (
        all_metrics
        .groupby(["model", "edge_method", "split", "trait"], as_index=False)
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
            n_edges_mean=("n_edges_undirected", "mean"),
        )
    )

    summary_by_trait.to_csv(OUTPUT_DIR / "summary_by_trait.csv", index=False)

    ranking = (
        all_metrics[all_metrics["trait"] == "AVERAGE"]
        .groupby(["model", "edge_method", "split"], as_index=False)
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
            n_edges_mean=("n_edges_undirected", "mean"),
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

    print(f"\nSaved tuned GNN results to: {OUTPUT_DIR}/")

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

    if EDGE_SELECTION_MODE not in {"avg_degree", "threshold"}:
        raise ValueError("EDGE_SELECTION_MODE must be either 'avg_degree' or 'threshold'.")

    if EDGE_SELECTION_MODE == "avg_degree" and TARGET_AVERAGE_DEGREE <= 0:
        raise ValueError("TARGET_AVERAGE_DEGREE must be positive when using avg_degree mode.")

    if SELECTION_METRIC in {"rmse", "mae", "val_loss"} and SELECTION_MODE != "min":
        raise ValueError(f"For {SELECTION_METRIC}, use SELECTION_MODE='min'.")

    if SELECTION_METRIC in {"r2", "pearson", "spearman"} and SELECTION_MODE != "max":
        raise ValueError(f"For {SELECTION_METRIC}, use SELECTION_MODE='max'.")

    device = get_device(preferred_gpu=1)
    print(f"Using device: {device}")

    all_seed_metrics = []
    all_tuning_metrics = []
    all_best_configs = []

    for seed in SEEDS:
        selected_metrics, tuning_metrics, best_configs = run_seed(seed, device)
        all_seed_metrics.append(selected_metrics)
        all_tuning_metrics.append(tuning_metrics)
        all_best_configs.append(best_configs)

    all_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    all_tuning = pd.concat(all_tuning_metrics, ignore_index=True)
    all_best = pd.concat(all_best_configs, ignore_index=True)

    summarize_results(all_metrics, all_tuning, all_best)


if __name__ == "__main__":
    main()
