from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

PREPROCESSED_DIR = Path("preprocessed")
OUTPUT_DIR = Path("ml_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

ID_COLS = ["sample_id", "variety", "varietyID", "block", "plot"]

# Use ONE trait:
# TARGET_TRAITS = ["PlantHeight_cm"]

# Or use MULTIPLE traits:
TARGET_TRAITS = [
    "ShootDryWeight_g",
    # "ShootFreshWeight_g",
    # "GrowthStage",
    # "NumberOfNodes",
    # "NumberOfTillers",
    # "PlantHeight_cm",
    # "TillerLength_cm",
]

OMICS_TO_USE = ["genotype", "metabolome", "microbiome"]


MODELS_NEED_X_SCALING = {}


USE_TARGET_SCALING = True


# ============================================================
# Optional XGBoost import
# ============================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("XGBoost is not installed. XGBoost model will be skipped.")
    print("Install with: pip install xgboost")


# ============================================================
# Data loading
# ============================================================

def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def assert_same_ids(y_df: pd.DataFrame, x_df: pd.DataFrame, split_name: str, omic_name: str):
    """Make sure X and y are same samples in same order."""
    y_ids = y_df[ID_COLS].astype(str).reset_index(drop=True)
    x_ids = x_df[ID_COLS].astype(str).reset_index(drop=True)

    if not y_ids.equals(x_ids):
        raise ValueError(
            f"ID/order mismatch for split={split_name}, omic={omic_name}. "
            f"Check y_{split_name}.csv and {omic_name}_{split_name}.csv."
        )


def load_seed_split(seed: int, split_name: str, omics_to_use=OMICS_TO_USE):
    """Load X, y, and IDs for one seed and one split."""
    seed_dir = PREPROCESSED_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed folder: {seed_dir}")

    y_path = seed_dir / f"y_{split_name}.csv"
    y_df = read_csv_checked(y_path)

    missing_traits = [t for t in TARGET_TRAITS if t not in y_df.columns]
    if missing_traits:
        raise ValueError(f"Missing target traits in {y_path}: {missing_traits}")

    x_parts = []

    for omic in omics_to_use:
        x_path = seed_dir / f"{omic}_{split_name}.csv"
        x_df = read_csv_checked(x_path)

        assert_same_ids(y_df, x_df, split_name, omic)

        feature_cols = [c for c in x_df.columns if c not in ID_COLS]
        x_part = x_df[feature_cols].copy()

        # Prefix feature names to prevent duplicate column names after concatenation.
        x_part.columns = [
            f"{omic}__{c}" if not c.startswith(f"{omic}__") else c
            for c in x_part.columns
        ]

        x_parts.append(x_part)

    X = pd.concat(x_parts, axis=1)
    y = y_df[TARGET_TRAITS].copy()
    ids = y_df[ID_COLS].copy()

    return X, y, ids


def load_seed_data(seed: int):
    X_train, y_train, ids_train = load_seed_split(seed, "train")
    X_val, y_val, ids_val = load_seed_split(seed, "val")
    X_test, y_test, ids_test = load_seed_split(seed, "test")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "ids_train": ids_train,
        "X_val": X_val,
        "y_val": y_val,
        "ids_val": ids_val,
        "X_test": X_test,
        "y_test": y_test,
        "ids_test": ids_test,
    }


# ============================================================
# Target shape helpers
# ============================================================

def is_multi_target():
    return len(TARGET_TRAITS) > 1


def get_y_for_fit(y_df: pd.DataFrame):
    """
    For one target, return a Series.
    For multiple targets, return a DataFrame.
    """
    if is_multi_target():
        return y_df[TARGET_TRAITS]
    return y_df[TARGET_TRAITS[0]]


def prediction_to_df(pred, index=None):
    """
    Convert model prediction into a DataFrame with TARGET_TRAITS columns.
    Works for both single-target and multi-target prediction.
    """
    pred = np.asarray(pred)

    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    if pred.shape[1] != len(TARGET_TRAITS):
        raise ValueError(
            f"Prediction shape mismatch. Expected {len(TARGET_TRAITS)} target columns, "
            f"got {pred.shape[1]}."
        )

    return pd.DataFrame(pred, columns=TARGET_TRAITS, index=index)


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


def evaluate_prediction_df(
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

    avg_row = {
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
    }

    rows.append(avg_row)
    return rows


# ============================================================
# Model definitions
# ============================================================

def maybe_wrap_multioutput(model):
    """
    Some models do not support multi-output regression directly.
    If predicting multiple traits, wrap them with MultiOutputRegressor.
    If predicting one trait, return the model as-is.
    """
    if is_multi_target():
        return MultiOutputRegressor(model)
    return model


def build_models(seed: int):
    models = {}

    # RandomForestRegressor supports both single-output and multi-output directly.
    models["random_forest"] = RandomForestRegressor(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    # SVR supports only single-output, so wrap only when needed.
    models["svm_rbf"] = maybe_wrap_multioutput(
        SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.1,
            gamma="scale",
        )
    )

    # MLPRegressor supports both single-output and multi-output directly.
    models["mlp"] = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )

    # XGBoost is safer with MultiOutputRegressor for multiple traits.
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

        models["xgboost"] = maybe_wrap_multioutput(xgb)

    return models


def make_training_estimator(base_model, model_name: str):
    regressor = clone(base_model)

    # Scale X only for models that need it.
    if model_name in MODELS_NEED_X_SCALING:
        regressor = Pipeline([
            ("x_scaler", StandardScaler()),
            ("regressor", regressor),
        ])

    # Scale target y during training.
    if USE_TARGET_SCALING:
        regressor = TransformedTargetRegressor(
            regressor=regressor,
            transformer=StandardScaler(),
        )

    return regressor


# ============================================================
# Save predictions
# ============================================================

def save_predictions(
    seed: int,
    model_name: str,
    split_name: str,
    ids: pd.DataFrame,
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
):
    out_dir = OUTPUT_DIR / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    actual = y_true_df[TARGET_TRAITS].copy()
    actual.columns = [f"actual_{c}" for c in actual.columns]

    pred = y_pred_df[TARGET_TRAITS].copy()
    pred.columns = [f"pred_{c}" for c in pred.columns]

    out = pd.concat(
        [
            ids.reset_index(drop=True),
            actual.reset_index(drop=True),
            pred.reset_index(drop=True),
        ],
        axis=1,
    )

    out_path = out_dir / f"{model_name}_{split_name}_predictions.csv"
    out.to_csv(out_path, index=False)


# ============================================================
# Training loop
# ============================================================

def print_average_metrics(val_rows, test_rows):
    val_avg = [r for r in val_rows if r["trait"] == "AVERAGE"][0]
    test_avg = [r for r in test_rows if r["trait"] == "AVERAGE"][0]

    print(
        f"VAL  | R2={val_avg['r2']:.3f} | RMSE={val_avg['rmse']:.3f} | "
        f"MAE={val_avg['mae']:.3f} | Pearson={val_avg['pearson']:.3f} | "
        f"Spearman={val_avg['spearman']:.3f}"
    )

    print(
        f"TEST | R2={test_avg['r2']:.3f} | RMSE={test_avg['rmse']:.3f} | "
        f"MAE={test_avg['mae']:.3f} | Pearson={test_avg['pearson']:.3f} | "
        f"Spearman={test_avg['spearman']:.3f}"
    )


def train_and_evaluate_seed(seed: int):
    print("\n==============================")
    print(f"Seed {seed}")
    print("==============================")

    data = load_seed_data(seed)

    X_train = data["X_train"]
    y_train_df = data["y_train"]

    X_val = data["X_val"]
    y_val_df = data["y_val"]

    X_test = data["X_test"]
    y_test_df = data["y_test"]

    y_train_fit = get_y_for_fit(y_train_df)

    print(f"Target traits: {TARGET_TRAITS}")
    print(f"Prediction type: {'multi-target' if is_multi_target() else 'single-target'}")
    print(f"X_train: {X_train.shape}, y_train: {y_train_df[TARGET_TRAITS].shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val_df[TARGET_TRAITS].shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test_df[TARGET_TRAITS].shape}")

    models = build_models(seed)

    all_rows = []

    for model_name, base_model in models.items():
        print(f"\n--- Training {model_name} ---")

        model = make_training_estimator(base_model, model_name)
        model.fit(X_train, y_train_fit)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_pred_df = prediction_to_df(val_pred, index=y_val_df.index)
        test_pred_df = prediction_to_df(test_pred, index=y_test_df.index)

        val_rows = evaluate_prediction_df(
            y_true_df=y_val_df,
            y_pred_df=val_pred_df,
            seed=seed,
            model_name=model_name,
            split_name="val",
        )

        test_rows = evaluate_prediction_df(
            y_true_df=y_test_df,
            y_pred_df=test_pred_df,
            seed=seed,
            model_name=model_name,
            split_name="test",
        )

        all_rows.extend(val_rows)
        all_rows.extend(test_rows)

        save_predictions(
            seed=seed,
            model_name=model_name,
            split_name="val",
            ids=data["ids_val"],
            y_true_df=y_val_df,
            y_pred_df=val_pred_df,
        )

        save_predictions(
            seed=seed,
            model_name=model_name,
            split_name="test",
            ids=data["ids_test"],
            y_true_df=y_test_df,
            y_pred_df=test_pred_df,
        )

        print_average_metrics(val_rows, test_rows)

    seed_metrics = pd.DataFrame(all_rows)

    seed_out_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(seed_out_dir / "metrics.csv", index=False)

    return seed_metrics


# ============================================================
# Summaries
# ============================================================

def summarize_results(all_metrics: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics.to_csv(OUTPUT_DIR / "all_seed_metrics_long.csv", index=False)

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

    model_ranking = (
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

    model_ranking.to_csv(OUTPUT_DIR / "model_ranking_average_traits.csv", index=False)

    print("\n==============================")
    print("Model ranking")
    print("==============================")
    print(model_ranking.round(4).to_string(index=False))

    print(f"\nSaved results to: {OUTPUT_DIR}/")

    return summary_by_trait, model_ranking


def main():
    if len(TARGET_TRAITS) == 0:
        raise ValueError("TARGET_TRAITS cannot be empty.")

    all_seed_metrics = []

    for seed in SEEDS:
        seed_metrics = train_and_evaluate_seed(seed)
        all_seed_metrics.append(seed_metrics)

    all_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    summarize_results(all_metrics)


if __name__ == "__main__":
    main()