"""
Preprocess multi-omics data and save train/validation/test files by seed.
=======================================================================

This script DOES NOT train any model.

Main design:
1. Aggregate phenotype from plant-level to plot-level using median.
2. Keep only selected phenotype traits in y_*.csv.
3. Remove plots that are missing any selected y trait before splitting.
4. Keep phenotype/target columns OUT of genotype/metabolome/microbiome files.
5. Split by variety to avoid variety leakage.
6. Fit all learned preprocessing steps on TRAIN only for each seed.
7. Transform validation/test using the train-fitted preprocessing.

Output structure:
preprocessed/
  seed_42/
    metadata_train.csv
    metadata_val.csv
    metadata_test.csv

    y_train.csv
    y_val.csv
    y_test.csv

    genotype_train.csv
    genotype_val.csv
    genotype_test.csv

    metabolome_train.csv
    metabolome_val.csv
    metabolome_test.csv

    microbiome_train.csv
    microbiome_val.csv
    microbiome_test.csv

    split_assignments.csv
    preprocessing_summary.csv
    preprocessing_config.json

Important:
- y_*.csv contains only the selected phenotype traits listed in Y_TRAITS,
  plus their selected `<trait>__n` count columns.
- omic files contain only ID columns + preprocessed omic features.
- No phenotype trait or `<trait>__n` column appears in genotype/metabolome/microbiome files.
- Plots missing any selected y trait are removed before train/validation/test splitting.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PATHS = {
    "phenotype":  "raw_dataset/phenotype.csv",
    "metabolome": "raw_dataset/metabolome.csv",
    "microbiome": "raw_dataset/microbiome.csv",
    "genotype":   "raw_dataset/genoMarker_LD0.001_SNP3078.csv",
}

GROUP_COL = "variety"

META_COLS = ["year", "place", "sowing", "varietyID", "variety", "block", "plot", "ind"]
# All phenotype traits available in the raw phenotype file.
# These are used only for plant-to-plot aggregation and leakage checks.
ALL_PHENOTYPE_TRAITS = [
    "LeafDryWeight_g", "ShootDryWeight_g", "ShootFreshWeight_g", "LeafFreshWeight_g",
    "GrowthStage", "NumberOfNodes", "NumberOfTillers", "PlantHeight_cm", "TillerLength_cm",
]

# Traits to save in y_*.csv and require as complete/non-missing.
# Any plot missing at least one of these traits will be removed before splitting.
Y_TRAITS = [
    "ShootDryWeight_g", "ShootFreshWeight_g",
    "GrowthStage", "NumberOfNodes", "NumberOfTillers",
    "PlantHeight_cm", "TillerLength_cm",
]

# Backward-compatible alias used by older helper functions.
TARGETS = ALL_PHENOTYPE_TRAITS

# Recommended split for a small dataset.
# 75/15/15 is invalid because it sums to 105.
SPLIT_FRACS = (0.70, 0.15, 0.15)

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
GENO_JOIN = "inner"
OUTPUT_DIR = "preprocessed"

# Preprocessing choices
GENO_MIN_MAF = 0.05
METABOLOME_VARIANCE_THRESHOLD = 0.0
MICROBIOME_MIN_PREVALENCE = 0.05
MICROBIOME_CLR_PSEUDOCOUNT = 1.0

# These columns are safe to repeat in every file because they are identifiers only.
# They are not phenotype values and should not be used as model features unless you
# explicitly decide to use metadata later.
ID_COLS = ["sample_id", "variety", "varietyID", "block", "plot"]


# --------------------------------------------------------------------------- #
# Load + merge
# --------------------------------------------------------------------------- #
def aggregate_phenotype_to_plot(ph, targets=TARGETS):
    """Collapse plant-level phenotype to one row per plot using median.

    Also creates `<trait>__n` columns showing how many plant-level values
    contributed to each plot-level median.
    """
    keys = ["variety", "varietyID", "block", "plot"]

    med = ph.groupby(keys, as_index=False)[targets].median()
    cnt = (
        ph.groupby(keys)[targets]
        .count()
        .rename(columns={t: f"{t}__n" for t in targets})
        .reset_index()
    )

    return med.merge(cnt, on=keys, how="left")


def summarize_plot_level_phenotype_missingness(ph_plot, targets=TARGETS):
    """Return missingness summary after plant-level rows are aggregated to plot level."""
    rows = []
    total_plots = len(ph_plot)

    for trait in targets:
        if trait not in ph_plot.columns:
            continue

        n_col = f"{trait}__n"
        missing = int(ph_plot[trait].isna().sum())
        non_missing = int(ph_plot[trait].notna().sum())

        if n_col in ph_plot.columns:
            zero_contributor_plots = int((ph_plot[n_col] == 0).sum())
            one_contributor_plots = int((ph_plot[n_col] == 1).sum())
        else:
            zero_contributor_plots = missing
            one_contributor_plots = None

        rows.append({
            "trait": trait,
            "total_plots": int(total_plots),
            "missing_after_median": missing,
            "non_missing_after_median": non_missing,
            "missing_percent": round((missing / total_plots) * 100, 2) if total_plots else 0.0,
            "zero_contributor_plots": zero_contributor_plots,
            "one_contributor_plots": one_contributor_plots,
        })

    return pd.DataFrame(rows)


def print_plot_level_phenotype_missingness(ph_plot, targets=TARGETS):
    """Print missing values remaining after plant-to-plot median aggregation."""
    summary = summarize_plot_level_phenotype_missingness(ph_plot, targets=targets)

    print("\n=== phenotype missingness after plant-to-plot median aggregation ===")
    if summary.empty:
        print("No phenotype trait columns found.")
    else:
        print(summary.to_string(index=False))

    return summary


def filter_complete_y_traits(df, y_traits=Y_TRAITS, verbose=True):
    """Remove plots missing any selected y trait.

    This is done after all omics are merged, so every saved omic and y file
    contains the exact same retained plot/sample set for that seed.
    """
    missing_traits = [t for t in y_traits if t not in df.columns]
    if missing_traits:
        raise ValueError(f"Y_TRAITS not found in merged dataframe: {missing_traits}")

    before = len(df)
    missing_by_trait = df[y_traits].isna().sum().astype(int)
    keep_mask = df[y_traits].notna().all(axis=1)
    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())

    if verbose:
        print("\n=== filtering plots with missing selected y traits ===")
        print("Selected y traits:")
        for trait in y_traits:
            print(f"  - {trait}")
        print("\nMissing values before filtering, after omics merge:")
        print(missing_by_trait.to_string())
        print(f"\nRows before y-trait filtering: {before}")
        print(f"Rows removed because at least one selected y trait is missing: {removed}")
        print(f"Rows kept: {kept}")

    return df.loc[keep_mask].reset_index(drop=True)


def load_and_merge(paths=PATHS, geno_join=GENO_JOIN, verbose=True):
    """Return one plot-level table with prefixed omic feature columns."""
    # Phenotype: plant-level -> plot-level.
    ph = pd.read_csv(paths["phenotype"], index_col=0)
    ph_plot = aggregate_phenotype_to_plot(ph)

    if verbose:
        print_plot_level_phenotype_missingness(ph_plot)

    # Metabolome: plot-level. Remove metadata before prefixing feature columns.
    mtb = pd.read_csv(paths["metabolome"])
    mtb_feat = [c for c in mtb.columns if c not in (["ID"] + META_COLS)]
    mtb = mtb[["block", "plot"] + mtb_feat].rename(
        columns={c: f"mtb_{c}" for c in mtb_feat}
    )

    # Microbiome: plot-level, index like W1.001 -> block, plot.
    mb = pd.read_csv(paths["microbiome"], index_col=0)
    mb.columns = [f"mb_{i}" for i in range(mb.shape[1])]
    keys_mb = pd.DataFrame({
        "block": mb.index.astype(str).str.split(".").str[0].values,
        "plot": mb.index.astype(str).str.split(".").str[1].astype(int).values,
    })
    mb = pd.concat([keys_mb, mb.reset_index(drop=True)], axis=1)

    # Merge plot-level omics.
    n0 = len(ph_plot)
    df = ph_plot.merge(mtb, on=["block", "plot"], how="inner")
    n1 = len(df)
    df = df.merge(mb, on=["block", "plot"], how="inner")
    n2 = len(df)

    # Genotype: variety-level -> attach to plots by variety.
    g = pd.read_csv(paths["genotype"], index_col=0)
    g.columns = [f"geno_{c}" for c in g.columns]
    g = g.reset_index().rename(columns={"index": "variety"})
    df = df.merge(g, on="variety", how="inner" if geno_join == "inner" else "left")
    n3 = len(df)

    # Stable plot sample identifier.
    df["sample_id"] = (
        df["block"].astype(str).str.replace(" ", "", regex=False)
        + "_"
        + df["plot"].astype(str)
    )

    # Remove plots missing any selected y trait before creating seed splits.
    # This keeps y_train/y_val/y_test complete for the selected phenotype set.
    df = filter_complete_y_traits(df, y_traits=Y_TRAITS, verbose=verbose)

    # Only prefixed omic columns are features.
    # This prevents phenotype columns such as PlantHeight_cm from entering omic files.
    feature_cols = [c for c in df.columns if c.startswith(("geno_", "mtb_", "mb_"))]

    if verbose:
        print(
            f"plots: phenotype {n0} -> +metabolome {n1} -> +microbiome {n2} "
            f"-> +genotype({geno_join}) {n3}"
        )
        print(
            f"final df: {df.shape} | n_features: {len(feature_cols)} "
            f"(geno {sum(c.startswith('geno_') for c in feature_cols)}, "
            f"mtb {sum(c.startswith('mtb_') for c in feature_cols)}, "
            f"mb {sum(c.startswith('mb_') for c in feature_cols)})"
        )

    return df, feature_cols


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform for compositional microbiome data."""
    def __init__(self, pseudocount=1.0):
        self.pseudocount = pseudocount

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float) + self.pseudocount
        log_x = np.log(X)
        return log_x - log_x.mean(axis=1, keepdims=True)


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Keep taxa present in at least min_prevalence of TRAIN samples."""
    def __init__(self, min_prevalence=0.05, min_value=0.0):
        self.min_prevalence = min_prevalence
        self.min_value = min_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        prevalence = (X > self.min_value).mean(axis=0)
        self.mask_ = prevalence >= self.min_prevalence
        if not self.mask_.any():
            self.mask_ = prevalence >= 0
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


class MAFFilter(BaseEstimator, TransformerMixin):
    """Drop SNPs whose minor allele frequency is below min_maf on TRAIN only."""
    def __init__(self, min_maf=0.05):
        self.min_maf = min_maf

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        dosage = X + 1.0  # {-1, 0, 1} -> {0, 1, 2}
        freq = dosage.mean(axis=0) / 2.0
        maf = np.minimum(freq, 1.0 - freq)
        self.mask_ = maf >= self.min_maf
        if not self.mask_.any():
            self.mask_ = maf >= 0
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.mask_]


# --------------------------------------------------------------------------- #
# Separate omic pipelines
# --------------------------------------------------------------------------- #
def build_genotype_pipeline():
    return Pipeline([
        ("maf", MAFFilter(min_maf=GENO_MIN_MAF)),
        ("impute", SimpleImputer(strategy="most_frequent")),
        # ("scale", StandardScaler()),
    ])


def build_metabolome_pipeline():
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=METABOLOME_VARIANCE_THRESHOLD)),
        ("scale", StandardScaler()),
    ])


def build_microbiome_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("prevalence", PrevalenceFilter(min_prevalence=MICROBIOME_MIN_PREVALENCE)),
        ("clr", CLRTransformer(pseudocount=MICROBIOME_CLR_PSEUDOCOUNT)),
        ("scale", StandardScaler()),
    ])


# --------------------------------------------------------------------------- #
# Split first, then fit preprocessing on train only
# --------------------------------------------------------------------------- #
def make_grouped_split_indices(
    df,
    group_col=GROUP_COL,
    fracs=SPLIT_FRACS,
    random_state=42,
    verbose=True,
):
    """Return row indices for grouped train/val/test split.

    This preprocessing split does NOT drop rows with missing phenotype values.
    Target-specific missing values should be handled later in the training script.
    """
    assert abs(sum(fracs) - 1.0) < 1e-9, "SPLIT_FRACS must sum to 1.0"

    f_train, f_val, f_test = fracs

    use = df.reset_index(drop=True)
    groups = use[group_col].to_numpy()

    gss_test = GroupShuffleSplit(n_splits=1, test_size=f_test, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(use, groups=groups))

    rel_val = f_val / (f_train + f_val)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=random_state)
    tr_rel, val_rel = next(
        gss_val.split(use.iloc[trainval_idx], groups=groups[trainval_idx])
    )

    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[val_rel]

    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])

    assert g_train.isdisjoint(g_val), "variety leakage between train and validation"
    assert g_train.isdisjoint(g_test), "variety leakage between train and test"
    assert g_val.isdisjoint(g_test), "variety leakage between validation and test"

    if verbose:
        print(
            f"seed={random_state} | total={len(use)} | "
            f"train={len(train_idx)} ({len(g_train)} varieties) | "
            f"val={len(val_idx)} ({len(g_val)} varieties) | "
            f"test={len(test_idx)} ({len(g_test)} varieties)"
        )

    return use, {"train": train_idx, "val": val_idx, "test": test_idx}


def get_output_columns(original_cols, pipeline, omic_name):
    """Recover feature names after train-fitted feature filters."""
    original_cols = np.array(original_cols)

    if omic_name == "genotype":
        mask = pipeline.named_steps["maf"].mask_
        return original_cols[mask].tolist()

    if omic_name == "metabolome":
        mask = pipeline.named_steps["var"].get_support()
        return original_cols[mask].tolist()

    if omic_name == "microbiome":
        mask = pipeline.named_steps["prevalence"].mask_
        return original_cols[mask].tolist()

    raise ValueError(f"Unknown omic_name: {omic_name}")


def transform_and_save_omic(
    use,
    split_indices,
    omic_name,
    feature_cols,
    pipeline,
    out_dir,
    id_cols,
):
    """Fit one omic pipeline on train and save train/val/test CSV files.

    Phenotype columns are intentionally excluded from these files.
    """
    train_idx = split_indices["train"]

    X_train = use.iloc[train_idx][feature_cols]
    pipeline.fit(X_train)

    output_cols = get_output_columns(feature_cols, pipeline, omic_name)

    saved_shapes = {}
    for split_name, idx in split_indices.items():
        X_raw = use.iloc[idx][feature_cols]
        X_pre = pipeline.transform(X_raw)

        pre_df = pd.DataFrame(X_pre, columns=output_cols, index=use.iloc[idx].index)

        # Only identifiers are saved with omic features. No target/phenotype leakage.
        save_df = pd.concat(
            [use.iloc[idx][id_cols].reset_index(drop=True), pre_df.reset_index(drop=True)],
            axis=1,
        )

        # Defensive check: no phenotype columns in omic files.
        forbidden = set(ALL_PHENOTYPE_TRAITS + [f"{t}__n" for t in ALL_PHENOTYPE_TRAITS])
        leaked = forbidden.intersection(save_df.columns)
        if leaked:
            raise ValueError(f"Phenotype columns leaked into {omic_name}_{split_name}: {sorted(leaked)}")

        save_path = out_dir / f"{omic_name}_{split_name}.csv"
        save_df.to_csv(save_path, index=False)
        saved_shapes[split_name] = {
            "rows": int(save_df.shape[0]),
            "features": int(len(output_cols)),
            "file": str(save_path),
        }

    return pipeline, saved_shapes


def save_metadata_and_targets(use, split_indices, out_dir):
    """Save ID-only metadata and all phenotype traits/counts in y files."""
    id_cols = [c for c in ID_COLS if c in use.columns]
    trait_cols = [c for c in Y_TRAITS if c in use.columns]
    trait_count_cols = [f"{t}__n" for t in Y_TRAITS if f"{t}__n" in use.columns]
    y_cols = id_cols + trait_cols + trait_count_cols

    assignments = []
    missing_rows = []

    for split_name, idx in split_indices.items():
        # Metadata files: identifiers only, no phenotype values.
        meta = use.iloc[idx][id_cols].copy()
        meta.insert(1, "split", split_name)
        meta.to_csv(out_dir / f"metadata_{split_name}.csv", index=False)

        # y files: all traits + all trait count columns.
        y_df = use.iloc[idx][y_cols].copy()
        y_df.to_csv(out_dir / f"y_{split_name}.csv", index=False)

        # Split assignments for reproducibility.
        tmp = use.iloc[idx][id_cols].copy()
        tmp.insert(1, "split", split_name)
        assignments.append(tmp)

        for trait in trait_cols:
            missing_rows.append({
                "split": split_name,
                "trait": trait,
                "rows": int(len(y_df)),
                "missing": int(y_df[trait].isna().sum()),
                "non_missing": int(y_df[trait].notna().sum()),
            })

    pd.concat(assignments, ignore_index=True).to_csv(out_dir / "split_assignments.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(out_dir / "phenotype_missingness_summary.csv", index=False)

    return id_cols, y_cols


def assert_same_samples_across_saved_files(out_dir, splits=("train", "val", "test"), id_cols=ID_COLS):
    """Verify that y and every omic file contain exactly the same samples per split.

    This prevents silent misalignment between X files and y files.
    The check compares the ID columns in this exact order:
    sample_id, variety, varietyID, block, plot.
    """
    id_cols = list(id_cols)
    omics = ["genotype", "metabolome", "microbiome"]

    for split_name in splits:
        y_path = out_dir / f"y_{split_name}.csv"
        y_ids = pd.read_csv(y_path, usecols=id_cols)

        for omic in omics:
            x_path = out_dir / f"{omic}_{split_name}.csv"
            x_ids = pd.read_csv(x_path, usecols=id_cols)

            if len(y_ids) != len(x_ids):
                raise ValueError(
                    f"Sample count mismatch for {split_name}: "
                    f"y has {len(y_ids)} rows, {omic} has {len(x_ids)} rows"
                )

            same_order = y_ids.reset_index(drop=True).equals(x_ids.reset_index(drop=True))
            if not same_order:
                # More detailed diagnostics in case the same samples exist but are shuffled,
                # or if samples are truly missing from one file.
                y_keys = set(map(tuple, y_ids[id_cols].astype(str).to_numpy()))
                x_keys = set(map(tuple, x_ids[id_cols].astype(str).to_numpy()))
                missing_in_x = y_keys - x_keys
                extra_in_x = x_keys - y_keys

                if missing_in_x or extra_in_x:
                    raise ValueError(
                        f"Sample ID mismatch for {omic}_{split_name}.csv compared with y_{split_name}.csv. "
                        f"Missing in X: {len(missing_in_x)}, extra in X: {len(extra_in_x)}"
                    )

                raise ValueError(
                    f"Same samples but different row order for {omic}_{split_name}.csv "
                    f"compared with y_{split_name}.csv. Keep row order identical for downstream modeling."
                )

    print("Verified: y, genotype, metabolome, and microbiome files contain the same samples in the same order for every split.")


def save_preprocessed_for_seed(
    df,
    feature_cols,
    seed,
    output_root=OUTPUT_DIR,
    fracs=SPLIT_FRACS,
    group_col=GROUP_COL,
):
    """Create one seed split and save all preprocessed omic files."""
    out_dir = Path(output_root) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use, split_indices = make_grouped_split_indices(
        df=df,
        group_col=group_col,
        fracs=fracs,
        random_state=seed,
        verbose=True,
    )

    id_cols, y_cols = save_metadata_and_targets(use, split_indices, out_dir)

    geno_cols = [c for c in feature_cols if c.startswith("geno_")]
    mtb_cols = [c for c in feature_cols if c.startswith("mtb_")]
    mb_cols = [c for c in feature_cols if c.startswith("mb_")]

    summary_rows = []

    _, geno_shapes = transform_and_save_omic(
        use=use,
        split_indices=split_indices,
        omic_name="genotype",
        feature_cols=geno_cols,
        pipeline=build_genotype_pipeline(),
        out_dir=out_dir,
        id_cols=id_cols,
    )
    for split_name, info in geno_shapes.items():
        summary_rows.append({"seed": seed, "omic": "genotype", "split": split_name, **info})

    _, mtb_shapes = transform_and_save_omic(
        use=use,
        split_indices=split_indices,
        omic_name="metabolome",
        feature_cols=mtb_cols,
        pipeline=build_metabolome_pipeline(),
        out_dir=out_dir,
        id_cols=id_cols,
    )
    for split_name, info in mtb_shapes.items():
        summary_rows.append({"seed": seed, "omic": "metabolome", "split": split_name, **info})

    _, mb_shapes = transform_and_save_omic(
        use=use,
        split_indices=split_indices,
        omic_name="microbiome",
        feature_cols=mb_cols,
        pipeline=build_microbiome_pipeline(),
        out_dir=out_dir,
        id_cols=id_cols,
    )
    for split_name, info in mb_shapes.items():
        summary_rows.append({"seed": seed, "omic": "microbiome", "split": split_name, **info})

    # Final safety check: every X file must align exactly with its y file.
    assert_same_samples_across_saved_files(out_dir, id_cols=id_cols)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "preprocessing_summary.csv", index=False)

    config = {
        "seed": seed,
        "group_col": group_col,
        "split_fracs": fracs,
        "id_columns_saved_with_omic_files": id_cols,
        "y_columns": y_cols,
        "target_handling": "Only Y_TRAITS are saved in y files. Plots missing any selected y trait are removed before splitting.",
        "selected_y_traits": Y_TRAITS,
        "phenotype_leakage_rule": "Phenotype columns are saved only in y files, not in genotype/metabolome/microbiome files.",
        "genotype": {
            "maf_filter": GENO_MIN_MAF,
            "imputer": "most_frequent fit on train only",
            "scaler": "StandardScaler fit on train only",
        },
        "metabolome": {
            "transform": "log1p",
            "imputer": "median fit on train only",
            "variance_threshold": METABOLOME_VARIANCE_THRESHOLD,
            "scaler": "StandardScaler fit on train only",
        },
        "microbiome": {
            "imputer": "constant 0",
            "prevalence_filter": MICROBIOME_MIN_PREVALENCE,
            "clr_pseudocount": MICROBIOME_CLR_PSEUDOCOUNT,
            "scaler": "StandardScaler fit on train only",
        },
    }
    with open(out_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved seed {seed} files to: {out_dir}")
    return summary


def save_preprocessed_all_seeds(
    paths=PATHS,
    seeds=SEEDS,
    output_root=OUTPUT_DIR,
    fracs=SPLIT_FRACS,
):
    """Main entry point: save one preprocessed folder per seed."""
    df, feature_cols = load_and_merge(paths=paths, verbose=True)

    all_summaries = []
    for seed in seeds:
        print(f"\n=== preprocessing seed {seed} ===")
        seed_summary = save_preprocessed_for_seed(
            df=df,
            feature_cols=feature_cols,
            seed=seed,
            output_root=output_root,
            fracs=fracs,
            group_col=GROUP_COL,
        )
        all_summaries.append(seed_summary)

    full_summary = pd.concat(all_summaries, ignore_index=True)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    full_summary.to_csv(Path(output_root) / "all_seeds_preprocessing_summary.csv", index=False)
    print(f"\nDone. All outputs saved under: {output_root}/")
    return full_summary


if __name__ == "__main__":
    save_preprocessed_all_seeds()
