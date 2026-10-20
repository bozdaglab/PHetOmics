from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(".")
PREPROCESSED_DIR = BASE_DIR / "preprocessed"

ID_COLS = {"sample_id", "variety", "varietyID", "block", "plot"}


def strip_mtb_prefix(col: str) -> str:
    col = str(col).strip()
    return col.replace("mtb_", "", 1) if col.startswith("mtb_") else col


def ensure_mtb_prefix(col: str) -> str:
    col = str(col).strip()
    return col if col.startswith("mtb_") else f"mtb_{col}"


def clean_compound_name(col: str) -> str:
    """
    Clean a metabolome column for display/PubChem search.

    Keeps node_id as mtb_* elsewhere.
    """
    name = strip_mtb_prefix(col)

    # Remove R-style X prefix before numeric compound names.
    name = re.sub(r"^X(?=\d|\.)", "", name)

    # Replace dots with spaces.
    name = re.sub(r"\.+", " ", name)

    # Normalize spaces.
    name = re.sub(r"\s+", " ", name).strip()

    return name


def get_metabolite_columns_from_file(path: Path) -> list[str]:
    df_head = pd.read_csv(path, nrows=0)
    return [
        c for c in df_head.columns
        if c not in ID_COLS and c.startswith("mtb_")
    ]


def get_seed_metabolite_columns(seed_dir: Path, verify_same_across_splits: bool = True) -> list[str]:
    """
    Return metabolite columns for one seed.

    Since preprocessing fits the feature filter on train and transforms
    val/test using the same output columns, train/val/test should match.
    """
    split_cols = {}

    for split in ["train", "val", "test"]:
        path = seed_dir / f"metabolome_{split}.csv"

        if not path.exists():
            continue

        split_cols[split] = get_metabolite_columns_from_file(path)

    if not split_cols:
        raise FileNotFoundError(f"No metabolome_train/val/test files found in {seed_dir}")

    if verify_same_across_splits:
        unique_col_sets = {tuple(cols) for cols in split_cols.values()}

        if len(unique_col_sets) != 1:
            raise ValueError(
                f"Metabolome columns differ across train/val/test inside {seed_dir}. "
                "This should not happen if preprocessing used one train-fitted pipeline."
            )

    # Use train columns if available, otherwise first available split.
    if "train" in split_cols:
        return split_cols["train"]

    return next(iter(split_cols.values()))


def get_global_metabolite_columns() -> list[str]:
    """
    Collect the union of all mtb_* columns across all seed folders.

    Order is preserved by first appearance across sorted seed folders.
    """
    seed_dirs = sorted(PREPROCESSED_DIR.glob("seed_*"))

    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed folders found under {PREPROCESSED_DIR}. "
            "Run preprocess.py first."
        )

    seen = set()
    ordered_cols = []

    for seed_dir in seed_dirs:
        seed_cols = get_seed_metabolite_columns(seed_dir)

        for col in seed_cols:
            if col not in seen:
                ordered_cols.append(col)
                seen.add(col)

    return ordered_cols