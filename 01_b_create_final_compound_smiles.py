from pathlib import Path
import time

import pandas as pd
import requests

from metabolome_seed_helpers import (
    get_global_metabolite_columns,
    ensure_mtb_prefix,
)
# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(".")
GRAPH_DIR = BASE_DIR / "graph_output"
SMILE_DIR = GRAPH_DIR / "compound_smiles_mapping"

AUTO_MATCHED_PATH = SMILE_DIR / "compound_smiles.csv"
MANUAL_MAPPING_PATH = SMILE_DIR / "compound_smiles_unmatched_manual.csv"

FINAL_OUTPUT_PATH = SMILE_DIR / "compound_smiles_final.csv"
MANUAL_AUDIT_PATH = SMILE_DIR / "compound_smiles_manual_mapping_audit.csv"
UNMATCHED_FINAL_PATH = SMILE_DIR / "compound_smiles_still_unmatched.csv"


# ============================================================
# Settings
# ============================================================

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
REQUEST_DELAY_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


# ============================================================
# Known corrections
# ============================================================
# This corrects the Excel-corrupted compound name and the incorrect
# multi-component CID shown in the previous screenshot.
KNOWN_MANUAL_CORRECTIONS = {
    "mtb_X1.Amino.1.c+A2:A98yclopentanecarboxylic.acid": {
        "compound_col": "mtb_X1.Amino.1.cyclopentanecarboxylic.acid",
        "pubchem_cid": 2901,
        "correction_note": (
            "Corrected corrupted compound name and replaced "
            "multi-component CID 146171660 with parent compound CID 2901."
        ),
    },
    # Backward-compatible version in case your manual file was edited before mtb_ was added.
    "X1.Amino.1.c+A2:A98yclopentanecarboxylic.acid": {
        "compound_col": "mtb_X1.Amino.1.cyclopentanecarboxylic.acid",
        "pubchem_cid": 2901,
        "correction_note": (
            "Corrected corrupted compound name and replaced "
            "multi-component CID 146171660 with parent compound CID 2901."
        ),
    },
}


# ============================================================
# Helper functions
# ============================================================
def standardize_cid(value):
    """
    Convert PubChem CID values such as 8742.0 to integer 8742.
    Return pandas NA when a CID is blank or invalid.
    """
    if pd.isna(value):
        return pd.NA

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return pd.NA


def apply_known_corrections(manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply corrections for known issues in the manually edited mapping file.
    """
    manual_df = manual_df.copy()

    if "correction_note" not in manual_df.columns:
        manual_df["correction_note"] = ""

    for incorrect_name, correction in KNOWN_MANUAL_CORRECTIONS.items():
        mask = manual_df["compound_col"] == incorrect_name

        if mask.any():
            print("\nApplying known correction:")
            print(f"  Incorrect row : {incorrect_name}")
            print(f"  Correct name  : {correction['compound_col']}")
            print(f"  Correct CID   : {correction['pubchem_cid']}")

            manual_df.loc[mask, "compound_col"] = correction["compound_col"]
            manual_df.loc[mask, "pubchem_cid"] = correction["pubchem_cid"]
            manual_df.loc[mask, "correction_note"] = correction["correction_note"]

    return manual_df


def retrieve_pubchem_properties_by_cid(cid: int) -> dict:
    """
    Retrieve chemical properties from PubChem using a manually selected CID.
    """

    property_requests = [
        "SMILES,ConnectivitySMILES,InChIKey,IUPACName,MolecularFormula",
        "IsomericSMILES,CanonicalSMILES,InChIKey,IUPACName,MolecularFormula",
    ]

    for properties in property_requests:
        url = (
            f"{PUBCHEM_BASE}/compound/cid/{cid}/"
            f"property/{properties}/JSON"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)

                if response.status_code == 200:
                    data = response.json()
                    records = data.get("PropertyTable", {}).get("Properties", [])

                    if not records:
                        break

                    record = records[0]

                    smiles = (
                        record.get("SMILES")
                        or record.get("IsomericSMILES")
                        or record.get("ConnectivitySMILES")
                        or record.get("CanonicalSMILES")
                        or ""
                    )

                    connectivity_smiles = (
                        record.get("ConnectivitySMILES")
                        or record.get("CanonicalSMILES")
                        or ""
                    )

                    if smiles:
                        return {
                            "smiles": smiles,
                            "connectivity_smiles": connectivity_smiles,
                            "inchikey": record.get("InChIKey", ""),
                            "iupac_name": record.get("IUPACName", ""),
                            "molecular_formula": record.get("MolecularFormula", ""),
                            "pubchem_fetch_status": "success",
                            "pubchem_fetch_error": "",
                        }

                if response.status_code == 404:
                    return {
                        "smiles": "",
                        "connectivity_smiles": "",
                        "inchikey": "",
                        "iupac_name": "",
                        "molecular_formula": "",
                        "pubchem_fetch_status": "failed",
                        "pubchem_fetch_error": f"CID {cid} not found in PubChem.",
                    }

                if attempt < MAX_RETRIES:
                    time.sleep(1)

            except requests.RequestException as error:
                if attempt == MAX_RETRIES:
                    return {
                        "smiles": "",
                        "connectivity_smiles": "",
                        "inchikey": "",
                        "iupac_name": "",
                        "molecular_formula": "",
                        "pubchem_fetch_status": "failed",
                        "pubchem_fetch_error": str(error),
                    }

                time.sleep(1)

    return {
        "smiles": "",
        "connectivity_smiles": "",
        "inchikey": "",
        "iupac_name": "",
        "molecular_formula": "",
        "pubchem_fetch_status": "failed",
        "pubchem_fetch_error": f"No structure properties returned for CID {cid}.",
    }


def process_automatic_matches(auto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare automatically matched compounds.
    """
    auto_df = auto_df.copy()

    auto_df["pubchem_cid"] = auto_df["pubchem_cid"].apply(standardize_cid)
    auto_df["match_status"] = "matched"
    auto_df["mapping_source"] = "automatic_pubchem_name_lookup"
    auto_df["correction_note"] = ""
    auto_df["pubchem_fetch_status"] = "already_retrieved"
    auto_df["pubchem_fetch_error"] = ""

    return auto_df


def process_manual_mappings(manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    For manually entered CIDs:
        - Retrieve PubChem properties
        - Mark resolved rows as manually_mapped

    For rows without CIDs:
        - Mark as unmatched
    """
    manual_df = manual_df.copy()
    manual_df = apply_known_corrections(manual_df)

    manual_df["pubchem_cid"] = manual_df["pubchem_cid"].apply(standardize_cid)

    results = []

    total = len(manual_df)

    for index, row in manual_df.iterrows():
        compound_col = row["compound_col"]
        cid = row["pubchem_cid"]
        correction_note = row.get("correction_note", "")

        print(f"[{index + 1}/{total}] Processing: {compound_col}")

        if pd.isna(cid):
            print("    Status: unmatched, no manually entered CID")

            results.append({
                "compound_col": compound_col,
                "smiles": "",
                "pubchem_cid": pd.NA,
                "query_used": "",
                "connectivity_smiles": "",
                "inchikey": "",
                "iupac_name": "",
                "molecular_formula": "",
                "match_status": "unmatched",
                "mapping_source": "no_mapping_available",
                "correction_note": correction_note,
                "pubchem_fetch_status": "not_requested",
                "pubchem_fetch_error": "",
            })

            continue

        cid = int(cid)

        print(f"    Manual CID: {cid}")
        properties = retrieve_pubchem_properties_by_cid(cid)

        if properties["pubchem_fetch_status"] == "success":
            print(f"    Status: manually_mapped | IUPAC: {properties['iupac_name']}")

            results.append({
                "compound_col": compound_col,
                "smiles": properties["smiles"],
                "pubchem_cid": cid,
                "query_used": f"manual_cid:{cid}",
                "connectivity_smiles": properties["connectivity_smiles"],
                "inchikey": properties["inchikey"],
                "iupac_name": properties["iupac_name"],
                "molecular_formula": properties["molecular_formula"],
                "match_status": "manually_mapped",
                "mapping_source": "manual_pubchem_cid",
                "correction_note": correction_note,
                "pubchem_fetch_status": properties["pubchem_fetch_status"],
                "pubchem_fetch_error": properties["pubchem_fetch_error"],
            })

        else:
            print(f"    Status: manual_mapping_failed | CID: {cid}")

            results.append({
                "compound_col": compound_col,
                "smiles": "",
                "pubchem_cid": cid,
                "query_used": f"manual_cid:{cid}",
                "connectivity_smiles": "",
                "inchikey": "",
                "iupac_name": "",
                "molecular_formula": "",
                "match_status": "manual_mapping_failed",
                "mapping_source": "manual_pubchem_cid",
                "correction_note": correction_note,
                "pubchem_fetch_status": properties["pubchem_fetch_status"],
                "pubchem_fetch_error": properties["pubchem_fetch_error"],
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(results)


def preserve_metabolome_column_order(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve metabolite order based on the union of mtb_* columns
    found across all seed folders.
    """
    original_compounds = get_global_metabolite_columns()

    order_df = pd.DataFrame({
        "compound_col": original_compounds,
        "original_compound_order": range(len(original_compounds))
    })

    final_df = final_df.copy()
    final_df["compound_col"] = final_df["compound_col"].apply(ensure_mtb_prefix)

    ordered_df = order_df.merge(final_df, on="compound_col", how="left")

    missing_mapping_rows = ordered_df["match_status"].isna()

    if missing_mapping_rows.any():
        missing_names = ordered_df.loc[missing_mapping_rows, "compound_col"].tolist()

        print("\nWarning: The following metabolome compounds do not appear")
        print("in either automatic or manual mapping files:")
        for name in missing_names:
            print(f" - {name}")

        ordered_df.loc[missing_mapping_rows, "match_status"] = "unmatched"
        ordered_df.loc[missing_mapping_rows, "mapping_source"] = "missing_from_mapping_files"

    return ordered_df.sort_values("original_compound_order").reset_index(drop=True)
# ============================================================
# Main
# ============================================================
def main():
    print("Loading input files...")

    auto_df = pd.read_csv(AUTO_MATCHED_PATH)
    manual_df = pd.read_csv(MANUAL_MAPPING_PATH)


    auto_df["compound_col"] = auto_df["compound_col"].apply(ensure_mtb_prefix)
    manual_df["compound_col"] = manual_df["compound_col"].apply(ensure_mtb_prefix)

    required_auto_columns = {
        "compound_col",
        "smiles",
        "pubchem_cid",
        "query_used",
        "connectivity_smiles",
        "inchikey",
        "iupac_name",
        "molecular_formula",
    }

    required_manual_columns = {
        "compound_col",
        "pubchem_cid",
    }

    if not required_auto_columns.issubset(auto_df.columns):
        missing = required_auto_columns - set(auto_df.columns)
        raise ValueError(
            f"Automatic matched file is missing required columns: {missing}"
        )

    if not required_manual_columns.issubset(manual_df.columns):
        missing = required_manual_columns - set(manual_df.columns)
        raise ValueError(
            f"Manual mapping file is missing required columns: {missing}"
        )

    print(f"Automatically matched rows : {len(auto_df)}")
    print(f"Originally unmatched rows   : {len(manual_df)}")
    print(f"Manually mapped CIDs found  : {manual_df['pubchem_cid'].notna().sum()}")
    print(f"Still without CIDs          : {manual_df['pubchem_cid'].isna().sum()}")

    print("\nPreparing automatic matches...")
    auto_processed = process_automatic_matches(auto_df)

    print("\nRetrieving structures for manually mapped CIDs...")
    manual_processed = process_manual_mappings(manual_df)

    manual_processed.to_csv(MANUAL_AUDIT_PATH, index=False)

    # Combine automatically mapped and manual/unresolved rows.
    final_df = pd.concat(
        [auto_processed, manual_processed],
        ignore_index=True,
        sort=False
    )

    # Prevent duplicate compound nodes.
    duplicate_compounds = final_df[
        final_df["compound_col"].duplicated(keep=False)
    ].sort_values("compound_col")

    if not duplicate_compounds.empty:
        print("\nWarning: Duplicate compound rows detected:")
        print(
            duplicate_compounds[
                ["compound_col", "pubchem_cid", "match_status"]
            ].to_string(index=False)
        )

        final_df = (
            final_df
            .sort_values(
                by="match_status",
                key=lambda values: values.map({
                    "manually_mapped": 0,
                    "matched": 1,
                    "manual_mapping_failed": 2,
                    "unmatched": 3,
                }).fillna(4)
            )
            .drop_duplicates("compound_col", keep="first")
            .reset_index(drop=True)
        )

    # Put compounds back in metabolome column order.
    final_df = preserve_metabolome_column_order(final_df)

    # Final clean column order.
    final_columns = [
        "original_compound_order",
        "compound_col",
        "match_status",
        "mapping_source",
        "pubchem_cid",
        "smiles",
        "connectivity_smiles",
        "inchikey",
        "iupac_name",
        "molecular_formula",
        "query_used",
        "correction_note",
        "pubchem_fetch_status",
        "pubchem_fetch_error",
    ]

    for column in final_columns:
        if column not in final_df.columns:
            final_df[column] = ""

    final_df = final_df[final_columns]

    # Save final combined file.
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)

    # Save only still-unresolved/problematic rows.
    unresolved_df = final_df[
        final_df["match_status"].isin(
            ["unmatched", "manual_mapping_failed"]
        )
    ].copy()

    unresolved_df.to_csv(UNMATCHED_FINAL_PATH, index=False)

    # Summary
    status_summary = (
        final_df["match_status"]
        .value_counts(dropna=False)
        .rename_axis("match_status")
        .reset_index(name="count")
    )

    print("\n========================================")
    print("Final compound SMILES mapping completed")
    print("========================================")
    print(status_summary.to_string(index=False))

    print("\nOutput files:")
    print(f" - Final combined mapping : {FINAL_OUTPUT_PATH}")
    print(f" - Manual mapping audit   : {MANUAL_AUDIT_PATH}")
    print(f" - Still unresolved       : {UNMATCHED_FINAL_PATH}")

    print("\nExpected status meanings:")
    print(" matched               = retrieved automatically from compound name")
    print(" manually_mapped       = CID entered manually and properties retrieved")
    print(" unmatched             = no CID available")
    print(" manual_mapping_failed = CID present but PubChem retrieval failed")

    print("\nImportant:")
    print(
        "Rows marked manually_mapped contain PubChem structures retrieved "
        "from your selected CID, but uncommon compounds should still be "
        "manually checked before generating final graph embeddings."
    )


if __name__ == "__main__":
    main()