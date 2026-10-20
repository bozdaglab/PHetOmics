from pathlib import Path
from urllib.parse import quote
import re
import time

import pandas as pd
import requests
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from metabolome_seed_helpers import (get_global_metabolite_columns,strip_mtb_prefix,)
# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(".")
GRAPH_DIR = BASE_DIR / "graph_output"

SMILE_DIR = GRAPH_DIR / "compound_smiles_mapping"
SMILE_DIR.mkdir(exist_ok=True, parents=True)

SMILES_OUT_PATH = SMILE_DIR / "compound_smiles.csv"
AUDIT_OUT_PATH = SMILE_DIR / "compound_smiles_mapping_audit.csv"
UNMATCHED_OUT_PATH = SMILE_DIR / "compound_smiles_unmatched.csv"

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Keep requests slower than PubChem's recommended maximum rate.
SLEEP_SECONDS = 0.25
TIMEOUT_SECONDS = 30


# ============================================================
# Compound-name cleaning
# ============================================================
def build_name_candidates(column_name: str) -> list[str]:
    """
    Build possible PubChem query names from metabolome node IDs.

    Important:
    - compound_col stays as mtb_* for graph node alignment.
    - PubChem query removes mtb_ before searching.
    """
    raw = strip_mtb_prefix(str(column_name).strip())

    # R often adds X before names that start with numbers.
    without_leading_x = re.sub(r"^X(?=\d)", "", raw)

    candidates = []

    # Candidate 1: preserve likely prime notation.
    prime_version = without_leading_x.replace("..", "'-")
    prime_version = prime_version.replace(".", "-")
    prime_version = re.sub(r"-+", "-", prime_version).strip("-")
    candidates.append(prime_version)

    # Candidate 2: replace dots with spaces.
    space_version = without_leading_x.replace("..", " ")
    space_version = space_version.replace(".", " ")
    space_version = re.sub(r"\s+", " ", space_version).strip()
    candidates.append(space_version)

    # Candidate 3: replace dots with hyphens.
    hyphen_version = without_leading_x.replace("..", "-")
    hyphen_version = hyphen_version.replace(".", "-")
    hyphen_version = re.sub(r"-+", "-", hyphen_version).strip("-")
    candidates.append(hyphen_version)

    # Candidate 4: raw name without mtb_.
    candidates.append(raw)

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)

    return unique_candidates

# ============================================================
# PubChem lookup
# ============================================================
def request_pubchem_properties(query_name: str) -> dict | None:
    """
    Query PubChem using a compound name and return SMILES and identifiers.

    PubChem currently uses:
        SMILES             -> isomeric/stereochemistry-aware SMILES
        ConnectivitySMILES -> connectivity-level SMILES

    A fallback request using historical property names is included for
    compatibility.
    """

    encoded_name = quote(query_name, safe="")

    property_sets = [
        "SMILES,ConnectivitySMILES,InChIKey,IUPACName,MolecularFormula",
        "IsomericSMILES,CanonicalSMILES,InChIKey,IUPACName,MolecularFormula",
    ]

    for properties in property_sets:
        url = (
            f"{PUBCHEM_BASE}/compound/name/{encoded_name}/"
            f"property/{properties}/JSON"
        )

        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)

            if response.status_code == 404:
                continue

            if response.status_code != 200:
                continue

            data = response.json()
            records = data.get("PropertyTable", {}).get("Properties", [])

            if not records:
                continue

            record = records[0]

            smiles = (
                record.get("SMILES")
                or record.get("IsomericSMILES")
                or record.get("ConnectivitySMILES")
                or record.get("CanonicalSMILES")
            )

            connectivity_smiles = (
                record.get("ConnectivitySMILES")
                or record.get("CanonicalSMILES")
            )

            if not smiles:
                continue

            return {
                "pubchem_cid": record.get("CID"),
                "smiles": smiles,
                "connectivity_smiles": connectivity_smiles,
                "inchikey": record.get("InChIKey"),
                "iupac_name": record.get("IUPACName"),
                "molecular_formula": record.get("MolecularFormula"),
            }

        except requests.RequestException:
            continue
        except ValueError:
            continue

    return None


def lookup_compound(column_name: str) -> dict:
    candidates = build_name_candidates(column_name)

    for candidate in candidates:
        result = request_pubchem_properties(candidate)
        time.sleep(SLEEP_SECONDS)

        if result is not None:
            return {
                "compound_col": column_name,
                "query_used": candidate,
                "match_status": "matched",
                **result,
            }

    return {
        "compound_col": column_name,
        "query_used": "",
        "match_status": "unmatched",
        "pubchem_cid": None,
        "smiles": "",
        "connectivity_smiles": "",
        "inchikey": "",
        "iupac_name": "",
        "molecular_formula": "",
    }


# ============================================================
# Main
# ============================================================
def main():
    print("Collecting metabolite columns from all seed folders...")

    compound_cols = get_global_metabolite_columns()

    print(f"Number of unique metabolite nodes across all seeds: {len(compound_cols)}")

    results = []

    for index, compound in enumerate(compound_cols, start=1):
        print(f"[{index}/{len(compound_cols)}] Searching: {compound}")

        result = lookup_compound(compound)
        results.append(result)

        if result["match_status"] == "matched":
            print(
                f"    Matched using '{result['query_used']}' "
                f"| CID={result['pubchem_cid']}"
            )
        else:
            print("    Not matched")

    audit_df = pd.DataFrame(results)

    # Full audit file: includes matched and unmatched compounds.
    audit_df.to_csv(AUDIT_OUT_PATH, index=False)

    # Final file consumed by the embedding/edge-generation script.
    matched_df = audit_df[
        audit_df["match_status"] == "matched"
    ][
        [
            "compound_col",
            "smiles",
            "pubchem_cid",
            "query_used",
            "connectivity_smiles",
            "inchikey",
            "iupac_name",
            "molecular_formula",
        ]
    ].copy()

    matched_df.to_csv(SMILES_OUT_PATH, index=False)

    # Unmatched compounds for manual resolution.
    unmatched_df = audit_df[
        audit_df["match_status"] == "unmatched"
    ][["compound_col"]].copy()

    unmatched_df.to_csv(UNMATCHED_OUT_PATH, index=False)

    print("\n======================================")
    print("SMILES retrieval completed")
    print("======================================")
    print(f"Total compounds      : {len(compound_cols)}")
    print(f"Matched compounds    : {len(matched_df)}")
    print(f"Unmatched compounds  : {len(unmatched_df)}")

    print("\nCreated files:")
    print(f" - {SMILES_OUT_PATH}")
    print(f" - {AUDIT_OUT_PATH}")
    print(f" - {UNMATCHED_OUT_PATH}")

    if len(unmatched_df) > 0:
        print("\nUnmatched compounds need manual checking:")
        print(unmatched_df.to_string(index=False))


if __name__ == "__main__":
    main()