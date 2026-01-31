#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given a txt with UniProt accessions (one per line), for each ID:
1) Use RCSB Search API to detect experimental structures;
2) If none, download AlphaFold PDB and compute mean pLDDT;
3) Output CSV: protein_id, has_experimental, confidence, note
   (confidence=100 if experimental structure exists).
Dependencies: requests (pip install requests)
Usage: python scripts/preprocess/check_struct_confidence.py ids.txt output.csv
"""

import sys, csv, time, requests
from typing import Optional, Tuple
from tqdm import tqdm

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
AFDB_PRED_API   = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

# ---- Utilities --------------------------------------------------------------

def clean_id(s: str) -> str:
    return s.strip().split()[0]

def rcsb_has_experimental(uniprot_acc: str, timeout: float = 20.0) -> bool:
    """Check whether any experimental structure exists for a UniProt accession."""
    payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        # Match by UniProt accession
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_acc
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        # Experimental structures only
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }
    try:
        r = requests.post(RCSB_SEARCH_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        total = js.get("total_count", 0)
        if total and total > 0:
            return True
        # Some responses omit total_count but include result_set
        rs = js.get("result_set") or []
        return len(rs) > 0
    except Exception:
        return False  # On error, treat as not found

def parse_plddt_from_pdb_text(pdb_text: str) -> Optional[float]:
    """Parse mean pLDDT from AlphaFold PDB B-factors (CA atoms)."""
    values = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        # PDB columns: atom name 13-16 (0-based [12:16]), B-factor 61-66 ([60:66])
        try:
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            bfac_str = line[60:66].strip()
            if bfac_str:
                values.append(float(bfac_str))
        except Exception:
            continue
    if values:
        return sum(values) / len(values)
    return None

def fetch_afdb_mean_plddt(uniprot_acc: str, timeout: float = 30.0) -> Tuple[Optional[float], str]:
    """
    Call AFDB API to get model PDB URL, download, and compute mean pLDDT.
    Returns: (mean_plddt, note)
    """
    url = AFDB_PRED_API.format(acc=uniprot_acc)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None, f"AFDB API HTTP {r.status_code}"
        arr = r.json()
        if not arr:
            return None, "AFDB has no prediction model"
        # Pick entry with largest coverage span
        def span(rec):
            return (rec.get("uniprotEnd", 0) or 0) - (rec.get("uniprotStart", 0) or 0)
        rec = max(arr, key=span)
        # Support possible field name changes
        pdb_url = (
            rec.get("pdbUrl")
            or rec.get("modelPdbUrl")
            or rec.get("pdb_url")
            or rec.get("model_url")
        )
        if not pdb_url:
            return None, "AFDB entry missing PDB URL"
        p = requests.get(pdb_url, timeout=timeout)
        if p.status_code != 200:
            return None, f"PDB download failed HTTP {p.status_code}"
        mean_plddt = parse_plddt_from_pdb_text(p.text)
        if mean_plddt is None:
            return None, "Failed to parse pLDDT from PDB"
        return float(f"{mean_plddt:.2f}"), "OK"
    except Exception as e:
        return None, f"Exception: {e}"

# ---- Main ---------------------------------------------------------------

def main(in_txt: str, out_csv: str, sleep_sec: float = 0.1):
    rows = []
    with open(in_txt, "r", encoding="utf-8") as f:
        ids = [clean_id(x) for x in f if x.strip()]
    for acc in tqdm(ids):
        has_exp = rcsb_has_experimental(acc)
        if has_exp:
            rows.append({
                "protein_id": acc,
                "has_experimental": True,
                "confidence": 100,
                "note": "experimental structure found"
            })
        else:
            mean_plddt, note = fetch_afdb_mean_plddt(acc)
            rows.append({
                "protein_id": acc,
                "has_experimental": False,
                "confidence": mean_plddt if mean_plddt is not None else "",
                "note": note
            })
        time.sleep(sleep_sec)  # gentle throttling
    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["protein_id","has_experimental","confidence","note"])
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/preprocess/check_struct_confidence.py ids.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
