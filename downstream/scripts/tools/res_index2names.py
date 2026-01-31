import argparse
import csv

from Bio.PDB import PDBParser


def extract_residue_mapping(pdb_path, chain_id="A", output_csv="residue_map.csv"):
    """
    Extract standard residues from a chain in a PDB file and save a mapping:
    residue_index (0-based) -> original residue id (e.g., A_13).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]
    
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found in the structure.")

    chain = model[chain_id]

    residue_list = []
    for i, residue in enumerate(chain):
        hetfield, resseq, icode = residue.id
        if hetfield == " ":  # skip hetero/water
            residue_list.append((i, f"{chain_id}_{resseq}"))

    # Save as CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["residue_index", "original_residue"])
        writer.writerows(residue_list)

    print(f"Saved {len(residue_list)} residues to {output_csv}")


def parse_args():
    ap = argparse.ArgumentParser(description="Extract residue index mapping from a PDB chain.")
    ap.add_argument("--pdb", required=True, help="Path to PDB file")
    ap.add_argument("--chain", default="A", help="Chain ID (default: A)")
    ap.add_argument("--out", default="residue_map.csv", help="Output CSV path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_residue_mapping(args.pdb, chain_id=args.chain, output_csv=args.out)
