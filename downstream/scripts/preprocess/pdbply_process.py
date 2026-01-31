import argparse
import os
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppi.paths import get_data_root, get_vertex_residue_dir

def invert_json_mapping(json_dict):
    """Invert {"A_102": 4} to {4: 102}."""
    inverted = {}
    for key, value in json_dict.items():
        if key.startswith("A_"):
            try:
                residue = int(key[2:])  # extract 102
                inverted[value] = residue
            except ValueError:
                continue
    return inverted

def process_ply_and_json(ply_path, json_path, output_path=None):
    # Load JSON and build residue_id mapping.
    with open(json_path, 'r') as jf:
        json_data = json.load(jf)
        id_map = invert_json_mapping(json_data)

    with open(ply_path, 'r') as pf:
        lines = pf.readlines()

    # Locate header end and vertex count.
    header_end = 0
    vertex_count = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.strip().split()[-1])
        if line.strip() == "end_header":
            header_end = i
            break

    # Split header and data.
    header = lines[:header_end+1]
    vertex_lines = lines[header_end+1:header_end+1+vertex_count]
    face_lines = lines[header_end+1+vertex_count:]

    # Replace residue_id (6th column) in each vertex row.
    updated_vertex_lines = []
    for line in vertex_lines:
        parts = line.strip().split()
        if len(parts) >= 7:
            try:
                old_id = int(float(parts[5]))
                new_id = id_map.get(old_id, old_id)  # keep original if not found
                parts[5] = str(new_id)
            except ValueError:
                pass
        updated_vertex_lines.append(" ".join(parts) + "\n")

    # Merge and write output.
    output_file = output_path if output_path else ply_path
    with open(output_file, 'w') as out:
        out.writelines(header + updated_vertex_lines + face_lines)

def batch_process(ply_folder, json_folder, output_folder=None):
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for fname in os.listdir(ply_folder):
        if fname.endswith(".ply"):
            ply_path = os.path.join(ply_folder, fname)
            json_path = os.path.join(json_folder, fname.replace(".ply", ".json"))

            if not os.path.exists(json_path):
                print(f"[WARN] JSON file not found for {fname}, skipping.")
                continue

            output_path = os.path.join(output_folder, fname) if output_folder else None
            process_ply_and_json(ply_path, json_path, output_path)
            print(f"Processed {fname}")

def load_protein_lengths(csv_path):
    protein_lengths = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                protein_id, length = row
                protein_lengths[protein_id.strip()] = int(length)
    return protein_lengths

def get_pdb_protein_length(pdb_file):
    residues = set()
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                res_id = (line[21], int(line[22:26]))  # (chainID, resSeq)
                residues.add(res_id)
    return len(residues)

def check_pdb_lengths(folder_path, csv_path, log_path="length_mismatches.txt"):
    protein_lengths = load_protein_lengths(csv_path)

    mismatches = []
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith(".pdb"):
                protein_id = os.path.splitext(file)[0]
                pdb_path = os.path.join(root, file)
                actual_length = get_pdb_protein_length(pdb_path)
                expected_length = protein_lengths.get(protein_id)

                if expected_length is None:
                    mismatches.append(f"{protein_id}: not found in CSV\n")
                elif actual_length != expected_length:
                    mismatches.append(
                        f"{protein_id}: expected {expected_length}, got {actual_length}\n"
                    )

    with open(log_path, 'w') as log_file:
        log_file.writelines(mismatches)

    print(f"Check complete. Found {len(mismatches)} mismatches.")
    print(f"Details saved to: {log_path}")

def calculate_protein_lengths(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    protein_lengths = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            protein_id = lines[i][1:].strip()
            sequence = lines[i+1].strip()  # line i+1 is sequence
            # Ignore labels line (i+2) if present.
            protein_lengths[protein_id] = len(sequence)
            i += 3  # move to next protein
        else:
            i += 1

    # Write results to output file.
    with open(output_file, 'w') as f_out:
        for pid, length in protein_lengths.items():
            f_out.write(f"{pid},{length}\n")


def parse_args():
    ap = argparse.ArgumentParser(description="PLY/JSON residue id processing utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_batch = sub.add_parser("batch", help="Replace residue_id in PLY using JSON mapping.")
    sp_batch.add_argument("--ply-dir", default=str(get_data_root() / "ply_temp"))
    sp_batch.add_argument("--json-dir", default=str(get_vertex_residue_dir()))
    sp_batch.add_argument("--out-dir", default=str(get_data_root() / "ply_temp_right_residue"))

    sp_lengths = sub.add_parser("lengths", help="Compute protein lengths from triplet txt.")
    sp_lengths.add_argument("--input", required=True)
    sp_lengths.add_argument("--output", required=True)

    sp_check = sub.add_parser("check", help="Check PDB lengths against CSV.")
    sp_check.add_argument("--pdb-dir", required=True)
    sp_check.add_argument("--length-csv", required=True)
    sp_check.add_argument("--log", default="length_mismatches.txt")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "batch":
        batch_process(args.ply_dir, args.json_dir, args.out_dir)
    elif args.cmd == "lengths":
        calculate_protein_lengths(args.input, args.output)
    elif args.cmd == "check":
        check_pdb_lengths(args.pdb_dir, args.length_csv, args.log)
