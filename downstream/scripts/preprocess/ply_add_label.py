#!/usr/bin/env python3
"""
Add per-vertex labels to PLY files using a residue_id -> label mapping from a TXT file.

TXT format (repeats in triplets of lines):
>protein_id
SEQUENCE_AS_LETTERS
010010001100...  (same length as sequence; characters '0'/'1')

PLY assumptions:
- Each vertex has a "residue_id" property (float) that indexes into the label string.
- Indices may be 0-based or 1-based; this script auto-detects or can be forced via flag.
- ASCII PLY is handled natively; binary PLY is handled if `plyfile` is installed.

Usage:
  python ply_add_label.py --ply-dir ./plys --labels-txt labels.txt --out-dir ./out
  # Optional flags:
  --recursive           # process PLYs in subfolders
  --force-base 0|1      # override auto-detection of residue index base
  --id-pattern REGEX    # regex to extract protein_id from filename (default: r'^([^_.]+)')
  --dry-run             # parse & validate, but do not write outputs
  --verbose             # print extra logs

Outputs new PLY files with an added vertex property:
  property int label
... and writes vertices with the integer label appended to each vertex row.
Faces and their properties are preserved as-is.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def read_label_txt(txt_path: Path) -> Dict[str, List[int]]:
    """
    Parse the labels TXT into a dict: {protein_id: [0/1, ...]}.
    Performs basic validations.
    """
    mapping: Dict[str, List[int]] = {}
    with txt_path.open('r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    i = 0
    triplet = 0
    while i < len(lines):
        # Skip empties
        if not lines[i].strip():
            i += 1
            continue
        if not lines[i].startswith('>'):
            raise ValueError(f"Expected line starting with '>' at line {i+1}, got: {lines[i][:40]}")
        prot_id = lines[i][1:].strip()
        if i+2 >= len(lines):
            raise ValueError(f"Incomplete triplet for protein {prot_id} starting at line {i+1}")
        seq = lines[i+1].strip()
        lab = lines[i+2].strip()

        if not seq or not lab:
            raise ValueError(f"Empty sequence/label for protein {prot_id}")
        if len(seq) != len(lab):
            raise ValueError(f"Length mismatch for {prot_id}: seq={len(seq)}, labels={len(lab)}")

        # Validate label chars and transform to list[int]
        label_list = []
        for idx, ch in enumerate(lab, start=1):
            if ch not in ('0', '1'):
                raise ValueError(f"Invalid label '{ch}' for {prot_id} at position {idx}")
            label_list.append(0 if ch == '0' else 1)

        mapping[prot_id] = label_list
        i += 3
        triplet += 1
    if not mapping:
        raise ValueError("No protein entries parsed from TXT file.")
    return mapping

def extract_protein_id(filename: str, pattern: str) -> Optional[str]:
    """
    Extract protein id from filename using a regex.
    Default pattern captures everything before first '.' or '_' (e.g., A0A0A0 from A0A0A0_A.ply).
    """
    m = re.match(pattern, filename)
    if not m:
        return None
    return m.group(1)

def detect_base(res_ids: List[int], seq_len: int) -> int:
    """
    Auto-detect 0- or 1-based indexing.
    - If 1 <= min <= max <= seq_len -> base 1
    - elif 0 <= min <= max <= seq_len-1 -> base 0
    Else raise.
    """
    mn, mx = min(res_ids), max(res_ids)
    if 1 <= mn and mx <= seq_len:
        return 1
    if 0 <= mn and mx <= (seq_len - 1):
        return 0
    raise ValueError(f"Residue indices out of range (min={mn}, max={mx}, seq_len={seq_len}). "
                     "Use --force-base 0|1 if this is expected.")

def try_import_plyfile():
    try:
        import plyfile  # type: ignore
        return plyfile
    except Exception:
        return None

def is_ascii_ply(header_lines: List[str]) -> bool:
    for line in header_lines:
        if line.startswith("format "):
            return "ascii" in line
    # Fallback assumption: if no format line, assume ascii (some exporters omit it)
    return True

def parse_ply_ascii(path: Path):
    """
    Minimal ASCII PLY parser for the expected structure.
    Returns (header_lines, vertex_props, face_props, vertices, faces)
    - vertex_props: list[str] in order (e.g., ['x','y','z','charge',...,'nz'])
    - vertices: List[List[str]] rows as strings (we will append label and write back)
    - faces: List[str] raw lines after vertex block
    """
    with path.open('r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Parse header
    header_lines = []
    i = 0
    if not lines or not lines[0].strip().startswith("ply"):
        raise ValueError("Not a PLY file (missing 'ply' magic).")
    while i < len(lines):
        header_lines.append(lines[i])
        if lines[i].strip() == "end_header":
            i += 1
            break
        i += 1
    else:
        raise ValueError("Missing 'end_header' in PLY.")

    # Extract counts and properties
    vertex_count = None
    face_count = None
    vertex_props: List[str] = []
    face_prop_lines: List[str] = []
    in_vertex_props = False
    in_face_props = False

    for h in header_lines:
        hs = h.strip()
        if hs.startswith("element vertex "):
            vertex_count = int(hs.split()[-1])
            in_vertex_props = True
            in_face_props = False
            continue
        if hs.startswith("element face "):
            face_count = int(hs.split()[-1])
            in_vertex_props = False
            in_face_props = True
            continue
        if hs.startswith("element ") and not (hs.startswith("element vertex") or hs.startswith("element face")):
            # Another element not handled
            pass
        if hs.startswith("property ") and in_vertex_props:
            parts = hs.split()
            # property float x -> name is last token
            vertex_props.append(parts[-1])
        if hs.startswith("property ") and in_face_props:
            face_prop_lines.append(h)

    if vertex_count is None or face_count is None:
        raise ValueError("Could not find both 'element vertex' and 'element face' in header.")

    # Collect vertex rows
    vertex_rows = []
    for _ in range(vertex_count):
        if i >= len(lines):
            raise ValueError(f"Unexpected EOF while reading vertices ({len(vertex_rows)}/{vertex_count} read).")
        row = lines[i].strip()
        if not row:
            raise ValueError(f"Empty vertex row at line {i+1}.")
        vertex_rows.append(row.split())
        i += 1

    # Collect face rows (retain raw lines)
    face_rows = []
    for _ in range(face_count):
        if i >= len(lines):
            raise ValueError(f"Unexpected EOF while reading faces ({len(face_rows)}/{face_count} read).")
        face_rows.append(lines[i])
        i += 1

    return header_lines, vertex_props, face_prop_lines, vertex_rows, face_rows

def write_ply_ascii(out_path: Path, header_lines: List[str], vertex_props: List[str],
                    vertex_rows_with_label: List[List[str]], face_prop_lines: List[str], face_rows: List[str]):
    """
    Write ASCII PLY with an added 'property int label' to the vertex props.
    """
    with out_path.open('w', encoding='utf-8') as f:
        # Rebuild header with new property appended
        # We will reconstruct the header to ensure property position is correct.
        # Approach: iterate header_lines and inject the new property after the last vertex property.
        emitted_label_prop = False
        in_vertex_props = False
        for line in header_lines:
            s = line.strip()
            if s.startswith("element vertex "):
                f.write(line + "\n")
                in_vertex_props = True
                continue
            if s.startswith("element face "):
                # If we haven't written the label prop yet, write it just before the face element
                if in_vertex_props and not emitted_label_prop:
                    f.write("property int label\n")
                    emitted_label_prop = True
                in_vertex_props = False
                f.write(line + "\n")
                continue
            if s.startswith("property ") and in_vertex_props:
                f.write(line + "\n")
                continue
            if s == "end_header":
                # If header prematurely ends, ensure we added the label prop
                if in_vertex_props and not emitted_label_prop:
                    f.write("property int label\n")
                    emitted_label_prop = True
                f.write(line + "\n")
                continue
            # other header lines (format, comments, etc.)
            f.write(line + "\n")

        # Write vertices
        for row in vertex_rows_with_label:
            f.write(" ".join(map(str, row)) + "\n")

        # Write faces
        for row in face_rows:
            f.write(row + "\n")

def process_one_ascii(ply_path: Path, out_path: Path, label_map: Dict[str, List[int]],
                      prot_id: str, force_base: Optional[int], verbose: bool):
    # Parse
    header_lines, vertex_props, face_prop_lines, vertex_rows, face_rows = parse_ply_ascii(ply_path)

    if "residue_id" not in vertex_props:
        raise ValueError(f"{ply_path.name}: missing 'residue_id' in vertex properties.")
    resid_idx = vertex_props.index("residue_id")

    labels = label_map.get(prot_id)
    if labels is None:
        raise KeyError(f"Protein id '{prot_id}' not found in labels TXT for file {ply_path.name}.")

    # Gather residue ids as ints (round if slightly off)
    resid_vals: List[int] = []
    for row in vertex_rows:
        try:
            v = float(row[resid_idx])
        except Exception:
            raise ValueError(f"{ply_path.name}: could not parse residue_id value '{row[resid_idx]}'")
        iv = int(round(v))
        if abs(v - iv) > 1e-6:
            raise ValueError(f"{ply_path.name}: residue_id value {v} not close to an integer at vertex row.")
        resid_vals.append(iv)

    base = force_base if force_base in (0, 1) else detect_base(resid_vals, len(labels))
    if verbose:
        print(f"[INFO] {ply_path.name}: using {'1-based' if base==1 else '0-based'} indexing; vertices={len(vertex_rows)}")

    # Build label per vertex
    vertex_rows_with_label = []
    for rid, row in zip(resid_vals, vertex_rows):
        idx = rid - 1 if base == 1 else rid
        if idx < 0 or idx >= len(labels):
            raise IndexError(f"{ply_path.name}: residue index {rid} (base={base}) out of range for seq_len={len(labels)}")
        lbl = labels[idx]
        # Append integer label to the row
        vertex_rows_with_label.append(row + [str(int(lbl))])

    # Write out
    write_ply_ascii(out_path, header_lines, vertex_props, vertex_rows_with_label, face_prop_lines, face_rows)

def process_one_binary_with_plyfile(ply_path: Path, out_path: Path, label_map: Dict[str, List[int]],
                                    prot_id: str, force_base: Optional[int], verbose: bool):
    plyfile = try_import_plyfile()
    if plyfile is None:
        raise RuntimeError("Binary PLY detected but 'plyfile' is not installed. Install via: pip install plyfile")

    from plyfile import PlyData, PlyElement
    pd = PlyData.read(ply_path)

    if "vertex" not in pd:
        raise ValueError(f"{ply_path.name}: no 'vertex' element.")
    vertex = pd["vertex"]
    props = vertex.data.dtype.names
    if "residue_id" not in props:
        raise ValueError(f"{ply_path.name}: 'vertex' missing 'residue_id' property.")

    resid_vals_float = vertex["residue_id"]
    resid_vals_int = []
    for v in resid_vals_float:
        iv = int(round(float(v)))
        if abs(float(v) - iv) > 1e-6:
            raise ValueError(f"{ply_path.name}: residue_id value {v} not close to an integer.")
        resid_vals_int.append(iv)

    labels = label_map.get(prot_id)
    if labels is None:
        raise KeyError(f"Protein id '{prot_id}' not found in labels TXT for file {ply_path.name}.")

    base = force_base if force_base in (0, 1) else detect_base(resid_vals_int, len(labels))
    if verbose:
        print(f"[INFO] {ply_path.name}: using {'1-based' if base==1 else '0-based'} indexing; vertices={len(resid_vals_int)}")

    # Build new structured array with 'label' appended
    import numpy as np
    lbl_values = np.empty(len(resid_vals_int), dtype=np.int32)
    for i, rid in enumerate(resid_vals_int):
        idx = rid - 1 if base == 1 else rid
        if idx < 0 or idx >= len(labels):
            raise IndexError(f"{ply_path.name}: residue index {rid} (base={base}) out of range for seq_len={len(labels)}")
        lbl_values[i] = int(labels[idx])

    # Extend dtype with a new field 'label'
    old_dtype = vertex.data.dtype
    new_dtype = old_dtype.descr + [('label', '<i4')]
    new_data = np.empty(vertex.count, dtype=new_dtype)
    for name in old_dtype.names:
        new_data[name] = vertex.data[name]
    new_data['label'] = lbl_values

    new_vertex = PlyElement.describe(new_data, 'vertex')
    # Replace vertex in PlyData and write out
    # Preserve other elements (e.g., face)
    other_elements = [el for el in pd.elements if el.name != 'vertex']
    new_pd = PlyData([new_vertex] + other_elements, text=pd.text)
    new_pd.write(out_path)

def process_ply(ply_path: Path, out_dir: Path, label_map: Dict[str, List[int]],
                id_pattern: str, force_base: Optional[int], dry_run: bool, verbose: bool):
    prot_id = extract_protein_id(ply_path.name, id_pattern)
    if prot_id is None:
        raise ValueError(f"Could not extract protein id from filename '{ply_path.name}' using pattern '{id_pattern}'")

    # Prefer ASCII native; if binary, try plyfile
    # Read just the header quickly
    with ply_path.open('rb') as fb:
        head = fb.read(2048).decode(errors='ignore')
    header_lines = head.splitlines()
    ascii_like = is_ascii_ply(header_lines)

    out_path = out_dir / ply_path.name
    if dry_run:
        if verbose:
            print(f"[DRY-RUN] Would process {ply_path} -> {out_path} (ASCII={ascii_like})")
        return

    if ascii_like:
        process_one_ascii(ply_path, out_path, label_map, prot_id, force_base, verbose)
    else:
        process_one_binary_with_plyfile(ply_path, out_path, label_map, prot_id, force_base, verbose)

def main():
    ap = argparse.ArgumentParser(description="Add per-vertex labels to PLY files using residue_id and a labels TXT.")
    ap.add_argument("--ply-dir", required=True, type=Path, help="Directory containing input .ply files")
    ap.add_argument("--labels-txt", required=True, type=Path, help="TXT file with protein sequences and 0/1 labels")
    ap.add_argument("--out-dir", required=True, type=Path, help="Directory to write updated .ply files")
    ap.add_argument("--recursive", action="store_true", help="Process .ply files in subdirectories")
    ap.add_argument("--id-pattern", default=r"^([^_.]+)", help="Regex to extract protein_id from filename (default: before first '.' or '_')")
    ap.add_argument("--force-base", choices=['0', '1'], help="Force residue index base (0 or 1)")
    ap.add_argument("--dry-run", action="store_true", help="Validate only; do not write outputs")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    if not args.ply_dir.is_dir():
        ap.error(f"--ply-dir '{args.ply_dir}' is not a directory")
    if not args.labels_txt.is_file():
        ap.error(f"--labels-txt '{args.labels_txt}' does not exist")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse label map
    label_map = read_label_txt(args.labels_txt)
    if args.verbose:
        print(f"[INFO] Parsed {len(label_map)} protein entries from {args.labels_txt}")

    # Gather ply files
    pattern = "**/*.ply" if args.recursive else "*.ply"
    ply_files = sorted(args.ply_dir.glob(pattern))
    if not ply_files:
        print(f"[WARN] No .ply files found in {args.ply_dir} (recursive={args.recursive})")
        return

    force_base = int(args.force_base) if args.force_base is not None else None
    errors = 0
    for ply_path in ply_files:
        try:
            process_ply(ply_path, args.out_dir, label_map, args.id_pattern, force_base, args.dry_run, args.verbose)
            if args.verbose and not args.dry_run:
                print(f"[OK] Wrote {args.out_dir / ply_path.name}")
        except Exception as e:
            errors += 1
            print(f"[ERROR] {ply_path.name}: {e}", file=sys.stderr)

    if errors:
        sys.exit(1)

if __name__ == "__main__":
    main()
