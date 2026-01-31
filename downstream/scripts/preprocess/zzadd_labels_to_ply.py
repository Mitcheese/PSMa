#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add residue-level labels to PLY vertex data based on a FASTA-like TXT.
Each PLY's filename contains the protein ID (e.g., A0A011_A.ply -> A0A011).
We read the residue_id for each vertex and append a `label` property whose
value is the per-residue label from the TXT file.

Assumptions:
- PLY format: ASCII (format ascii 1.0). Binary PLY is not supported in this script.
- Vertex element appears before other elements (faces, etc.), which is standard.
- `residue_id` property exists and is a float. We cast it to int.
- Residue indexing is 1-based by default. Set --zero_based to treat 0-based indices.
- If a residue_id is out of range, we write --out_of_range_value (default: -1.0).

Usage:
    python add_labels_to_ply.py --txt labels.txt --ply_dir /path/to/plys --out_dir /path/to/out
Optional:
    --in_place                # overwrite original files (use with caution)
    --id_regex "^(\\w+)"      # custom regex to extract protein ID (first capture group)
    --id_split "_" --id_part 0 # split filename by delimiter and take the Nth part (default behavior)
    --zero_based              # treat residue_id as 0-based index
    --out_of_range_value -1   # value to write when residue_id is out of label string bounds

TXT format (repeating triplets of lines):
    >P15309
    MRAAPL......
    000001001...   (one char per residue; will be cast to float)
"""
import argparse
import os
import re
import sys
from typing import Dict, List, Tuple
import numpy as np

def load_labels(txt_path: str) -> Dict[str, str]:
    """Load labels from TXT. Expected repeating blocks:
       >PROTID
       SEQUENCE
       LABELS
       Returns dict: {PROTID: LABELS_STRING (whitespace stripped)}
    """
    labels = {}
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"TXT not found: {txt_path}")
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    i = 0
    while i < len(lines):
        if not lines[i].startswith('>'):
            raise ValueError(f"Expected '>' line at row {i+1}, got: {lines[i][:50]}")
        prot_id = lines[i][1:].strip()
        if i + 2 >= len(lines):
            raise ValueError(f"Incomplete triplet for protein {prot_id} near row {i+1}.")
        seq = lines[i+1].strip().replace(" ", "")
        lab = lines[i+2].strip().replace(" ", "")
        # Basic sanity check
        if len(lab) < len(seq):
            # Warn but accept; sometimes labels can be shorter (e.g., masked tail)
            sys.stderr.write(f"[WARN] Label shorter than sequence for {prot_id}: "
                             f"{len(lab)} < {len(seq)}. Will still proceed.\n")
        labels[prot_id] = lab
        i += 3
    return labels

def extract_id_from_filename(fname: str, id_regex: str, id_split: str, id_part: int) -> str:
    stem = os.path.splitext(os.path.basename(fname))[0]
    if id_regex:
        m = re.search(id_regex, stem)
        if not m:
            raise ValueError(f"Filename '{fname}' does not match id_regex '{id_regex}'.")
        return m.group(1)
    # default split
    parts = stem.split(id_split) if id_split else [stem]
    if id_part < 0 or id_part >= len(parts):
        raise IndexError(f"id_part {id_part} out of range for filename '{fname}' split by '{id_split}'.")
    return parts[id_part]

def parse_header(header_lines: List[str]) -> Tuple[str, int, List[str], int, List[str]]:
    """
    Parse PLY header lines.
    Returns:
      (format_line, vertex_count, vertex_props, vertex_prop_start_idx, full_header_lines)
    Where vertex_props is the ordered list of property names for the vertex element,
    and vertex_prop_start_idx is the index in header_lines where the first vertex
    property appears (to allow inserting 'property float label' after them).
    """
    format_line = None
    vertex_count = None
    vertex_props: List[str] = []
    vertex_prop_start_idx = -1

    in_vertex_props = False
    for idx, ln in enumerate(header_lines):
        if ln.startswith('format '):
            format_line = ln
        if ln.startswith('element vertex'):
            in_vertex_props = True
            # parse count
            parts = ln.split()
            if len(parts) != 3 or not parts[2].isdigit():
                # allow non-digit if it's number-like
                try:
                    vertex_count = int(float(parts[2]))
                except Exception:
                    raise ValueError(f"Cannot parse vertex count from line: {ln}")
            else:
                vertex_count = int(parts[2])
            continue
        if ln.startswith('element ') and not ln.startswith('element vertex'):
            # another element begins, stop collecting vertex props
            in_vertex_props = False
        if in_vertex_props and ln.startswith('property '):
            # property <type> <name>
            parts = ln.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed property line: {ln}")
            prop_name = parts[-1]
            vertex_props.append(prop_name)
            if vertex_prop_start_idx == -1:
                vertex_prop_start_idx = idx
    if format_line is None:
        raise ValueError("PLY header missing 'format' line.")
    if vertex_count is None:
        raise ValueError("PLY header missing 'element vertex' line.")
    if not vertex_props:
        raise ValueError("No vertex properties found in header.")
    return format_line, vertex_count, vertex_props, vertex_prop_start_idx, header_lines

def add_label_property_to_header(header_lines: List[str],
                                vertex_props: List[str],
                                vertex_prop_start_idx: int) -> List[str]:
    """Insert 'property float label' if not present, after existing vertex properties."""
    if 'label' in vertex_props:
        return header_lines  # already present
    # Find where vertex properties end: the first header line after the last vertex prop
    # We know vertex_prop_start_idx is where the first vertex prop was found.
    # Count how many vertex props exist:
    num_vertex_props = len(vertex_props)
    insert_after_idx = vertex_prop_start_idx + num_vertex_props - 1
    new_header = header_lines[:insert_after_idx + 1] + ["property float label"] + header_lines[insert_after_idx + 1:]
    return new_header

def build_vertex_labels(residue_id_f32: np.ndarray,
                        label_str: str,
                        zero_based: bool | None = None,
                        eps: float = 1e-3):
    """
    Build vertex labels (0/1, uint8) from PLY residue_id and a label_str of '0'/'1'.
    - Auto-detect 0/1-based indexing (or pass zero_based explicitly)
    - Round before int conversion to avoid float noise (e.g., 0.999/1.001)
    - Returns: labels(uint8), bad_idx(np.ndarray[int])  // bad_idx are out-of-range rows
    """
    # 1) Build sequence array (0/1)
    try:
        seq = np.fromiter((1 if c == '1' else 0 for c in label_str), dtype=np.uint8, count=len(label_str))
    except Exception as e:
        raise ValueError(f"label_str is not a 0/1 string: {e}")
    L = int(seq.shape[0])

    # 2) residue_id: round then cast to int to avoid truncation traps
    rid = np.asarray(residue_id_f32, dtype=np.float64)
    if not np.isfinite(rid).all():
        bad = np.flatnonzero(~np.isfinite(rid))
        raise ValueError(f"residue_id has NaN/Inf; bad rows={bad.size}, examples={bad[:5]}")

    rid_int = np.rint(rid).astype(np.int64)  # round
    # 3) Auto-detect 0/1-based
    if zero_based is None:
        # Heuristic: if min==0 and max<=L-1, prefer 0-based; else 1-based
        zero_based = int(rid_int.min()) == 0

    idx = rid_int if zero_based else (rid_int - 1)

    # 4) Bounds check
    bad_mask = (idx < 0) | (idx >= L)
    bad_idx = np.flatnonzero(bad_mask)

    labels = np.zeros_like(rid_int, dtype=np.uint8)
    ok = ~bad_mask
    labels[ok] = seq[idx[ok]]
    for l in labels:
        if l < 0:
            raise ValueError("< 0")

    return labels, bad_idx, zero_based

def process_ply_ascii(in_path: str,
                      out_path: str,
                      label_str: str,
                      zero_based: bool,
                      out_of_range_value: float,
                      id_hint: str = "") -> None:
    """Read an ASCII PLY, append label column to vertex data, and write to out_path."""
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().splitlines()

    # Split header and body
    try:
        end_idx = lines.index('end_header')
    except ValueError:
        # Try to locate with potential trailing spaces
        end_idx = -1
        for i, ln in enumerate(lines):
            if ln.strip() == 'end_header':
                end_idx = i
                break
        if end_idx == -1:
            raise ValueError(f"'end_header' not found in {in_path}")

    header_lines = lines[:end_idx + 1]
    body_lines = lines[end_idx + 1:]

    format_line, vertex_count, vertex_props, vertex_prop_start_idx, _ = parse_header(header_lines)
    if vertex_prop_start_idx == -1:
        raise ValueError("prop start = -1")
    if 'ascii' not in format_line:
        raise NotImplementedError(f"Only ASCII PLY is supported. Found format: '{format_line}'. "
                                  f"Convert to ASCII or use a library (e.g., plyfile).")

    if 'residue_id' not in vertex_props:
        raise ValueError(f"'residue_id' property not found in vertex properties for file: {in_path}")

    res_idx = vertex_props.index('residue_id')

    # Prepare new header with 'label' property appended (if missing)
    new_header = add_label_property_to_header(header_lines, vertex_props, vertex_prop_start_idx)
    new_vertex_props = list(vertex_props) if 'label' in vertex_props else list(vertex_props) + ['label']
    label_already_present = ('label' in vertex_props)

    # Read vertex lines
    if len(body_lines) < vertex_count:
        raise ValueError(f"Vertex count {vertex_count} exceeds available lines {len(body_lines)} in {in_path}.")
    vertex_lines = body_lines[:vertex_count]
    rest_lines = body_lines[vertex_count:]
    
    residue_floats = []
    for i, ln in enumerate(vertex_lines):
        toks = ln.split()
        if len(toks) < len(vertex_props):
            raise ValueError(f"Vertex line {i+1} has {len(toks)} values but header defines {len(vertex_props)} props.")
        try:
            residue_floats.append(float(toks[res_idx]))
        except Exception:
            raise ValueError(f"Invalid residue_id '{toks[res_idx]}' on vertex line {i+1} in {in_path}")

    labels_arr, bad_rows, zbased = build_vertex_labels(residue_floats, label_str, zero_based)
    for l in labels_arr:
        if l < 0:
            raise ValueError("arr < 0")
    if len(bad_rows) > 0:
        raise RuntimeError(
            f"[LABEL-OUT-OF-RANGE] {id_hint or in_path} total_bad={len(bad_rows)} "
            f"rows(example,0-based)={bad_rows[:10].tolist()} zero_based={zbased}"
        )

    new_vertex_lines = []
    for i, ln in enumerate(vertex_lines):
        toks = ln.split()
        if len(toks) < len(vertex_props):
            raise ValueError(f"Vertex line {i+1} has {len(toks)} values but header defines {len(vertex_props)} props.")
        # Get residue_id; it's commonly float like 140.0
        """
        try:
            rid_float = float(toks[res_idx])
        except Exception:
            raise ValueError(f"Invalid residue_id '{toks[res_idx]}' on vertex line {i+1} in {in_path}")
        rid = int(round(rid_float))

        if not zero_based:
            rid_index = rid - 1
        else:
            rid_index = rid

        if 0 <= rid_index < len(label_str):
            ch = label_str[rid_index]
            try:
                lab_val = float(ch)
            except Exception:
                # Fallback: map non-numeric to 0.0
                lab_val = float(out_of_range_value)
        else:
            lab_val = float(out_of_range_value)
            """
        lab_val = float(labels_arr[i])

        if label_already_present:
            # Replace existing last value if we assume label is last; safer approach: append anyway
            toks.append(f"{lab_val:.6f}")
        else:
            toks.append(f"{lab_val:.6f}")
        new_vertex_lines.append(" ".join(toks))

    # Write output
    with open(out_path, 'w', encoding='utf-8') as out:
        for ln in new_header:
            out.write(ln + "\n")
        for ln in new_vertex_lines:
            out.write(ln + "\n")
        for ln in rest_lines:
            out.write(ln + "\n")

def main():
    ap = argparse.ArgumentParser(description="Append residue-level labels to PLY vertex data.")
    ap.add_argument("--txt", required=True, help="Path to labels TXT (triplets of >ID, SEQ, LABELS).")
    ap.add_argument("--ply_dir", required=True, help="Directory containing .ply files to process.")
    ap.add_argument("--out_dir", default=None, help="Output directory. Defaults to <ply_dir>/_labeled.")
    ap.add_argument("--in_place", action="store_true", help="Overwrite original PLY files (use with caution).")
    ap.add_argument("--id_regex", default=None, help="Regex with one capture group to extract protein ID from filename stem.")
    ap.add_argument("--id_split", default="_", help="Delimiter to split filename stem if no regex is given (default: '_').")
    ap.add_argument("--id_part", type=int, default=0, help="Index to pick after splitting filename stem (default: 0).")
    ap.add_argument("--zero_based", action="store_true", help="Treat residue_id as 0-based index (default: 1-based).")
    ap.add_argument("--out_of_range_value", type=float, default=-1.0, help="Value used if residue_id index is out of bounds.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for .ply files.")
    ap.add_argument("--suffix", default="_labeled", help="Suffix for output filenames when not using --in_place.")
    args = ap.parse_args()

    labels_map = load_labels(args.txt)
    if not os.path.isdir(args.ply_dir):
        raise NotADirectoryError(f"Not a directory: {args.ply_dir}")

    out_dir = args.out_dir
    if args.in_place:
        out_dir = None
    else:
        if out_dir is None:
            out_dir = os.path.join(args.ply_dir, "_labeled")
        os.makedirs(out_dir, exist_ok=True)

    # Gather PLY files
    ply_files: List[str] = []
    if args.recursive:
        for root, _, files in os.walk(args.ply_dir):
            for fn in files:
                if fn.lower().endswith(".ply"):
                    ply_files.append(os.path.join(root, fn))
    else:
        ply_files = [os.path.join(args.ply_dir, fn) for fn in os.listdir(args.ply_dir) if fn.lower().endswith(".ply")]

    if not ply_files:
        print(f"[INFO] No .ply files found in {args.ply_dir}")
        return

    processed = 0
    skipped = 0

    for ply_path in ply_files:
        try:
            prot_id = extract_id_from_filename(ply_path, args.id_regex, args.id_split, args.id_part)
            if prot_id not in labels_map:
                sys.stderr.write(f"[WARN] ID '{prot_id}' not found in TXT. Skipping {os.path.basename(ply_path)}.\n")
                skipped += 1
                continue
            label_str = labels_map[prot_id]

            if args.in_place:
                out_path = ply_path
            else:
                base = os.path.basename(ply_path)
                stem, ext = os.path.splitext(base)
                out_path = os.path.join(out_dir, f"{stem}{args.suffix}{ext}")

            process_ply_ascii(
                in_path=ply_path,
                out_path=out_path,
                label_str=label_str,
                zero_based=args.zero_based,
                out_of_range_value=args.out_of_range_value,
                id_hint=prot_id
            )
            processed += 1
            print(f"[OK] {os.path.basename(ply_path)} -> {os.path.basename(out_path)} (ID: {prot_id})")
        except NotImplementedError as nie:
            sys.stderr.write(f"[SKIP] {os.path.basename(ply_path)}: {nie}\n")
            skipped += 1
        except Exception as e:
            sys.stderr.write(f"[ERROR] {os.path.basename(ply_path)}: {e}\n")
            skipped += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total: {len(ply_files)}")

if __name__ == "__main__":
    main()
