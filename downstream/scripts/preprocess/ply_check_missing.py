#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read protein_id from a triplet-format txt:
>protein_id
SEQUENCE
LABELS
Check whether {protein_id}_A.ply exists in the given folder and write missing ids to a txt.

Usage:
python scripts/preprocess/ply_check_missing.py --txt labels.txt --ply-dir /path/to/plys --out missing_ids.txt
Options:
--suffix "_A.ply"   # customize filename suffix
--case-insensitive  # case-insensitive filename match
"""

from pathlib import Path
import argparse

def parse_id_blocks(txt_path: Path):
    """Parse triplet blocks and return an ordered, de-duplicated id list."""
    ids_in_order = []
    seen = set()

    # Support UTF-8 with BOM.
    with txt_path.open("r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith(">"):
            raise ValueError(f"Line {i+1} should start with '>': {lines[i]!r}")
        protein_id = lines[i][1:].strip().split()[0]  # strip '>' and take first token
        # Read next two lines: sequence and labels
        if i + 2 >= n:
            raise ValueError(f"Incomplete record at id={protein_id} (missing seq/label lines)")
        # lines[i+1] is sequence, lines[i+2] is labels; not used here
        i += 3

        if protein_id not in seen:
            seen.add(protein_id)
            ids_in_order.append(protein_id)

    return ids_in_order


def find_expected_ply(ply_dir: Path, protein_id: str, suffix: str, case_insensitive: bool):
    """Find {protein_id}{suffix} in ply_dir, return Path or None."""
    expected_name = f"{protein_id}{suffix}"
    p = ply_dir / expected_name
    if p.exists():
        return p

    if case_insensitive:
        target = expected_name.casefold()
        for fp in ply_dir.iterdir():
            if fp.is_file() and fp.name.casefold() == target:
                return fp
    return None


def main():
    ap = argparse.ArgumentParser(description="Check missing {protein_id}_A.ply files and output missing ids.")
    ap.add_argument("--txt", required=True, type=Path, help="Triplet label txt path")
    ap.add_argument("--ply-dir", required=True, type=Path, help="PLY directory")
    ap.add_argument("--out", type=Path, default=Path("missing_ids.txt"), help="Missing ids output txt")
    ap.add_argument("--suffix", default="_A.ply", help="PLY filename suffix (default: _A.ply)")
    ap.add_argument("--case-insensitive", action="store_true", help="Case-insensitive filename match")
    args = ap.parse_args()

    if not args.txt.exists():
        raise SystemExit(f"[ERROR] Label txt not found: {args.txt}")
    if not args.ply_dir.exists() or not args.ply_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid PLY directory: {args.ply_dir}")

    protein_ids = parse_id_blocks(args.txt)

    missing = []
    found_cnt = 0
    for pid in protein_ids:
        fp = find_expected_ply(args.ply_dir, pid, args.suffix, args.case_insensitive)
        if fp is None:
            missing.append(pid)
        else:
            found_cnt += 1

    # Write missing ids
    if missing:
        args.out.write_text("\n".join(missing) + "\n", encoding="utf-8")
    else:
        # Write empty file when none are missing.
        args.out.write_text("", encoding="utf-8")

    print(f"Total ids: {len(protein_ids)}")
    print(f"Found: {found_cnt}")
    print(f"Missing: {len(missing)}")
    print(f"Missing ids written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
