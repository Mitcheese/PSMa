#!/usr/bin/env python3
import os
import re
import argparse
import shutil

"""
python scripts/preprocess/ply_move_uniprot.py /path/to/src /path/to/dst
"""

# 6 alnum + '_' + 1 chain id (alnum) + .ply
PATTERN = re.compile(r'^[A-Za-z0-9]{6}_[A-Za-z0-9]\.ply$')

def move_matching_files(src_dir: str, dst_dir: str, dry_run: bool = False, overwrite: bool = False):
    os.makedirs(dst_dir, exist_ok=True)

    moved = 0
    skipped = 0

    with os.scandir(src_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith('.ply'):
                continue

            if PATTERN.match(name):
                src_path = entry.path
                dst_path = os.path.join(dst_dir, name)

                # Resolve name collisions: do not overwrite by default.
                if os.path.exists(dst_path) and not overwrite:
                    base, ext = os.path.splitext(name)
                    i = 1
                    while True:
                        alt_name = f"{base}__{i}{ext}"
                        alt_path = os.path.join(dst_dir, alt_name)
                        if not os.path.exists(alt_path):
                            dst_path = alt_path
                            break
                        i += 1

                print(f"[MOVE] {src_path} -> {dst_path}")
                if not dry_run:
                    shutil.move(src_path, dst_path)
                moved += 1
            else:
                skipped += 1

    print(f"Done. Moved {moved} files; skipped {skipped}.")

def main():
    parser = argparse.ArgumentParser(description="Move files like A0A0A0_A.ply to a target directory.")
    parser.add_argument("src", help="Source directory")
    parser.add_argument("dst", help="Destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Print moves without executing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if destination exists")
    args = parser.parse_args()

    move_matching_files(args.src, args.dst, dry_run=args.dry_run, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
