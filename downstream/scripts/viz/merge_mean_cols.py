#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge mean_label and mean_predict columns from all CSVs in a folder.
Usage example:
    python merge_mean_cols.py -i csvs -o merged_labels_preds.csv
    # To keep the source filename:
    python merge_mean_cols.py -i csvs -o merged_labels_preds.csv --include-filename
"""

import argparse
import csv
import glob
import os
import sys

def merge_mean_columns(input_dir: str, output_csv: str, include_filename: bool = False) -> None:
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not csv_files:
        print(f"[ERROR] No .csv files found under '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)

    written_rows = 0
    skipped_files = 0

    # Open output file
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        if include_filename:
            header = ["file_name", "mean_label", "mean_predict"]
        else:
            header = ["mean_label", "mean_predict"]

        writer = csv.writer(fout)
        writer.writerow(header)

        for fp in csv_files:
            try:
                with open(fp, "r", newline="", encoding="utf-8-sig") as fin:
                    reader = csv.reader(fin)
                    try:
                        cols = next(reader)
                    except StopIteration:
                        print(f"[WARN] Empty file skipped: {fp}.", file=sys.stderr)
                        continue

                    # Normalize column names for robust matching
                    lower_map = {c.strip().lower(): i for i, c in enumerate(cols)}
                    if "mean_label" not in lower_map or "mean_predict" not in lower_map:
                        print(f"[WARN] Missing required columns (mean_label/mean_predict): {fp}.", file=sys.stderr)
                        skipped_files += 1
                        continue

                    i_label = lower_map["mean_label"]
                    i_pred = lower_map["mean_predict"]

                    for row in reader:
                        # Row length check
                        if max(i_label, i_pred) >= len(row):
                            continue
                        v_label = row[i_label].strip()
                        v_pred = row[i_pred].strip()

                        # Optional: validate numeric values; skip invalid rows
                        try:
                            _ = float(v_label)
                            _ = float(v_pred)
                        except ValueError:
                            continue

                        if include_filename:
                            writer.writerow([os.path.basename(fp), v_label, v_pred])
                        else:
                            writer.writerow([v_label, v_pred])
                        written_rows += 1

            except Exception as e:
                print(f"[WARN] Failed to read file: {fp}. Reason: {e}", file=sys.stderr)
                skipped_files += 1

    print(f"[DONE] Wrote {written_rows} rows to '{output_csv}'. Skipped files: {skipped_files}")

def main():
    parser = argparse.ArgumentParser(description="Merge mean_label and mean_predict columns from CSVs.")
    parser.add_argument("-i", "--input-dir", default="csvs", help="Input CSV folder (default: csvs)")
    parser.add_argument("-o", "--output", default="merged_labels_preds.csv", help="Output CSV path (default: merged_labels_preds.csv)")
    parser.add_argument("--include-filename", action="store_true", help="Include source filename column")
    args = parser.parse_args()

    merge_mean_columns(args.input_dir, args.output, args.include_filename)

if __name__ == "__main__":
    main()
