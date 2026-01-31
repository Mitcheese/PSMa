import argparse
import os

import requests

def download_pdb_files(uid_file, output_folder, failed_uid_file):
    # Ensure output directory exists.
    os.makedirs(output_folder, exist_ok=True)
    
    # Read UID list.
    with open(uid_file, 'r', encoding='utf-8') as file:
        uids = [line.strip() for line in file if line.strip()]
    
    base_url = "https://files.rcsb.org/download/{uniprot_id}.pdb"
    failed_uids = []
    
    for uid in uids:
        url = base_url.format(uniprot_id=uid)
        output_path = os.path.join(output_folder, f"{uid}.pdb")
        
        if os.path.exists(output_path):
            print(f"{output_path} exists, skipping.")
            continue
        
        success = False
        for attempt in range(1):  # adjust retries here if needed
            print(f"Downloading {url} (attempt {attempt + 1}/1)...")
            try:
                response = requests.get(url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                    print(f"Downloaded {output_path}")
                    success = True
                    break
                else:
                    print(f"Download failed: {url} (status {response.status_code})")
            except requests.RequestException as e:
                print(f"Request error: {e}")
        
        if not success:
            failed_uids.append(uid)
    
    # Record failed UIDs.
    if failed_uids:
        with open(failed_uid_file, 'w', encoding='utf-8') as fail_file:
            for uid in failed_uids:
                fail_file.write(uid + '\n')
        print(f"{len(failed_uids)} UIDs failed, saved to {failed_uid_file}")
    else:
        print("All UIDs downloaded successfully.")


def parse_args():
    ap = argparse.ArgumentParser(description="Download PDB files by UID list.")
    ap.add_argument("--uids", required=True, help="Path to UID list text file")
    ap.add_argument("--out-dir", required=True, help="Output directory for PDB files")
    ap.add_argument("--failed", required=True, help="Output file for failed UIDs")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_pdb_files(args.uids, args.out_dir, args.failed)

# https://files.rcsb.org/download/{uniprot_id}.pdb
# https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb
# https://files.rcsb.org/pub/pdb/compatible/pdb_bundle/{uniprot_id[1:3]}/{uniprot_id}/{uniprot_id}-pdb-bundle.tar.gz
