import os
import logging
import numpy as np
import pandas as pd
import h5py
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, MMCIFParser
from tqdm import tqdm

# === Configure logging ===
logging.basicConfig(
    filename='ppi_extraction.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Amino acid 3-letter to 1-letter mapping ===
aa3_to_aa1 = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y'
}

# === Token ↔ ID mapping, including special tokens ===
token2id = {aa: i+1 for i, aa in enumerate(sorted(aa3_to_aa1.values()))}
token2id.update({'[PAD]':0, '[CLS]':21, '[SEP]':22, '[MASK]':23})
id2token = {v:k for k,v in token2id.items()}

# === Global PDB/CIF parsers ===
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

def fast_check_and_download(pdb_ids, pdb_dir, max_workers=10):
    """
    Ensure each PDB/CIF file is present locally; download missing ones in parallel.
    """
    os.makedirs(pdb_dir, exist_ok=True)
    def download_one(pdb_id):
        pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        cif_file = os.path.join(pdb_dir, f"{pdb_id}.cif")
        if os.path.exists(pdb_file) or os.path.exists(cif_file):
            logger.info(f"Exists: {pdb_id}")
            return
        for ext in ('pdb','cif'):
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{ext}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(os.path.join(pdb_dir, f"{pdb_id}.{ext}"), 'w') as out:
                        out.write(r.text)
                    logger.info(f"Downloaded {pdb_id}.{ext}")
                    return
            except Exception as e:
                logger.error(f"Download error for {pdb_id}.{ext}: {e}")
        logger.error(f"Failed to fetch both PDB and CIF for {pdb_id}")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        list(tqdm(exe.map(download_one, pdb_ids),
                  total=len(pdb_ids),
                  desc="Downloading structures"))

def process_pdb(pdb_id, pdb_dir, distance_threshold, min_adjacent_residues):
    """
    Parse structure, detect PPI interface residues between each chain pair,
    and build tokenized examples.
    Returns a list of tuples: (input_ids, attn_mask, interface_mask, chain_id, record_id)
    """
    # locate local file
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    cif_path = os.path.join(pdb_dir, f"{pdb_id}.cif")
    try:
        if os.path.exists(pdb_path):
            struct = pdb_parser.get_structure(pdb_id, pdb_path)[0]
        elif os.path.exists(cif_path):
            struct = cif_parser.get_structure(pdb_id, cif_path)[0]
        else:
            logger.warning(f"{pdb_id}: no file found")
            return []
    except Exception as e:
        logger.error(f"{pdb_id}: parse error {e}")
        return []

    chains = list(struct.get_chains())
    if len(chains) < 2:
        logger.info(f"{pdb_id}: only {len(chains)} chain(s), skip")
        return []

    results = []
    from itertools import combinations
    for c1, c2 in combinations(chains, 2):
        # extract atom coordinates (optionally only Cα to speed up)
        atoms1 = [a for a in c1.get_atoms() if a.get_id()=='CA']
        atoms2 = [a for a in c2.get_atoms() if a.get_id()=='CA']

        # build KD-trees and search neighbors
        tree2 = cKDTree([a.coord for a in atoms2])
        nbrs1 = tree2.query_ball_point([a.coord for a in atoms1], distance_threshold)
        inter1 = {atoms1[i].get_parent() for i, nb in enumerate(nbrs1) if nb}

        tree1 = cKDTree([a.coord for a in atoms1])
        nbrs2 = tree1.query_ball_point([a.coord for a in atoms2], distance_threshold)
        inter2 = {atoms2[i].get_parent() for i, nb in enumerate(nbrs2) if nb}

        total_hits = len(inter1) + len(inter2)
        if total_hits < min_adjacent_residues:
            continue

        # build residue lists and masks
        res1 = [r for r in c1 if r.get_id()[0]==' ']
        idx1 = {r.get_id()[1] for r in inter1}
        seq1 = [aa3_to_aa1.get(r.get_resname(), 'X') for r in res1]
        mask1 = [1 if r.get_id()[1] in idx1 else 0 for r in res1]

        res2 = [r for r in c2 if r.get_id()[0]==' ']
        idx2 = {r.get_id()[1] for r in inter2}
        seq2 = [aa3_to_aa1.get(r.get_resname(), 'X') for r in res2]
        mask2 = [1 if r.get_id()[1] in idx2 else 0 for r in res2]

        # convert to token IDs
        ids1 = [token2id.get(aa, 0) for aa in seq1]
        ids2 = [token2id.get(aa, 0) for aa in seq2]

        # assemble final input: [CLS] seq1 [SEP] seq2 [SEP]
        inp = [token2id['[CLS]']] + ids1 + [token2id['[SEP]']] + ids2 + [token2id['[SEP]']]
        attn = [1] * len(inp)
        inter_mask = [0] + mask1 + [0] + mask2 + [0]
        chain_id = [0] * (len(seq1)+2) + [1] * (len(seq2)+1)

        rec_id = f"{pdb_id}_{c1.id}-{c2.id}"
        logger.info(f"Detected PPI {rec_id}: {total_hits} residues")
        results.append((
            np.array(inp, dtype=np.int32),
            np.array(attn, dtype=np.int32),
            np.array(inter_mask, dtype=np.int32),
            np.array(chain_id, dtype=np.int32),
            rec_id
        ))

    return results

def extract_from_csv_and_detect_ppi(csv_file, pdb_dir, output_h5,
                                    distance_threshold=5.0,
                                    min_adjacent_residues=10,
                                    max_workers=6,
                                    prealloc_chunk=1024):
    """
    Main driver: read CSV of PDB IDs, download structures, detect PPIs,
    and write to an HDF5 file with batch‐writes and deduplication.
    """
    logger.info(f"Start extraction: CSV={csv_file}, out={output_h5}")
    df = pd.read_csv(csv_file, skiprows=1)
    pdb_ids = df['PDB'].dropna().str.lower().unique()

    # download missing PDB/CIF files
    fast_check_and_download(pdb_ids, pdb_dir, max_workers=max_workers)

    # open HDF5 and create variable-length datasets
    with h5py.File(output_h5, 'w') as h5f:
        vlint = h5py.vlen_dtype(np.int32)
        vlstr = h5py.string_dtype()
        for name in ('input_ids','attn_mask','interface_mask','chain_id'):
            h5f.create_dataset(name, (0,), maxshape=(None,), dtype=vlint, chunks=True)
        h5f.create_dataset('ids', (0,), maxshape=(None,), dtype=vlstr, chunks=True)

        seen = set()  # to dedupe identical seq pairs

        # launch parallel PDB processing
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(process_pdb, pid, pdb_dir,
                                   distance_threshold, min_adjacent_residues): pid
                       for pid in pdb_ids}

            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc='Extracting PPIs'):
                pid = futures[fut]
                try:
                    res_list = fut.result()
                except Exception as e:
                    logger.error(f"{pid}: error {e}")
                    continue

                # filter out duplicates in this batch
                unique = []
                for inp, attn, imask, cid, rid in res_list:
                    key = tuple(inp.tolist())
                    if key not in seen:
                        seen.add(key)
                        unique.append((inp, attn, imask, cid, rid))
                if not unique:
                    continue

                n_new = len(unique)
                # batch‐write each field
                for ds_name, extractor in [
                    ('input_ids',     lambda t: t[0]),
                    ('attn_mask',     lambda t: t[1]),
                    ('interface_mask',lambda t: t[2]),
                    ('chain_id',      lambda t: t[3])
                ]:
                    ds = h5f[ds_name]
                    old = ds.shape[0]
                    # ds.resize((old + n_new,), axis=0)
                    ds.resize(old + n_new, axis=0)
                    ds[old:old+n_new] = [extractor(t) for t in unique]

                # write record IDs
                ds_ids = h5f['ids']
                old = ds_ids.shape[0]
                # ds_ids.resize((old + n_new,), axis=0)
                ds_ids.resize(old + n_new, axis=0)
                ds_ids[old:old+n_new] = [t[4] for t in unique]

    logger.info(f"Finished, saved to {output_h5}")
    print(f"✅ Saved to {output_h5}")

if __name__ == '__main__':
    extract_from_csv_and_detect_ppi(
        'pdb_chain_uniprot/pdb_chain_uniprot.csv',
        'pdb',
        'ppi_interface.h5',
        distance_threshold=5.0,
        min_adjacent_residues=8,
        max_workers=6
    )
