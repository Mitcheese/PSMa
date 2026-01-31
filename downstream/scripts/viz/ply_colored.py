import argparse
import os
import glob
import time
import numpy as np
from plyfile import PlyData, PlyElement

# Try to use tqdm; fall back to simple progress if unavailable.
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def colorize_and_save(ply_in, ply_out,
                      base=(180,180,180), hit=(255,64,64), thresh=0.5):
    """Color vertices by label and export. Returns: 'ok' | 'skip' | 'err'."""
    try:
        ply = PlyData.read(ply_in)
        v = ply['vertex'].data
        faces = ply['face'].data if 'face' in ply else None

        if 'label' not in v.dtype.names:
            return 'skip', "no label field"

        labels = np.asarray(v['label'], dtype=float)
        mask = labels > thresh

        r = np.full(len(v), base[0], dtype=np.uint8)
        g = np.full(len(v), base[1], dtype=np.uint8)
        b = np.full(len(v), base[2], dtype=np.uint8)
        r[mask], g[mask], b[mask] = hit

        names = v.dtype.names
        if all(c in names for c in ('red','green','blue')):
            new_v = v.copy()
            new_v['red']   = r
            new_v['green'] = g
            new_v['blue']  = b
        else:
            new_dtype = v.dtype.descr + [('red','u1'), ('green','u1'), ('blue','u1')]
            new_v = np.empty(len(v), dtype=new_dtype)
            for name in v.dtype.names:
                new_v[name] = v[name]
            new_v['red']   = r
            new_v['green'] = g
            new_v['blue']  = b

        vert_el = PlyElement.describe(new_v, 'vertex', comments=['with vertex colors'])
        elements = [vert_el]
        if faces is not None:
            elements.append(PlyElement.describe(faces, 'face'))

        os.makedirs(os.path.dirname(ply_out), exist_ok=True)
        PlyData(elements, text=True).write(ply_out)
        return 'ok', None
    except Exception as e:
        return 'err', str(e)

def batch_colorize(in_dir, out_dir, suffix="_color",
                   thresh=0.5, base=(180,180,180), hit=(255,64,64)):
    os.makedirs(out_dir, exist_ok=True)
    ply_list = sorted(glob.glob(os.path.join(in_dir, "*.ply")))
    total = len(ply_list)
    if total == 0:
        print(f"[WARN] no .ply in {in_dir}")
        return

    n_ok = n_skip = n_err = 0
    start = time.time()

    iterator = ply_list
    if tqdm is not None:
        iterator = tqdm(ply_list, total=total, unit="file", desc="Colorizing", dynamic_ncols=True)

    for i, pin in enumerate(iterator, 1):
        name = os.path.basename(pin)
        stem = os.path.splitext(name)[0]
        pout = os.path.join(out_dir, f"{stem}{suffix}.ply")

        status, info = colorize_and_save(pin, pout, base=base, hit=hit, thresh=thresh)
        if status == 'ok':
            n_ok += 1
            msg = f"[OK] {name}"
        elif status == 'skip':
            n_skip += 1
            msg = f"[SKIP] {name} ({info})"
        else:
            n_err += 1
            msg = f"[ERR] {name} ({info})"

        if tqdm is not None:
            iterator.set_postfix_str(f"ok:{n_ok} skip:{n_skip} err:{n_err}")
        else:
            # Simple progress display (no tqdm).
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            remain = total - i
            eta = remain / rate if rate > 0 else float('inf')
            print(f"\r[{i}/{total}] {msg} | ok:{n_ok} skip:{n_skip} err:{n_err} | ETA {eta:,.1f}s", end="", flush=True)

    if tqdm is None:
        print()  # newline

    elapsed = time.time() - start
    print(f"Done. total={total} ok={n_ok} skip={n_skip} err={n_err} | time={elapsed:,.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch colorize PLY files by label.")
    parser.add_argument("--input", required=True, help="Input directory with .ply files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--suffix", default="_color", help="Output filename suffix")
    parser.add_argument("--thresh", type=float, default=0.5, help="Label threshold")
    parser.add_argument("--base", type=int, nargs=3, default=(180, 180, 180), help="Base RGB color")
    parser.add_argument("--hit", type=int, nargs=3, default=(255, 64, 64), help="Hit RGB color")
    args = parser.parse_args()

    batch_colorize(
        args.input,
        args.output,
        suffix=args.suffix,
        thresh=args.thresh,
        base=tuple(args.base),
        hit=tuple(args.hit),
    )
