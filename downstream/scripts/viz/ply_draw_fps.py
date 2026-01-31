# draw_ply_fps.py
import argparse, sys, numpy as np
import matplotlib.pyplot as plt

def load_ply_vertices(path):
    # 1) try trimesh
    try:
        import trimesh
        m = trimesh.load(path, process=False)
        if hasattr(m, "vertices"):
            return np.asarray(m.vertices, dtype=np.float64)
        if hasattr(m, "points"):
            return np.asarray(m.points, dtype=np.float64)
    except Exception:
        pass
    # 2) try plyfile
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"].data
        return np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    except Exception:
        pass
    # 3) minimal ASCII PLY fallback
    with open(path, "r") as f:
        if f.readline().strip() != "ply":
            raise RuntimeError("Not a PLY file.")
        fmt = f.readline().strip()
        if "ascii" not in fmt:
            raise RuntimeError("Binary PLY detected. Please install trimesh/plyfile or convert to ASCII.")
        n_vert, props = None, []
        line = f.readline()
        while line and "end_header" not in line:
            if line.startswith("element vertex"):
                n_vert = int(line.strip().split()[-1])
            if line.startswith("property"):
                parts = line.strip().split()
                # property <type> <name>
                if len(parts) >= 3:
                    props.append(parts[-1])
            line = f.readline()
        if n_vert is None:
            raise RuntimeError("Cannot find 'element vertex' in header.")
        xyz = []
        for _ in range(n_vert):
            vals = f.readline().strip().split()
            # x,y,z assumed to be the first three columns; use plyfile/trimesh if order differs
            xyz.append([float(vals[0]), float(vals[1]), float(vals[2])])
        return np.asarray(xyz, dtype=np.float64)

def fps_numpy(points, k, random_start=False, seed=0):
    """Farthest Point Sampling on Nx3 numpy array. O(Nk)."""
    N = points.shape[0]
    k = min(k, N)
    centers = np.empty(k, dtype=int)
    rng = np.random.default_rng(seed)
    if random_start:
        centers[0] = rng.integers(N)
    else:
        # start from farthest to centroid
        c = points.mean(axis=0)
        centers[0] = np.argmax(np.linalg.norm(points - c, axis=1))
    dists = np.full(N, np.inf)
    last = points[centers[0]]
    dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
    for i in range(1, k):
        centers[i] = int(np.argmax(dists))
        last = points[centers[i]]
        dists = np.minimum(dists, np.linalg.norm(points - last, axis=1))
    return centers

def set_equal_3d(ax, P, margin=0.05):
    mins = P.min(0); maxs = P.max(0)
    span = (maxs - mins).max()
    center = (mins + maxs) / 2.0
    half = span * (0.5 + margin)
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)

def plot_cloud(P, centers_idx=None, elev=20, azim=-60, out_path="out.png"):
    fig = plt.figure(figsize=(4,4), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:,0], P[:,1], P[:,2], s=1, depthshade=False)  # all points
    if centers_idx is not None:
        C = P[centers_idx]
        ax.scatter(C[:,0], C[:,1], C[:,2], s=18, depthshade=False)  # sampled centers
    set_equal_3d(ax, P)
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=str, help="path to .ply")
    ap.add_argument("--k", type=int, default=128, help="number of FPS centers")
    ap.add_argument("--max-points", type=int, default=200000,
                    help="optional random downsample for visualization speed")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--out-prefix", type=str, default="pc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    P = load_ply_vertices(args.ply)
    if P.shape[1] != 3:
        raise RuntimeError(f"Expect Nx3 vertices, got shape {P.shape}")
    # Random subsampling only for faster plotting (keep plots comparable)
    if P.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(P.shape[0], args.max_points, replace=False)
        P = P[idx]

    centers_idx = fps_numpy(P, args.k, random_start=False, seed=args.seed)

    plot_cloud(P, centers_idx=None, elev=args.elev, azim=args.azim,
               out_path=f"{args.out_prefix}_point_cloud.png")
    plot_cloud(P, centers_idx=centers_idx, elev=args.elev, azim=args.azim,
               out_path=f"{args.out_prefix}_centers.png")
    print("Saved:",
          f"{args.out_prefix}_point_cloud.png",
          f"{args.out_prefix}_centers.png")

if __name__ == "__main__":
    main()
