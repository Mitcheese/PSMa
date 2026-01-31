
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perturb regional red areas and randomly flip some gray to red in an ASCII PLY mesh with vertex colors.

- Detect connected components among *red-colored vertices* using face connectivity.
- Flip a fraction (or count) of those red components to gray to simulate "regional" decolorization.
- Randomly flip a fraction of gray vertices to red to simulate "random" noise.
- Preserves all vertex scalar properties and faces; only modifies the RGB triplets.
- Requires: numpy

Usage:
  python perturb_ply_colors.py input.ply -o output.ply \
      --flip-red-components-fraction 0.3 \
      --flip-grey-random-fraction 0.02 \
      --min-component-size 10 \
      --red 255 64 64 \
      --gray 180 180 180 \
      --seed 42 \
      [--mode random|largest|smallest]

Notes:
- Works for ASCII PLY with vertex RGB attributes named 'red','green','blue' (uchar).
- If your file lacks colors, you can generate colors from a float label named 'iface' (0/1) via --fallback-from-iface.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
import sys
import math
import random

def read_ply_ascii(path: Path):
    """Read an ASCII PLY with vertex floats + vertex uchar RGB + faces (list of 3 ints).
    Returns (header_lines, float_prop_names, uchar_prop_names, VxF floats, Vx3 colors, Fx3 faces).
    """
    header_lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            header_lines.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        # Parse header
        vertex_count = None
        face_count = None
        float_props: List[str] = []
        uchar_props: List[str] = []
        in_vertex_props = True
        for ln in header_lines:
            if ln.startswith("element vertex"):
                vertex_count = int(ln.split()[-1])
            elif ln.startswith("element face"):
                face_count = int(ln.split()[-1])
                in_vertex_props = False
            elif ln.startswith("property float"):
                # Only floats before colors assumed in vertex block
                if in_vertex_props:
                    float_props.append(ln.split()[-1])
            elif ln.startswith("property uchar"):
                if in_vertex_props:
                    uchar_props.append(ln.split()[-1])
        if vertex_count is None or face_count is None:
            raise ValueError("PLY missing element vertex/face header lines.")

        # Read vertices
        v_floats = np.zeros((vertex_count, len(float_props)), dtype=float)
        v_colors = np.zeros((vertex_count, len(uchar_props)), dtype=np.uint8) if uchar_props else None
        for i in range(vertex_count):
            parts = f.readline().split()
            if not parts:
                raise ValueError(f"Unexpected EOF while reading vertex {i}.")
            # First len(float_props) are floats; next len(uchar_props) are uchars
            if len(parts) < len(float_props) + len(uchar_props):
                raise ValueError(f"Vertex line {i} has {len(parts)} tokens, expected at least {len(float_props) + len(uchar_props)}")
            v_floats[i, :] = np.array(parts[:len(float_props)], dtype=float)
            if v_colors is not None:
                v_colors[i, :] = np.array(parts[len(float_props):len(float_props)+len(uchar_props)], dtype=np.uint8)

        # Read faces
        faces = np.zeros((face_count, 3), dtype=np.int32)
        for j in range(face_count):
            parts = f.readline().split()
            if not parts:
                raise ValueError(f"Unexpected EOF while reading face {j}.")
            n = int(parts[0])
            if n != 3:
                raise ValueError(f"Only triangular faces supported; found {n} vertices in face {j}.")
            if len(parts) < 4:
                raise ValueError(f"Face line {j} has too few tokens.")
            faces[j, :] = np.array(parts[1:4], dtype=np.int32)

    return header_lines, float_props, uchar_props, v_floats, v_colors, faces

def write_ply_ascii(path: Path, header_lines: List[str], float_props: List[str], uchar_props: List[str],
                    v_floats: np.ndarray, v_colors: np.ndarray, faces: np.ndarray):
    """Write back an ASCII PLY, preserving original header structure when possible.
    We will regenerate the header to avoid inconsistencies.
    """
    v_count = v_floats.shape[0]
    f_count = faces.shape[0]

    # Rebuild a clean header
    header = []
    header.append("ply")
    header.append("format ascii 1.0")
    header.append(f"element vertex {v_count}")
    for name in float_props:
        header.append(f"property float {name}")
    if v_colors is not None and v_colors.shape[1] == 3:
        header.append("property uchar red")
        header.append("property uchar green")
        header.append("property uchar blue")
    header.append(f"element face {f_count}")
    header.append("property list uchar int vertex_indices")
    header.append("end_header")

    with open(path, "w", encoding="utf-8") as f:
        for ln in header:
            f.write(ln + "\n")
        # Vertices
        for i in range(v_count):
            floats_str = " ".join(f"{x:.8f}" for x in v_floats[i])
            if v_colors is not None:
                r, g, b = v_colors[i].tolist()
                f.write(f"{floats_str} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{floats_str}\n")
        # Faces
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

def build_vertex_adjacency(faces: np.ndarray, v_count: int) -> List[List[int]]:
    """Build an undirected adjacency list for vertices from triangular faces."""
    adj = [[] for _ in range(v_count)]
    for i, j, k in faces:
        # add undirected edges
        for a, b in [(i, j), (j, k), (k, i)]:
            if b not in adj[a]:
                adj[a].append(b)
            if a not in adj[b]:
                adj[b].append(a)
    return adj

def connected_components_on_mask(adj: List[List[int]], mask: np.ndarray, min_size: int = 1) -> List[np.ndarray]:
    """Return a list of boolean masks per component (over vertices), restricted to vertices where mask==True.
    Components smaller than min_size are filtered out.
    """
    n = len(adj)
    visited = np.zeros(n, dtype=bool)
    comps = []
    for v in range(n):
        if not mask[v] or visited[v]:
            continue
        # BFS
        stack = [v]
        visited[v] = True
        members = [v]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if mask[w] and not visited[w]:
                    visited[w] = True
                    stack.append(w)
                    members.append(w)
        if len(members) >= min_size:
            comp_mask = np.zeros(n, dtype=bool)
            comp_mask[members] = True
            comps.append(comp_mask)
    return comps

def parse_rgb(vals: List[int]) -> Tuple[int, int, int]:
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("RGB must have exactly 3 integers.")
    for v in vals:
        if v < 0 or v > 255:
            raise argparse.ArgumentTypeError("RGB values must be in [0,255].")
    return (int(vals[0]), int(vals[1]), int(vals[2]))

def main():
    ap = argparse.ArgumentParser(description="Perturb red/gray areas in ASCII PLY vertex colors.")
    ap.add_argument("input", type=Path, help="Input ASCII PLY path")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output PLY path")
    ap.add_argument("--red", type=int, nargs=3, default=[255, 64, 64], help="RGB for red (default: 255 64 64)")
    ap.add_argument("--gray", type=int, nargs=3, default=[180, 180, 180], help="RGB for gray (default: 180 180 180)")
    ap.add_argument("--flip-red-components-fraction", type=float, default=0.3,
                    help="Fraction of RED connected components to flip to gray (0~1). Ignored if --flip-red-components-count is set.")
    ap.add_argument("--flip-red-components-count", type=int, default=None,
                    help="Exact number of RED components to flip to gray. Overrides fraction if provided.")
    ap.add_argument("--flip-grey-random-fraction", type=float, default=0.02,
                    help="Fraction of GRAY vertices to randomly flip to red (0~1).")
    ap.add_argument("--min-component-size", type=int, default=10,
                    help="Minimum component size (in vertices) to consider as a 'regional' red area.")
    ap.add_argument("--mode", choices=["random", "largest", "smallest"], default="random",
                    help="Which red components to flip: random, largest first, or smallest first.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument("--fallback-from-iface", action="store_true",
                    help="If vertex colors missing, generate colors from float property 'iface' (0->gray, >0.5->red).")

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    header_lines, float_props, uchar_props, v_floats, v_colors, faces = read_ply_ascii(args.input)

    if v_colors is None or v_colors.shape[1] != 3:
        if not args.fallback_from_iface:
            raise SystemExit("No vertex colors found and --fallback-from-iface not set. Aborting.")
        # Fallback: colorize from iface property
        if "iface" not in float_props:
            raise SystemExit("No 'iface' float property found for fallback.")
        iface_idx = float_props.index("iface")
        iface_vals = v_floats[:, iface_idx]
        v_colors = np.zeros((v_floats.shape[0], 3), dtype=np.uint8)
        red_rgb = np.array(args.red, dtype=np.uint8)
        gray_rgb = np.array(args.gray, dtype=np.uint8)
        v_colors[:] = gray_rgb
        v_colors[iface_vals > 0.5] = red_rgb

    red_rgb = np.array(args.red, dtype=np.uint8)
    gray_rgb = np.array(args.gray, dtype=np.uint8)

    # Masks
    is_red = np.all(v_colors == red_rgb, axis=1)
    is_gray = np.all(v_colors == gray_rgb, axis=1)

    # Build adjacency and connected components among red vertices
    adj = build_vertex_adjacency(faces, v_floats.shape[0])
    comps = connected_components_on_mask(adj, is_red, min_size=args.min_component_size)

    total_red_vertices = int(is_red.sum())
    total_gray_vertices = int(is_gray.sum())
    comp_sizes = [int(c.sum()) for c in comps]

    # Decide which components to flip
    if len(comps) > 0:
        indices = list(range(len(comps)))
        if args.mode == "largest":
            indices.sort(key=lambda i: comp_sizes[i], reverse=True)
        elif args.mode == "smallest":
            indices.sort(key=lambda i: comp_sizes[i])
        else:
            random.shuffle(indices)
        if args.flip_red_components_count is not None:
            k = max(0, min(len(comps), args.flip_red_components_count))
        else:
            k = int(round(args.flip_red_components_fraction * len(comps)))
        chosen = indices[:k]
    else:
        chosen = []

    # Apply flips: chosen red components -> gray
    if chosen:
        combined_mask = np.zeros(v_colors.shape[0], dtype=bool)
        for idx in chosen:
            combined_mask |= comps[idx]
        v_colors[combined_mask] = gray_rgb

    # Randomly flip some gray vertices -> red
    gray_indices = np.flatnonzero(np.all(v_colors == gray_rgb, axis=1))
    if len(gray_indices) > 0 and args.flip_grey_random_fraction > 0:
        n_flip = int(round(args.flip_grey_random_fraction * len(gray_indices)))
        if n_flip > 0:
            flip_idx = np.random.choice(gray_indices, size=n_flip, replace=False)
            v_colors[flip_idx] = red_rgb

    # Report
    new_is_red = np.all(v_colors == red_rgb, axis=1)
    new_is_gray = np.all(v_colors == gray_rgb, axis=1)
    print(f"Vertices: {v_floats.shape[0]}, Faces: {faces.shape[0]}")
    print(f"Original red vertices: {total_red_vertices}, gray vertices: {total_gray_vertices}, red components (>= {args.min_component_size}): {len(comps)}")
    print(f"Flipped {len(chosen)} red components to gray (mode={args.mode}).")
    print(f"Randomly flipped ~{args.flip_grey_random_fraction*100:.1f}% gray -> red: {int(new_is_red.sum()) - (total_red_vertices - sum(comp_sizes[i] for i in chosen))} verts")
    print(f"Now red vertices: {int(new_is_red.sum())}, gray vertices: {int(new_is_gray.sum())}")

    # Write output
    write_ply_ascii(args.output, float_props=float_props, uchar_props=["red","green","blue"],
                    header_lines=header_lines, v_floats=v_floats, v_colors=v_colors, faces=faces)

if __name__ == "__main__":
    main()
