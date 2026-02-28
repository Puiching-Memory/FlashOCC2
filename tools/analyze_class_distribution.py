#!/usr/bin/env python
"""Analyze class distribution of the NuScenes occupancy dataset and save as a chart.

Usage:
    python tools/analyze_class_distribution.py data/flashocc2-nuscenes_infos_train.pkl
    python tools/analyze_class_distribution.py data/flashocc2-nuscenes_infos_train.pkl \\
        --occ-root data/nuScenes/gts --output vis_output/class_distribution.png \\
        --max-samples 1000
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from flashocc.constants import OCC_CLASS_NAMES, NUM_OCC_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze class distribution of the NuScenes occupancy dataset")
    parser.add_argument("ann_file", help="Path to the data info pkl file")
    parser.add_argument("--occ-root", default=None,
                        help="OCC GT root directory (if not specified, reads from occ_path in info)")
    parser.add_argument("--output", default="vis_output/class_distribution.png",
                        help="Output image path (default: vis_output/class_distribution.png)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples (for quick preview, leave empty for all)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU cores)")
    parser.add_argument("--log-scale", action="store_true",
                        help="Use logarithmic scale")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the chart in a window")
    return parser.parse_args()


def load_infos(ann_file: str) -> list[dict]:
    """Load info pkl file."""
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Cannot parse ann_file format: {type(data)}")


def _count_single_sample(args: tuple) -> np.ndarray:
    """Single sample counting worker (for multiprocessing)."""
    idx, occ_path, occ_root = args
    counts = np.zeros(NUM_OCC_CLASSES, dtype=np.int64)

    # Try to locate labels.npz
    candidates = []
    if occ_root:
        candidates.append(os.path.join(occ_root, occ_path, "labels.npz"))
        candidates.append(os.path.join(occ_path, "labels.npz"))
    else:
        candidates.append(os.path.join(occ_path, "labels.npz"))

    for path in candidates:
        if os.path.isfile(path):
            try:
                data = np.load(path)
                semantics = data["semantics"]
                c = np.bincount(semantics.flatten().astype(np.int64),
                                minlength=NUM_OCC_CLASSES)
                counts[:len(c)] += c[:NUM_OCC_CLASSES]
                return counts
            except Exception:
                pass

    # Try to load voxel_semantics directly from occ_path
    for suffix in ["", ".npy", ".npz"]:
        full = occ_path + suffix if not occ_path.endswith(suffix) else occ_path
        if os.path.isfile(full):
            try:
                if full.endswith(".npz"):
                    data = np.load(full)
                    key = "semantics" if "semantics" in data else list(data.keys())[0]
                    semantics = data[key]
                else:
                    semantics = np.load(full)
                c = np.bincount(semantics.flatten().astype(np.int64),
                                minlength=NUM_OCC_CLASSES)
                counts[:len(c)] += c[:NUM_OCC_CLASSES]
                return counts
            except Exception:
                pass

    if idx < 3:
        print(f"Warning: OCC GT file not found for sample {idx}: {occ_path}")
    return counts


def count_class_voxels(infos: list[dict], occ_root: str | None = None,
                       max_samples: int | None = None,
                       num_workers: int | None = None) -> np.ndarray:
    """Count voxel numbers for each class (multiprocessing).

    Args:
        infos: Dataset info list.
        occ_root: OCC GT root directory.
        max_samples: Max number of samples.
        num_workers: Number of parallel workers, None for CPU cores.

    Returns:
        class_counts: (NUM_OCC_CLASSES,) — Voxel count per class.
    """
    from multiprocessing import Pool, cpu_count

    n = min(len(infos), max_samples) if max_samples else len(infos)
    tasks = [
        (i, infos[i].get("occ_path", ""), occ_root)
        for i in range(n)
    ]

    nw = num_workers if num_workers is not None else cpu_count()
    nw = min(nw, n)  # Cannot exceed total samples

    class_counts = np.zeros(NUM_OCC_CLASSES, dtype=np.int64)

    if nw <= 1:
        for t in tqdm(tasks, desc="Counting class distribution"):
            class_counts += _count_single_sample(t)
    else:
        print(f"Using {nw} processes for parallel counting...")
        with Pool(nw) as pool:
            for c in tqdm(pool.imap_unordered(_count_single_sample, tasks,
                                              chunksize=64),
                          total=n, desc="Counting class distribution"):
                class_counts += c

    return class_counts


def plot_distribution(class_counts: np.ndarray, output_path: str,
                      log_scale: bool = False, show: bool = True) -> None:
    """Plot class distribution and save it.

    Args:
        class_counts: (NUM_OCC_CLASSES,) — Total voxels per class.
        output_path: Output image path.
        log_scale: Whether to use log scale for y-axis.
        show: Whether to display a window.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    names = list(OCC_CLASS_NAMES)
    counts = class_counts.astype(np.float64)
    total = counts.sum()
    
    free_idx = names.index("free") if "free" in names else -1
    if free_idx != -1:
        free_cnt = counts[free_idx]
        free_pct = free_cnt / total * 100 if total > 0 else 0
        non_free_mask = np.ones(len(names), dtype=bool)
        non_free_mask[free_idx] = False
        
        counts_to_plot = counts[non_free_mask]
        names_to_plot = [names[i] for i in range(len(names)) if i != free_idx]
        non_free_total = counts_to_plot.sum()
        pct_to_plot = counts_to_plot / non_free_total * 100 if non_free_total > 0 else counts_to_plot
    else:
        counts_to_plot = counts
        names_to_plot = names
        non_free_total = total
        pct_to_plot = counts / total * 100 if total > 0 else counts
        free_cnt = 0
        free_pct = 0

    # Sort descending
    sort_idx = np.argsort(-counts_to_plot)
    sorted_names = [names_to_plot[i] for i in sort_idx]
    sorted_counts = counts_to_plot[sort_idx]
    sorted_pct = pct_to_plot[sort_idx]

    cmap = plt.cm.tab20
    colors = []
    for name in sorted_names:
        orig_idx = names.index(name)
        colors.append(cmap(orig_idx / len(names)))

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(sorted_names)), sorted_counts, color=colors,
                   edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.invert_yaxis()

    if log_scale:
        ax.set_xscale("log")
        ax.set_xlabel("Voxel Count (Log Scale)", fontsize=12)
    else:
        ax.set_xlabel("Voxel Count", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
            else (f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}")))

    ax.set_title("NuScenes OCC Dataset - Non-Free Class Distribution", fontsize=14, fontweight="bold")

    # Percentage string next to bars
    for bar, pct, cnt in zip(bars, sorted_pct, sorted_counts):
        width = bar.get_width()
        label = f" {pct:.2f}% ({cnt:.0f})"
        ax.text(width, bar.get_y() + bar.get_height() / 2, label,
                va="center", fontsize=9, color="#333333")

    # Text summary
    ax.text(0.98, 0.08, f"Total Non-Free: {non_free_total:,.0f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="gray", style="italic")
    if free_idx != -1:
        ax.text(0.98, 0.02, f"Total Free: {free_cnt:,.0f} ({free_pct:.2f}%)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10, color="gray", style="italic")

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved class distribution chart to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def print_table(class_counts: np.ndarray) -> None:
    """Print class distribution as a table."""
    total = class_counts.sum()
    names = list(OCC_CLASS_NAMES)
    
    free_idx = names.index("free") if "free" in names else -1
    free_cnt = class_counts[free_idx] if free_idx != -1 else 0
    non_free_total = total - free_cnt

    print(f"\n{'='*75}")
    print(f"{'Class':<25} {'Voxel Count':>15} {'% of Non-Free':>15} {'% of Total':>15}")
    print(f"{'-'*75}")
    
    sort_idx = np.argsort(-class_counts)
    for i in sort_idx:
        name = OCC_CLASS_NAMES[i]
        cnt = class_counts[i]
        pct_total = cnt / total * 100 if total > 0 else 0
        
        if i == free_idx:
            print(f"{name:<25} {cnt:>15,} {'-':>15} {pct_total:>14.2f}%")
        else:
            pct_non_free = cnt / non_free_total * 100 if non_free_total > 0 else 0
            print(f"{name:<25} {cnt:>15,} {pct_non_free:>14.2f}% {pct_total:>14.2f}%")
            
    print(f"{'-'*75}")
    print(f"{'Total Non-Free':<25} {non_free_total:>15,} {'100.00%':>15} {(non_free_total/total*100 if total > 0 else 0):>14.2f}%")
    print(f"{'Total All':<25} {total:>15,} {'-':>15} {'100.00%':>15}")
    print(f"{'='*75}\n")


def main():
    args = parse_args()

    print(f"Loading data: {args.ann_file}")
    infos = load_infos(args.ann_file)
    print(f"Number of samples: {len(infos)}")

    class_counts = count_class_voxels(
        infos, occ_root=args.occ_root, max_samples=args.max_samples,
        num_workers=args.num_workers)

    print_table(class_counts)

    plot_distribution(
        class_counts,
        output_path=args.output,
        log_scale=args.log_scale,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
