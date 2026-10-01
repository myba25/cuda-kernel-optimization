#!/usr/bin/env python3
"""Line charts from the benchmark CSVs in results/<gpu>/.

One line per variant, x = problem size, y = a throughput column.

  python scripts/plot.py results/rtx_4050_laptop_gpu/bench_copy.csv
  python scripts/plot.py results/<gpu>/bench_gemm.csv --y gflops
  python scripts/plot.py results/<gpu>/bench_reduce.csv --variants r5,r6,cub

The PNG goes to results/<gpu>/plots/<csv name>.png unless --out is given.
If y is gbps and the CSV has a peak_gbps column, the theoretical peak is drawn as a reference line.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

# Categorical colors in a fixed order, checked for color-vision deficiency as neighbours.
# A variant's color comes from its position in the CSV, so --variants never repaints the others.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

LABELS = {
    "n": "Problem size N",
    "bytes": "Bytes moved per launch",
    "gbps": "Bandwidth (GB/s)",
    "gflops": "Throughput (GFLOPS)",
    "median_ms": "Median time (ms)",
    "pct_of_peak": "% of theoretical peak",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="benchmark CSV from results/<gpu>/")
    p.add_argument("--x", default="n", help="x column (default: n)")
    p.add_argument("--y", default="gbps", help="y column (default: gbps)")
    p.add_argument("--hue", default="variant", help="one line per value of this column (default: variant)")
    p.add_argument("--variants", help="comma-separated subset of variants to draw")
    p.add_argument("--linear-x", action="store_true", help="linear x axis instead of log2")
    p.add_argument("--title", help="chart title (default: '<kernel> · <gpu>')")
    p.add_argument("--out", type=Path, help="output PNG path")
    return p.parse_args()


def tick_label(value: float, column: str) -> str:
    """Binary units for byte columns, 2^k for powers of two, a plain number otherwise."""
    if column == "bytes":
        for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
            if value >= scale:
                return f"{value / scale:g} {unit}"
        return f"{value:g} B"
    if value > 0 and math.log2(value).is_integer():
        return f"$2^{{{int(math.log2(value))}}}$"
    return f"{value:,.0f}"


def label_line_ends(ax, ends: list[tuple[str, float, float]]) -> None:
    """Name each line at its right end, unless two ends are too close to label cleanly."""
    lo, hi = ax.get_ylim()
    ys = sorted(y for _, _, y in ends)
    if any((b - a) / (hi - lo) < 0.06 for a, b in zip(ys, ys[1:])):
        return  # converging lines: the legend carries identity instead
    for name, x, y in ends:
        ax.annotate(name, xy=(x, y), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", color=INK_2, annotation_clip=False)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    for col in (args.x, args.y, args.hue):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {args.csv} (columns: {', '.join(df.columns)})")
    if df.empty:
        sys.exit(f"{args.csv} has no rows")

    variants = list(dict.fromkeys(df[args.hue]))  # order of first appearance = bench order
    if len(variants) > len(SERIES):
        sys.exit(f"{len(variants)} variants but {len(SERIES)} colors: draw a subset with --variants")
    color = dict(zip(variants, SERIES))
    shown = args.variants.split(",") if args.variants else variants
    unknown = [v for v in shown if v not in color]
    if unknown:
        sys.exit(f"unknown variant(s): {', '.join(unknown)} (available: {', '.join(variants)})")
    data = df[df[args.hue].isin(shown)]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ends = []
    for v in shown:
        d = data[data[args.hue] == v].sort_values(args.x)
        ax.plot(d[args.x], d[args.y], label=v, color=color[v], lw=1.5,
                solid_capstyle="round", solid_joinstyle="round",
                marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        ends.append((v, d[args.x].iloc[-1], d[args.y].iloc[-1]))

    top = data[args.y].max()
    if args.y == "gbps" and "peak_gbps" in df.columns:
        peak = float(df["peak_gbps"].iloc[0])
        ax.axhline(peak, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
        # Label on the right: large sizes are DRAM-bound and sit below the peak there, while small
        # sizes on the left can exceed it because they fit in L2.
        ax.annotate(f"theoretical peak {peak:.0f} GB/s", xy=(1, peak), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", color=INK_2)
        top = max(top, peak)
    ax.set_ylim(0, top * 1.12)

    if not args.linear_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(sorted(data[args.x].unique())))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: tick_label(val, args.x)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}"))

    # Recessive chrome: hairline horizontal grid, one baseline, no box.
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, pad=6)
    ax.set_xlabel(LABELS.get(args.x, args.x), color=INK_2, labelpad=8)
    ax.set_ylabel(LABELS.get(args.y, args.y), color=INK_2, labelpad=8)

    kernel = str(df["kernel"].iloc[0]) if "kernel" in df.columns else args.csv.stem
    gpu = str(df["gpu"].iloc[0]) if "gpu" in df.columns else ""
    title = args.title or (f"{kernel} · {gpu}" if gpu else kernel)
    if len(shown) == 1 and not args.title and shown[0] != kernel:
        title += f" ({shown[0]})"  # one line: the title names it, no legend box
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)

    if len(shown) >= 2:
        ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
        if len(shown) <= 4:
            label_line_ends(ax, ends)

    out = args.out or args.csv.parent / "plots" / f"{args.csv.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
