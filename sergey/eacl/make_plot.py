#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
import pandas as pd

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# pip install pillow
from pathlib import Path
from math import ceil, sqrt
from PIL import Image

def make_plots_grid(
    outdir: Path,
    pattern: str = "plot_*.png",
    cols: int = None,
    thumb_size: tuple[int, int] = (900, 600),  # w,h for each tile
    pad: int = 20,                             # padding around and between tiles
    bg: tuple[int, int, int] = (255, 255, 255),
    out_png: str = "all_plots_grid.png",
    out_pdf: str = "all_plots_grid.pdf",
) -> tuple[Path, Path]:
    """
    Make a single large grid image (and PDF) from all plot_*.png files in outdir.
    Returns paths to the PNG and PDF (or None if no images).
    """
    outdir = Path(outdir)
    imgs = sorted(outdir.glob(pattern))
    if not imgs:
        print(f"No images matching {pattern} in {outdir}")
        return None, None

    # choose columns if not provided
    n = len(imgs)
    if cols is None:
        cols = max(1, int(ceil(sqrt(n))))
    rows = int(ceil(n / cols))

    # normalize images to a consistent thumbnail size
    W, H = thumb_size
    thumbs = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        # Fit inside WxH while keeping aspect, then pad to exact size
        im.thumbnail((W, H))
        canvas = Image.new("RGB", (W, H), bg)
        x = (W - im.width) // 2
        y = (H - im.height) // 2
        canvas.paste(im, (x, y))
        thumbs.append(canvas)

    # canvas size
    total_w = cols * W + (cols + 1) * pad
    total_h = rows * H + (rows + 1) * pad
    grid = Image.new("RGB", (total_w, total_h), bg)

    # paste thumbnails
    for idx, tile in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (W + pad)
        y0 = pad + r * (H + pad)
        grid.paste(tile, (x0, y0))

    png_path = outdir / out_png
    grid.save(png_path)
    print(f"Saved grid PNG → {png_path}")

    pdf_path = outdir / out_pdf
    grid.save(pdf_path, "PDF")
    print(f"Saved grid PDF → {pdf_path}")

    return png_path, pdf_path


_STEP_RE = re.compile(r"^checkpoint_(\d+)(?:_(\d+))?\.pt\.collapse$")

def plot_metric_vs_step(csv_path: Path,
                        metric_col: Optional[str] = None,
                        save_to: Optional[Path] = None):
    """
    Read a combined metrics CSV, extract 'step' from checkpoint_dir, and plot step vs metric.

    Parameters
    ----------
    csv_path : str | Path
        Path to the combined CSV (like combined__train.dec_emb.stats.csv).
    metric_col : str | None
        Column name of the averaged metric to plot. If None, will auto-pick the
        first numeric metric column (excluding metadata).
    save_to : str | Path | None
        If provided, saves the plot to this path; otherwise shows the plot.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Extract step from checkpoint_dir (checkpoint_X[_Y].pt.collapse)
    def _extract_step(name: str) -> int:
        m = _STEP_RE.match(name)
        if not m:
            return None
        group = int(m.group(1))
        step = int(m.group(2)) if m.group(2) is not None else 0
        return step

    df["step"] = df["checkpoint_dir"].apply(_extract_step)

    # Drop rows we couldn't parse
    df = df.dropna(subset=["step"]).copy()
    df["step"] = df["step"].astype(int)

    # Choose metric column if not given: first numeric column excluding metadata
    if metric_col is None:
        metadata = {"checkpoint_dir", "checkpoint_group", "checkpoint_step", "metric_file", "step"}
        numeric_cols = [c for c in df.select_dtypes("number").columns if c not in metadata]
        if not numeric_cols:
            raise ValueError("No numeric metric columns found to plot.")
        metric_col = numeric_cols[0]

    # Sort by step to keep plot clean and monotonic in x
    df = df.sort_values("step")

    # Plot
    plt.figure()
    plt.plot(df["step"], df[metric_col], marker="o")
    plt.xlabel("Checkpoint step")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs step")
    plt.tight_layout()

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to, dpi=150)
        print(f"Saved plot to: {save_to}")
        plt.close()
    else:
        plt.show()

    return df[["step", metric_col]].reset_index(drop=True)


# Which metric CSVs to look for inside each *.pt.collapse folder
METRIC_FILES = [
    "train.dec_emb.stats.csv",
    "train.dec_out.stats.csv",
    "train.enc_out.stats.csv",
]

# Regex: checkpoint_0.pt.collapse OR checkpoint_<num>_<step>.pt.collapse
CKPT_RE = re.compile(r"^checkpoint_(\d+)(?:_(\d+))?\.pt\.collapse$")

def sort_key(p: pathlib.Path):
    m = CKPT_RE.match(p.name)
    if not m:
        return (float("inf"), float("inf"))
    group = int(m.group(1))
    step = int(m.group(2)) if m.group(2) is not None else 0
    return (group, step)

def main():
    parser = argparse.ArgumentParser(
        description="Collect metric CSVs from *.pt.collapse subfolders, averaging metrics per checkpoint."
    )
    parser.add_argument("model_dir", type=pathlib.Path,
                        help="Path to model directory containing *.pt.collapse subfolders")
    parser.add_argument("--outdir", type=pathlib.Path,
                        default=pathlib.Path("./metrics_combined"),
                        help="Where to write combined CSVs (default: ./metrics_combined)")
    args = parser.parse_args()

    model_dir = Path(f"/projects/gus20642/sergey/models/{args.model_dir}")
    model_dir_name = Path(model_dir).name
    args.outdir = args.outdir / model_dir_name
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    collapse_dirs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.endswith(".pt.collapse")]
    collapse_dirs = [p for p in collapse_dirs if CKPT_RE.match(p.name)]
    if not collapse_dirs:
        print("No matching *.pt.collapse checkpoint folders found.")
        sys.exit(0)

    collapse_dirs.sort(key=sort_key)

    collected = {mf: [] for mf in METRIC_FILES}

    for cdir in collapse_dirs:
        m = CKPT_RE.match(cdir.name)
        group = int(m.group(1))
        step = int(m.group(2)) if m.group(2) is not None else 0
        ckpt_label = cdir.name

        for mf in METRIC_FILES:
            fpath = cdir / mf
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath, sep="\t")
            except Exception as e:
                print(f"WARNING: failed to read {fpath}: {e}", file=sys.stderr)
                continue

            # Average all numeric columns
            mean_vals = df.mean(numeric_only=True)

            # Turn into a single-row DataFrame
            row = mean_vals.to_frame().T
            row["checkpoint_dir"] = ckpt_label
            row["checkpoint_group"] = group
            row["checkpoint_step"] = step
            row["metric_file"] = mf

            collected[mf].append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)

    written_any = False
    for mf, parts in collected.items():
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        out_csv = args.outdir / f"combined__{mf}"
        combined.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}   (rows={len(combined)})")
        written_any = True

    if not written_any:
        print("No metric CSVs found to combine (after filtering).")

    def numerical_columns(csv_path: Path):
        df = pd.read_csv(csv_path)
        metadata = {"checkpoint_dir", "checkpoint_group", "checkpoint_step", "metric_file", "step"}
        return [c for c in df.select_dtypes("number").columns if c not in metadata]

    for csv_file in args.outdir.iterdir():
        if not csv_file.name.startswith("combined__") or not csv_file.name.endswith(".csv"):
            continue
        name = csv_file.name[len("combined__"):-len(".csv")]
        print(f"Plotting metrics from {csv_file.name}...")
        for metric_column in numerical_columns(csv_file):
            plot_metric_vs_step(
                csv_path=csv_file,
                metric_col=metric_column,
                save_to=args.outdir / f"plot_{name}_{metric_column}.png"
            )
    # collect all plots in one html
    make_plots_grid(args.outdir)
    

if __name__ == "__main__":
    main()
