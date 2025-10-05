#!/usr/bin/env python3
"""
Overlay two experiments on the same plot (with legend labels).

Assumptions:
- You've already run your "combine metrics" script for each experiment dir.
- Combined CSVs live at: ./metrics_combined/<exp_dir_name>/combined__*.csv
- For each shared combined file, we plot all shared numeric metrics on a single
  axis: expA and expB as two lines with labels.

Example:
    python compare_two_experiments_overlay.py \
        /path/to/expA /path/to/expB \
        --outdir ./comparisons/expA_vs_expB_overlay
"""

import argparse
from pathlib import Path
import re
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil, sqrt

STEP_FROM_DIR_RE = re.compile(r"^checkpoint_(\d+)(?:_(\d+))?\.pt\.collapse$")
DEFAULT_METRICS_ROOT = Path("./metrics_combined")

def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _list_combined_csvs(exp_dir_name: str, metrics_root: Path) -> dict[str, Path]:
    exp_comb_dir = metrics_root / exp_dir_name
    out = {}
    if not exp_comb_dir.is_dir():
        return out
    for p in exp_comb_dir.iterdir():
        if p.is_file() and p.name.startswith("combined__") and p.suffix == ".csv":
            key = p.name[len("combined__"):]  # e.g., 'train.dec_out.stats.csv'
            out[key] = p
    return out

def _extract_step_from_checkpoint_dir(val: str) -> Optional[int]:
    if not isinstance(val, str):
        return None
    m = STEP_FROM_DIR_RE.match(val)
    if not m:
        return None
    return int(m.group(2)) if m.group(2) is not None else 0

def _prepare_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "checkpoint_step" in df.columns:
        step = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    else:
        step = pd.to_numeric(df.get("checkpoint_dir", "").apply(_extract_step_from_checkpoint_dir), errors="coerce")
    df = df.copy()
    df["step"] = step
    df = df.dropna(subset=["step"])
    df["step"] = df["step"].astype(int)
    df = df.sort_values("step")
    return df

def _numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    meta = {"checkpoint_dir", "checkpoint_group", "checkpoint_step", "metric_file", "step"}
    return [c for c in df.select_dtypes(include="number").columns if c not in meta]

def _nice_title(metric_file: str, metric_col: str) -> str:
    return f"{metric_file} • {metric_col}"

def map_name(name: str) -> str:
    """Map experiment dir names to nicer labels for plots."""
    mappings = {
        "00020_de_en_continuous_transformer": "ConMT + Dispersion",
        "00017_de_en_continuous_transformer": "ConMT (no reg.)",
        "00018_de_en_transformer_base": "Transformer-big fp16",
        "00019_de_en_transformer_base": "Transformer-big",
        "00048_de_en_transformer_base": "Transformer-base",
        "00025_de_en_transformer_base": "Transformer-big + R(E)",
    }
    return mappings.get(name, name)

def _plot_overlay(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    expA_name: str,
    expB_name: str,
    metric_col: str,
    title: str,
    save_to: Path,
    align_limits: bool = True,
):
    _safe_mkdir(save_to.parent)

    expA_name = map_name(expA_name)
    expB_name = map_name(expB_name)

    plt.figure(figsize=(7, 4))
    plt.plot(dfA["step"], dfA[metric_col], marker="o", label=expA_name)
    plt.plot(dfB["step"], dfB[metric_col], marker="o", label=expB_name)

    plt.xlabel("Checkpoint step")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    plt.grid()

    if align_limits:
        yA = pd.to_numeric(dfA[metric_col], errors="coerce")
        yB = pd.to_numeric(dfB[metric_col], errors="coerce")
        y = pd.concat([yA, yB]).dropna()
        if len(y):
            ymin, ymax = float(y.min()), float(y.max())
            pad = 1e-3 if ymax == 0 else abs(ymax) * 0.1
            ymin, ymax = ymin - pad, ymax + pad
            plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_to, dpi=300)
    plt.close()

def _make_grid(
    outdir: Path,
    pattern: str = "overlay__*.png",
    cols: Optional[int] = None,
    thumb_size: tuple[int, int] = (900, 600),
    pad: int = 20,
    bg: tuple[int, int, int] = (255, 255, 255),
    out_png: str = "all_overlay_plots.png",
    out_pdf: str = "all_overlay_plots.pdf",
):
    imgs = sorted(Path(outdir).glob(pattern))
    if not imgs:
        return

    n = len(imgs)
    if cols is None:
        cols = max(1, int(ceil(sqrt(n))))
    rows = int(ceil(n / cols))

    W, H = thumb_size
    thumbs = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        im.thumbnail((W, H))
        canvas = Image.new("RGB", (W, H), bg)
        x = (W - im.width) // 2
        y = (H - im.height) // 2
        canvas.paste(im, (x, y))
        thumbs.append(canvas)

    total_w = cols * W + (cols + 1) * pad
    total_h = rows * H + (rows + 1) * pad
    grid = Image.new("RGB", (total_w, total_h), bg)

    for idx, tile in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (W + pad)
        y0 = pad + r * (H + pad)
        grid.paste(tile, (x0, y0))

    png_path = Path(outdir) / out_png
    grid.save(png_path)
    pdf_path = Path(outdir) / out_pdf
    grid.save(pdf_path, "PDF")

def main():
    parser = argparse.ArgumentParser(description="Overlay two experiments on the same plot with legend labels.")
    parser.add_argument("expA", type=Path, help="Path to experiment A directory (original model dir).")
    parser.add_argument("expB", type=Path, help="Path to experiment B directory.")
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT,
                        help=f"Root where combined CSVs live (default: {DEFAULT_METRICS_ROOT})")
    parser.add_argument("--outdir", type=Path, default=Path("./comparisons_overlay"),
                        help="Where to save overlay plots.")
    parser.add_argument("--no-grid", action="store_true", help="Skip building a grid collage.")
    args = parser.parse_args()

    expA_name = args.expA.name
    expB_name = args.expB.name

    # Find combined CSVs
    combA = _list_combined_csvs(expA_name, args.metrics_root)
    combB = _list_combined_csvs(expB_name, args.metrics_root)
    if not combA:
        raise SystemExit(f"No combined CSVs for {expA_name} in {args.metrics_root / expA_name}")
    if not combB:
        raise SystemExit(f"No combined CSVs for {expB_name} in {args.metrics_root / expB_name}")

    shared_files = sorted(set(combA) & set(combB))
    if not shared_files:
        raise SystemExit("No overlapping combined CSVs between the two experiments.")

    outdir = args.outdir / f"{expA_name}__vs__{expB_name}"
    _safe_mkdir(outdir)

    produced = 0
    for metric_file in shared_files:
        dfA = _prepare_df(combA[metric_file])
        dfB = _prepare_df(combB[metric_file])

        colsA = set(_numeric_metric_columns(dfA))
        colsB = set(_numeric_metric_columns(dfB))
        common_metrics = sorted(colsA & colsB)
        if not common_metrics:
            print(f"[INFO] {metric_file}: no shared numeric columns; skipping.")
            continue

        base = metric_file.replace(".csv", "")
        for mcol in common_metrics:
            title = _nice_title(metric_file, mcol)
            save_path = outdir / f"overlay__{base}__{mcol}.png"
            _plot_overlay(dfA, dfB, expA_name, expB_name, mcol, title, save_path)
            print(f"Saved {save_path}")
            produced += 1

    if produced == 0:
        print("No overlay plots produced.")
    else:
        print(f"Produced {produced} overlay plots in {outdir}")

    if not args.no_grid and produced > 0:
        try:
            _make_grid(outdir)
            print(f"Built overlay grid(s) in {outdir}")
        except Exception as e:
            print(f"[WARN] Failed to build grid: {e}")

if __name__ == "__main__":
    main()
