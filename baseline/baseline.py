"""
YOLO Benchmark Evaluation Script
Compares YOLOv8 vs YOLO26 (Nano, Small, Medium) across:
  - MOT17  -> HOTA (lightweight tracking)
  - MOT20  -> HOTA (heavy/dense tracking)
  - COCO   -> mAP@50-95 (object detection)

Results are saved to baseline_outputs/benchmark_results.csv after evaluation.
To replot without re-running:  python baseline.py --plot-only

GPU utilization: Nano/Small/Medium models are lightweight and often underutilize
the GPU. Use --batch N (e.g. 16 or 32) for COCO validation to increase GPU load.
MOT evaluation is frame-by-frame so GPU saturation there is limited.

Requirements:
    pip install ultralytics boxmot matplotlib numpy

Laptop quick mode:
Set near the top of baseline.py:
COCO_DATA_YAML = "coco128.yaml" (already set).
MOT_SOURCES["MOT17"] = "./assets/MOT17-mini/train" (or your MOT17 subset path).
MOT_SOURCES["MOT20"] = None and MOT_ENABLED["MOT20"] = False to skip MOT20.

Server full benchmark mode:
Set:
COCO_DATA_YAML = "coco.yaml".
MOT_SOURCES = {"MOT17": "MOT17-ablation", "MOT20": "MOT20-ablation"}.
MOT_ENABLED = {"MOT17": True, "MOT20": True}.

"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SIZES = ["Nano", "Small", "Medium"]

BASE_DIR   = os.path.dirname(__file__)
# From `baseline/` we only need to go up one level to reach the repo root.
# Models live under `<repo_root>/models/legacy`.
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "legacy"))

# weights[size] = (yolov8_weight, yolo26_weight) — always load from models/legacy
WEIGHTS = {
    "Nano":   (os.path.join(MODELS_DIR, "yolov8n.pt"),  os.path.join(MODELS_DIR, "yolo26n.pt")),
    "Small":  (os.path.join(MODELS_DIR, "yolov8s.pt"),  os.path.join(MODELS_DIR, "yolo26s.pt")),
    "Medium": (os.path.join(MODELS_DIR, "yolov8m.pt"),  os.path.join(MODELS_DIR, "yolo26m.pt")),
}

SERIES       = ["YOLOv8", "YOLO26"]
BENCHMARKS   = ["MOT17", "MOT20", "COCO"]
METRIC_LABEL = {"MOT17": "HOTA", "MOT20": "HOTA", "COCO": "mAP@50-95"}


RUN_MOT = True  # Set False to skip MOT17/MOT20 (BoxMOT eval) entirely.

# Per‑benchmark MOT control. Values are BoxMOT --source strings.
# For quick local runs, you can point MOT17 to a small BoxMOT sample dataset
# (e.g. "./assets/MOT17-mini/train") and optionally disable MOT20.
MOT_SOURCES = {
    "MOT17": "MOT17-ablation",   # e.g. "./assets/MOT17-mini/train" for a tiny subset
    "MOT20": "MOT20-ablation",   # set to None to skip MOT20 on laptop
}

MOT_ENABLED = {
    "MOT17": True,
    "MOT20": True,
}

# COCO validation batch size. Larger = more GPU utilization (nano/small/medium
# models are light; try 32–64 or higher if you have VRAM to spare).
COCO_BATCH = 32

# COCO data config to use for evaluation.
# For full COCO on a server GPU, set this to "coco.yaml".
# For a much smaller subset suitable for quick local runs, use "coco128.yaml".
COCO_DATA_YAML = "coco.yaml"

OUT_DIR = os.path.join(BASE_DIR, "baseline_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUT_DIR, "benchmark_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_coco(weight: str, batch: int = 32) -> float:
    from ultralytics import YOLO
    print(f"    COCO  <- {weight}  (batch={batch})")
    try:
        model   = YOLO(weight)
        metrics = model.val(data=COCO_DATA_YAML, imgsz=640, batch=batch, verbose=True)
        return round(float(metrics.box.map), 4)
    except Exception as e:
        print(f"    ERROR: COCO eval failed for {weight}: {e}")
        return float("nan")


def run_mot(weight: str, dataset: str) -> float:
    """Run BoxMOT eval on MOT-style sources; returns HOTA (0–1 scale)."""
    if not RUN_MOT or not MOT_ENABLED.get(dataset, True):
        print(f"    {dataset} <- {weight} (skipped)")
        return float("nan")

    print(f"    {dataset} <- {weight}")
    # BoxMOT expects 'eval' (not 'track') for benchmark evaluation. Source can be
    # a named benchmark (e.g. MOT17-ablation / MOT20-ablation) or a custom path
    # like "./assets/MOT17-mini/train" for a much smaller subset.
    # --ci makes BoxMOT write results to botsort_output.json so we can read HOTA without capturing output.
    weight_path = os.path.abspath(weight)
    source_name = MOT_SOURCES.get(dataset) or f"{dataset}-ablation"
    cmd = [
        sys.executable,
        "-m",
        "boxmot.engine.cli",
        "eval",
        "--yolo-model",
        weight_path,
        "--tracking-method",
        "botsort",
        "--source",
        source_name,
        "--ci",
    ]
    # Stream stdout/stderr to terminal so user sees progress (download, generate, track, eval).
    proc = subprocess.run(cmd, timeout=3600)
    # BoxMOT writes botsort_output.json in the current working directory.
    # Use os.getcwd() so this works whether you run from the repo root or from baseline/.
    json_path = os.path.join(os.getcwd(), "botsort_output.json")
    hota = _parse_hota_from_json(json_path) if os.path.exists(json_path) else None

    if hota is None:
        print(f"    WARNING: could not get HOTA (missing or invalid {json_path})")
        return float("nan")

    return round(hota, 4)


def _parse_hota_from_json(json_path: str):
    """Read HOTA from BoxMOT --ci output (botsort_output.json). Returns 0–1 scale or None."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        # Single-class: data is {"HOTA": 69.42, "MOTA": ..., "per_sequence": {...}}
        if isinstance(data, dict) and "HOTA" in data:
            val = float(data["HOTA"])
            return val / 100.0 if val > 1.0 else val
        # Multi-class: data is {"pedestrian": {"HOTA": 69.42, ...}, ...}
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict) and "HOTA" in v:
                    val = float(v["HOTA"])
                    return val / 100.0 if val > 1.0 else val
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


def _parse_hota(text: str):
    """Extract HOTA from BoxMOT/TrackEval stdout/stderr (fallback)."""
    m = re.search(r"COMBINED\s*\([^)]+\)\s+([\d.]+)", text)
    if m:
        val = float(m.group(1))
        return val / 100.0 if val > 1.0 else val
    for pat in [r"HOTA\s*[:\(]?\s*([\d.]+)", r"'HOTA'\s*:\s*([\d.]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val
    return None


def run_all_evaluations(coco_batch: int = COCO_BATCH) -> dict:
    """
    Returns nested dict:
        results[benchmark][series][size] = score
    e.g. results["COCO"]["YOLOv8"]["Medium"] = 0.503
    """
    results = {
        bench: {series: {size: float("nan") for size in SIZES} for series in SERIES}
        for bench in BENCHMARKS
    }

    for size in SIZES:
        w_v8, w_v26 = WEIGHTS[size]
        print(f"\n{'─' * 52}")
        print(f"  Size: {size}  |  {w_v8}  vs  {w_v26}")
        print(f"{'─' * 52}")

        for bench in BENCHMARKS:
            if bench == "COCO":
                results[bench]["YOLOv8"][size] = run_coco(w_v8, batch=coco_batch)
                results[bench]["YOLO26"][size] = run_coco(w_v26, batch=coco_batch)
            elif RUN_MOT:
                results[bench]["YOLOv8"][size] = run_mot(w_v8, bench)
                results[bench]["YOLO26"][size] = run_mot(w_v26, bench)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# CSV save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: dict):
    """
    CSV format (long):
        Benchmark, Series, Size, Score
        MOT17, YOLOv8, Small, 0.6123
        MOT17, YOLO26, Small, 0.6401
        ...
    """
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Benchmark", "Series", "Size", "Score"])
        for bench in BENCHMARKS:
            for series in SERIES:
                for size in SIZES:
                    score = results[bench][series][size]
                    writer.writerow([bench, series, size, score])
    print(f"\n  Results saved -> {CSV_PATH}")


def load_csv() -> dict:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"'{CSV_PATH}' not found. Run without --plot-only first."
        )
    results = {
        bench: {series: {size: float("nan") for size in SIZES} for series in SERIES}
        for bench in BENCHMARKS
    }
    with open(CSV_PATH, "r") as f:
        for row in csv.DictReader(f):
            results[row["Benchmark"]][row["Series"]][row["Size"]] = float(row["Score"])
    print(f"  Results loaded from {CSV_PATH}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    col = 58
    print("\n" + "=" * col)
    print(f"  {'Benchmark':<8} {'Size':<8} {'YOLOv8':>10} {'YOLO26':>10} {'Delta':>10}")
    print("=" * col)
    for bench in BENCHMARKS:
        for size in SIZES:
            v8    = results[bench]["YOLOv8"][size]
            v26   = results[bench]["YOLO26"][size]
            delta = v26 - v8 if not (np.isnan(v8) or np.isnan(v26)) else float("nan")
            sign  = "+" if delta > 0 else ""
            print(f"  {bench:<8} {size:<8} {v8:>10.4f} {v26:>10.4f} {sign}{delta:>9.4f}")
        print("─" * col)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting — 3 subplots side by side
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {"YOLOv8": "#3A86FF", "YOLO26": "#FF6B6B"}

def plot_results(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor("#F8F9FA")

    for ax, bench in zip(axes, BENCHMARKS):
        _draw_subplot(ax, results[bench], bench)

    fig.suptitle(
        "YOLOv8 vs YOLO26  —  Nano / Small / Medium",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # Single shared legend underneath the subplots
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[s], label=s)
        for s in SERIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()

    out = os.path.join(OUT_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {out}")
    plt.show()


def _draw_subplot(ax, bench_data: dict, bench: str):
    """Draw one grouped bar chart for a single benchmark."""
    x     = np.arange(len(SIZES))
    width = 0.35

    ax.set_facecolor("#F8F9FA")
    ax.spines[["top", "right"]].set_visible(False)

    for i, series in enumerate(SERIES):
        color  = COLORS[series]
        values = [bench_data[series][size] for size in SIZES]
        offset = (i - 0.5) * width

        bars = ax.bar(
            x + offset, values, width,
            color=color, edgecolor="white", linewidth=0.8,
            zorder=3,
        )

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=11)
    ax.set_title(f"{bench}\n({METRIC_LABEL[bench]})",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    valid = [v for s in bench_data.values() for v in s.values()
             if not np.isnan(v)]
    if valid:
        ax.set_ylim(0, max(valid) * 1.20)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip evaluation, load from CSV and replot."
    )
    parser.add_argument(
        "--batch", type=int, default=COCO_BATCH,
        help=f"COCO validation batch size for better GPU utilization (default: {COCO_BATCH}). "
             "Increase (e.g. 64, 128) if you have VRAM to spare."
    )
    args = parser.parse_args()

    if args.plot_only:
        print("\n [--plot-only] Loading results from CSV...")
        results = load_csv()
    else:
        print(f"\n Running full benchmark evaluation (Nano / Small / Medium, COCO batch={args.batch})...")
        results = run_all_evaluations(coco_batch=args.batch)
        save_csv(results)

    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()