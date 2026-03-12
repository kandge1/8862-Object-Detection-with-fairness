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
import subprocess
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Optimized Config for 4070 Ti / 13900HX
# ─────────────────────────────────────────────────────────────────────────────

SIZES = ["Nano", "Small", "Medium"]

# Look for all model weights inside the shared `models/legacy` folder
BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "legacy"))

WEIGHTS = {
    "Nano":   (os.path.join(MODELS_DIR, "yolov8n.pt"),  os.path.join(MODELS_DIR, "yolo26n.pt")),
    "Small":  (os.path.join(MODELS_DIR, "yolov8s.pt"),  os.path.join(MODELS_DIR, "yolo26s.pt")),
    "Medium": (os.path.join(MODELS_DIR, "yolov8m.pt"),  os.path.join(MODELS_DIR, "yolo26m.pt")),
}

SERIES       = ["YOLOv8", "YOLO26"]
BENCHMARKS   = ["MOT17", "MOT20", "COCO"]
METRIC_LABEL = {"MOT17": "HOTA", "MOT20": "HOTA", "COCO": "mAP@50-95"}

# Set this to "coco.yaml" for your final paper results
COCO_DATA_YAML = "coco.yaml" 

# Use higher workers to leverage the 13900HX performance cores
NUM_WORKERS = 12 

MOT_SOURCES = {
    "MOT17": "MOT17-ablation",
    "MOT20": "MOT20-ablation",
}

OUT_DIR  = os.path.join(BASE_DIR, "baseline_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "benchmark_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Optimized Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

def run_coco(weight: str, batch: int) -> float:
    """Uses Mixed Precision (FP16) and high worker count to speed up COCO."""
    print(f"    Evaluating COCO | {weight} | Batch: {batch}")
    try:
        model = YOLO(weight)
        # half=True uses FP16, crucial for 4070 Ti throughput
        metrics = model.val(
            data=COCO_DATA_YAML, 
            imgsz=640, 
            batch=batch, 
            half=True, 
            workers=NUM_WORKERS, 
            verbose=False,
            plots=False
        )
        return round(float(metrics.box.map), 4)
    except Exception as e:
        print(f"    COCO Error: {e}")
        return float("nan")

def run_mot(weight: str, dataset: str) -> float:
    """Runs BoxMOT evaluation with FP16 inference."""
    source_name = MOT_SOURCES.get(dataset)
    if not source_name:
        return float("nan")

    print(f"    Evaluating {dataset} | {weight}")
    weight_path = os.path.abspath(weight)
    
    # We use subprocess because BoxMOT/TrackEval often have global state conflicts
    cmd = [
        sys.executable, "-m", "boxmot.engine.cli", "eval",
        "--yolo-model", weight_path,
        "--tracking-method", "botsort",
        "--source", source_name,
        "--imgsz", "640",
        "--device", "0",
        "--half", # Speeds up inference on RTX 40-series
        "--ci"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        json_path = os.path.join(os.getcwd(), "botsort_output.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            # Handle BoxMOT nested JSON structure
            if "HOTA" in data: return data["HOTA"] / 100.0
            for k in data.values():
                if isinstance(k, dict) and "HOTA" in k: return k["HOTA"] / 100.0
        return float("nan")
    except Exception as e:
        print(f"    MOT Error: {e}")
        return float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic & Plotting
# ─────────────────────────────────────────────────────────────────────────────

def get_optimized_batch(size: str) -> int:
    """Adjusts batch size based on VRAM (12GB) and model complexity."""
    mapping = {"Nano": 128, "Small": 64, "Medium": 32}
    return mapping.get(size, 16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    results = {b: {s: {sz: float("nan") for sz in SIZES} for s in SERIES} for b in BENCHMARKS}

    if not args.plot_only:
        for size in SIZES:
            w_v8, w_v26 = WEIGHTS[size]
            batch = get_optimized_batch(size)
            
            # COCO Eval
            results["COCO"]["YOLOv8"][size] = run_coco(w_v8, batch)
            results["COCO"]["YOLO26"][size] = run_coco(w_v26, batch)
            
            # MOT Eval
            for bench in ["MOT17", "MOT20"]:
                results[bench]["YOLOv8"][size] = run_mot(w_v8, bench)
                results[bench]["YOLO26"][size] = run_mot(w_v26, bench)

        save_csv(results)
    else:
        results = load_csv()

    print_summary(results)
    plot_results(results)

def save_csv(results: dict) -> None:
    """Save results dict to a flat CSV for later plotting."""
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "series", "size", "metric"])
        for bench in BENCHMARKS:
            for series in SERIES:
                for size in SIZES:
                    metric = results.get(bench, {}).get(series, {}).get(size, float("nan"))
                    writer.writerow([bench, series, size, metric])


def load_csv() -> dict:
    """Load results back from CSV into the nested dict structure."""
    results = {b: {s: {sz: float("nan") for sz in SIZES} for s in SERIES} for b in BENCHMARKS}
    if not os.path.exists(CSV_PATH):
        return results

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = row["benchmark"]
            series = row["series"]
            size = row["size"]
            try:
                metric = float(row["metric"])
            except ValueError:
                metric = float("nan")
            if bench in results and series in results[bench] and size in results[bench][series]:
                results[bench][series][size] = metric
    return results


def print_summary(results: dict) -> None:
    """Pretty-print a small text table of results."""
    for bench in BENCHMARKS:
        print(f"\n=== {bench} ({METRIC_LABEL[bench]}) ===")
        header = "Size".ljust(8)
        for series in SERIES:
            header += f"{series:>12}"
        print(header)

        for size in SIZES:
            row = size.ljust(8)
            for series in SERIES:
                val = results[bench][series][size]
                row += f"{val:12.4f}" if not np.isnan(val) else f"{'nan':>12}"
            print(row)


def plot_results(results: dict) -> None:
    """Generate simple bar plots per benchmark and save as PNGs in OUT_DIR."""
    for bench in BENCHMARKS:
        x = np.arange(len(SIZES))
        width = 0.35

        v8_vals = [results[bench]["YOLOv8"][sz] for sz in SIZES]
        v26_vals = [results[bench]["YOLO26"][sz] for sz in SIZES]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, v8_vals, width, label="YOLOv8")
        ax.bar(x + width / 2, v26_vals, width, label="YOLO26")

        ax.set_title(f"{bench} – {METRIC_LABEL[bench]}")
        ax.set_xticks(x)
        ax.set_xticklabels(SIZES)
        ax.set_ylabel(METRIC_LABEL[bench])
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        out_path = os.path.join(OUT_DIR, f"{bench.lower()}_results.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

if __name__ == "__main__":
    main()