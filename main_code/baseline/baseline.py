import csv
import os
import warnings

from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SIZES = ["Nano", "Small", "Medium", "Large", "X-Large"]

# Paths
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "legacy"))
OUT_DIR = os.path.join(BASE_DIR, "baseline_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "benchmark_results.csv")

# Weights / benchmarks
SERIES = ["YOLOv8", "YOLO12", "Yolo26"]
# For now, only evaluate the two MOT benchmarks
BENCHMARKS = ["MOT17", "MOT20"]
SIZE_MAP = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "X-Large",
}

# ROMA-style evaluation for pedestrian detection uses mAP.
# Ultralytics provides:
# - `results.box.map`   : mAP@0.5:0.95
# - `results.box.map50` : mAP@0.5
METRIC_KEYS = ["mAP"]

# Dataset configs: point to data.yaml files inside this repo
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

BENCHMARK_DATA_CONFIGS = {
    # ROMA uses MOT17Det/MOT20Det (pedestrian detection) metrics, so we must use the
    # detection YAMLs that point to YOLO-convertible labels.
    "MOT17": os.path.join(DATASETS_DIR, "mot17det.yaml"),
    "MOT20": os.path.join(DATASETS_DIR, "mot20det.yaml"),
}


def discover_models(models_dir: str):
    """Return list of model metadata dicts for all .pt files in models_dir."""
    models = []
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    for name in sorted(os.listdir(models_dir)):
        if not name.lower().endswith(".pt"):
            continue

        path = os.path.join(models_dir, name)
        stem = os.path.splitext(name)[0]  # e.g. "yolov8n"
        series = None
        size = None

        lower = stem.lower()
        if lower.startswith("yolov8"):
            series = "YOLOv8"
            size_code = lower.replace("yolov8", "")[:1]
            size = SIZE_MAP.get(size_code, size_code.upper())
        elif lower.startswith("yolo26"):
            series = "Yolo26"
            size_code = lower.replace("yolo26", "")[:1]
            size = SIZE_MAP.get(size_code, size_code.upper())
        else:
            series = stem
            size = ""

        models.append(
            {
                "name": stem,
                "series": series,
                "size": size,
                "path": path,
            }
        )

    if not models:
        raise RuntimeError(f"No .pt model files found in {models_dir}")

    return models


def ensure_csv_with_header(csv_path: str):
    """Create the CSV with header if it does not yet exist."""
    expected_header = ["benchmark", "model_name", "series", "size", *METRIC_KEYS]

    if os.path.exists(csv_path):
        # If the CSV header doesn't match our expected columns (e.g. after changing METRIC_KEYS),
        # rename the old file and regenerate with the new header.
        with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
            existing_header = next(csv.reader(f, delimiter=","), [])

        if existing_header == expected_header:
            return

        stem, ext = os.path.splitext(csv_path)
        backup_path = f"{stem}_old{ext}"
        os.replace(csv_path, backup_path)

    header = expected_header

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_result_row(csv_path: str, benchmark: str, model_meta: dict, metrics: dict):
    """Append a single benchmark/model result row to the CSV."""
    row = [
        benchmark,
        model_meta.get("name", ""),
        model_meta.get("series", ""),
        model_meta.get("size", ""),
    ]

    for key in METRIC_KEYS:
        row.append(metrics.get(key, None))

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def evaluate_model_on_benchmark(model_path: str, benchmark: str) -> dict:
    """Run the given model on the given benchmark and return metrics.

    Currently this uses Ultralytics YOLO's built-in `val()` to compute ROMA-style mAP.
    """
    data_cfg = BENCHMARK_DATA_CONFIGS.get(benchmark)
    if not data_cfg:
        raise ValueError(f"No dataset config (.yaml) defined for benchmark '{benchmark}'.")

    model = YOLO(model_path)
    results = model.val(data=data_cfg, verbose=False)

    # mAP@0.5 for ROMA-style evaluation.
    box_metrics = getattr(results, "box", None)
    map_value = float(getattr(box_metrics, "map50", getattr(box_metrics, "map", 0.0))) if box_metrics is not None else 0.0

    return {
        "mAP": map_value,
    }


def main():
    ensure_csv_with_header(CSV_PATH)

    models = discover_models(MODELS_DIR)
    print(f"Found {len(models)} models")

    for benchmark in BENCHMARKS:
        print(f"\n=== Benchmark: {benchmark} ===")

        for model_meta in models:
            model_name = model_meta["name"]
            model_path = model_meta["path"]

            print(f"\nRunning model '{model_name}' on {benchmark}...")

            metrics = evaluate_model_on_benchmark(model_path, benchmark)

            formatted_metrics = ", ".join(
                f"{k}={metrics.get(k):.4f}" if isinstance(metrics.get(k), (int, float)) else f"{k}={metrics.get(k)}"
                for k in METRIC_KEYS
            )
            print(f"Results: {formatted_metrics}")

            append_result_row(CSV_PATH, benchmark, model_meta, metrics)

    print(f"\nAll benchmarks complete. Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()