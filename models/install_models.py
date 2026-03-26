import os
from pathlib import Path
from ultralytics import YOLO

target_dir = "models\legacy"

model_path = Path(target_dir)
model_path.mkdir(parents=True, exist_ok=True)

model_names = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt","yolov8l.pt", "yolov8x.pt",
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt","yolo12l.pt", "yolo12x.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt","yolo26l.pt", "yolo26x.pt",
]

print(f"--- Starting Downloads to: {model_path.absolute()} ---")

# Ensure any temporary files created by Ultralytics are written inside models/legacy
original_cwd = os.getcwd()

for name in model_names:
    full_model_path = model_path / name

    if full_model_path.exists():
        print(f"[SKIP] {name} already exists in {target_dir}.")
        continue

    print(f"[DOWNLOADING] {name}...")
    try:
        # Change into the target directory so YOLO(name) writes files there
        os.chdir(model_path)
        model = YOLO(name)

        # If Ultralytics saved a local copy, move/rename it to the desired path
        local_file = model_path / name
        if local_file.exists():
            local_file.rename(full_model_path)
            print(f"[SUCCESS] Saved {name} to {target_dir}/")
        else:
            print(f"[WARNING] {name} was not found in {target_dir} after download.")
    finally:
        os.chdir(original_cwd)

print("\n--- All models are ready ---")
