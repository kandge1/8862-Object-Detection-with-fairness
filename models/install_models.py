import os
from pathlib import Path
from ultralytics import YOLO

target_dir = "models\legacy"

model_path = Path(target_dir)
model_path.mkdir(parents=True, exist_ok=True)

model_names = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt"
]

print(f"--- Starting Downloads to: {model_path.absolute()} ---")

for name in model_names:
    full_model_path = model_path / name
    
    if full_model_path.exists():
        print(f"[SKIP] {name} already exists in {target_dir}.")
    else:
        print(f"[DOWNLOADING] {name}...")
        model = YOLO(name)
        
        if os.path.exists(name):
            os.rename(name, full_model_path)
            print(f"[SUCCESS] Moved {name} to {target_dir}/")

print("\n--- All models are ready ---")
