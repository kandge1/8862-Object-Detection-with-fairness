import torch
import ultralytics
import boxmot

print("--- System Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:       {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")
else:
    print("WARNING: CUDA is not available. YOLO26 will run very slowly on CPU.")

print("\n--- Library Check ---")
print(f"Ultralytics (YOLO26): {ultralytics.__version__}")
print(f"BoxMOT Version:       {boxmot.__version__}")