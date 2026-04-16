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


#To test the requirements, run the following command:
#python -c "import torch; print(torch.__version__); print('cuda?', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
#You should see:
#cuda? True
#count 1
#GPU name like NVIDIA GeForce RTX 4060