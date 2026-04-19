import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("\n" + "="*50)
print(" 1. DATASET & ENVIRONMENT CHECKS")
print("="*50)

load_dotenv()
dataset_dir = os.getenv("TILED_DATA_DIR", "tiled_data")
abs_dataset_dir = Path(dataset_dir).absolute()

print(f"Target Dataset Path : {abs_dataset_dir}")

if not abs_dataset_dir.exists():
    print("[ERROR] Dataset folder DOES NOT exist. Did you run data_prep.py?")
else:
    print("[OK] Dataset folder found.")
    
    train_exists = (abs_dataset_dir / "train").exists()
    val_exists = (abs_dataset_dir / "val").exists()
    
    print(f"  -> 'train' folder : {'[OK] Found' if train_exists else '[ERROR] Missing'}")
    print(f"  -> 'val' folder   : {'[OK] Found' if val_exists else '[ERROR] Missing'}")

print("\n" + "="*50)
print(" 2. ULTRALYTICS (YOLO) CHECK")
print("="*50)

try:
    import ultralytics
    print(f"[OK] Ultralytics installed. Version: {ultralytics.__version__}")
except ImportError:
    print("[ERROR] Ultralytics is NOT installed. Run: pip install ultralytics")
    sys.exit(1)

print("\n" + "="*50)
print(" 3. PYTORCH & GPU (CUDA) CHECK")
print("="*50)

try:
    import torch
    print(f"[OK] PyTorch installed. Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        vram_bytes = torch.cuda.get_device_properties(current_device).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        
        print(f"[OK] GPU is AVAILABLE and ready for training!")
        print(f"  -> GPUs detected : {device_count}")
        print(f"  -> Active GPU    : {device_name}")
        print(f"  -> Total VRAM    : {vram_gb:.2f} GB")
        
        if vram_gb < 4.0:
            print("\n[WARNING] You have less than 4GB of VRAM. You might need to lower BATCH_SIZE to 8 or 4 in train.py.")
    else:
        print("\n[CRITICAL ERROR] PyTorch CANNOT see your GPU! It will fall back to CPU.")
        print("Possible reasons:")
        print("1. You don't have an NVIDIA GPU.")
        print("2. NVIDIA drivers are not installed or are outdated.")
        print("3. You installed the CPU-only version of PyTorch.")
        
except ImportError:
    print("[ERROR] PyTorch is NOT installed.")

print("\n" + "="*50)
print(" DIAGNOSTICS COMPLETE")
print("="*50 + "\n")