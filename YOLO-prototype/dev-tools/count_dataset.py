import os
from pathlib import Path

# ================= SETTINGS =================
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data")
# ==============================================

def count_images():
    base_path = Path(RAW_DATA_DIR)
    
    if not base_path.exists():
        print(f"Error: Directory does not exist: {RAW_DATA_DIR}")
        return

    classes = [d for d in base_path.iterdir() if d.is_dir()]
    total_dataset_images = 0

    print(f"\n{'='*50}")
    print(f"DATASET DISTRIBUTION SUMMARY")
    print(f"{'='*50}")

    for cls_dir in classes:
        print(f"\nClass: {cls_dir.name}")
        print(f"-"*40)
        
        date_folders = [d for d in cls_dir.iterdir() if d.is_dir()]
        class_total = 0
        
        for date_dir in date_folders:
            images = list(date_dir.glob('*.jpg')) + list(date_dir.glob('*.png')) + list(date_dir.glob('*.tif'))
            count = len(images)
            class_total += count
            
            print(f"  - {date_dir.name:<25} : {count} images")
            
        print(f"-"*40)
        print(f"  TOTAL for {cls_dir.name:<15} : {class_total} images")
        total_dataset_images += class_total

    print(f"\n{'='*50}")
    print(f"GRAND TOTAL (All classes)      : {total_dataset_images} images")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    count_images()