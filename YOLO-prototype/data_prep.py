import os
import cv2
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ================= SETTINGS =================
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data")
OUTPUT_DIR = os.getenv("TILED_DATA_DIR", "tiled_data")
TILE_SIZE = 640

# Splits for each date-folder:
VAL_RATIO = 0.1   # 10% for validation during training
TEST_RATIO = 0.1  # 10% for final evaluation (completely unseen)
# The remaining 80% goes to training automatically.

# Dynamic overlap to handle class imbalance
CLASS_OVERLAP = {
    "Wheat-dataset": 0.1,
    "Drybean-dataset": 0.1,
    "Canola dataset": 0.5,
    "Lentil-Wheat dataset": 0.75
}
DEFAULT_OVERLAP = 0.2
# ==============================================

def process_image(img_path, output_class_dir, date_identifier, overlap):
    """Cuts the image using smart tiling to handle 1098x798 resolution."""
    img = cv2.imread(str(img_path))
    if img is None:
        return

    h, w, _ = img.shape
    step = int(TILE_SIZE * (1 - overlap))
    step = max(1, step)
    base_name = img_path.stem

    # Generate smart coordinates to ensure edges are covered
    y_coords = list(range(0, max(1, h - TILE_SIZE + 1), step))
    if h > TILE_SIZE and (not y_coords or y_coords[-1] != h - TILE_SIZE):
        y_coords.append(h - TILE_SIZE)

    x_coords = list(range(0, max(1, w - TILE_SIZE + 1), step))
    if w > TILE_SIZE and (not x_coords or x_coords[-1] != w - TILE_SIZE):
        x_coords.append(w - TILE_SIZE)

    for y in y_coords:
        for x in x_coords:
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue

            tile_filename = output_class_dir / f"{date_identifier}_{base_name}_tile_{y}_{x}.jpg"
            cv2.imwrite(str(tile_filename), tile)

def main():
    raw_path = Path(RAW_DATA_DIR)
    out_path = Path(OUTPUT_DIR)
    
    classes = [d for d in raw_path.iterdir() if d.is_dir()]
    if not classes:
        print(f"No classes found in {RAW_DATA_DIR}.")
        return

    if out_path.exists():
        print(f"Czyszczenie istniejącego katalogu wyjściowego: {out_path}...")
        shutil.rmtree(out_path)

    for split in ['train', 'val', 'test']:
        for cls_dir in classes:
            (out_path / split / cls_dir.name).mkdir(parents=True, exist_ok=True)

    for cls_dir in classes:
        overlap = CLASS_OVERLAP.get(cls_dir.name, DEFAULT_OVERLAP)
        date_folders = [d for d in cls_dir.iterdir() if d.is_dir()]
        
        print(f"\nProcessing Class: {cls_dir.name} (Overlap: {overlap*100}%)")
        
        for date_dir in date_folders:
            images = list(date_dir.glob('*.jpg')) + list(date_dir.glob('*.png')) + list(date_dir.glob('*.tif'))
            if not images:
                continue
                
            date_identifier = date_dir.name 
            random.shuffle(images)
            
            # Stratified 3-way split
            total = len(images)
            val_count = int(total * VAL_RATIO)
            test_count = int(total * TEST_RATIO)
            
            val_images = images[:val_count]
            test_images = images[val_count:val_count + test_count]
            train_images = images[val_count + test_count:]
            
            # Logic for generating tiles
            splits = {'train': train_images, 'val': val_images, 'test': test_images}
            
            for split_name, img_list in splits.items():
                desc = f"    [{split_name.upper()}] {date_identifier}"
                for img_path in tqdm(img_list, desc=desc, leave=False):
                    process_image(img_path, out_path / split_name / cls_dir.name, date_identifier, overlap)

    print("\n[SUCCESS] Dataset prepared with Train, Val, and Test splits!")

if __name__ == "__main__":
    main()