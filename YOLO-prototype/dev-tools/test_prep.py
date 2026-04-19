import os
import cv2
import random
from pathlib import Path
import shutil
from dotenv import load_dotenv

load_dotenv()

# ================= SETTINGS =================
RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data")
TEST_OUTPUT_DIR = r"test_results\test_prep_results"
TILE_SIZE = 640

# Dynamic overlap settings to handle class imbalance.
CLASS_OVERLAP = {
    "Wheat-dataset": 0.1,         # Majority class (~3886 images)
    "Drybean-dataset": 0.1,       # Majority class (~3290 images)
    "Canola dataset": 0.5,        # Minority class (~1297 images)
    "Lentil-Wheat dataset": 0.75  # Extreme minority class (~365 images)
}
DEFAULT_OVERLAP = 0.2
# ==============================================

def process_and_save_single_image(img_path, output_dir, overlap):
    """Cuts a single image and saves tiles to demonstrate the overlap effect."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        return 0

    h, w, _ = img.shape
    step = int(TILE_SIZE * (1 - overlap))
    step = max(1, step)
    base_name = img_path.stem

    y_coords = list(range(0, max(1, h - TILE_SIZE + 1), step))
    if h > TILE_SIZE and (not y_coords or y_coords[-1] != h - TILE_SIZE):
        y_coords.append(h - TILE_SIZE)

    x_coords = list(range(0, max(1, w - TILE_SIZE + 1), step))
    if w > TILE_SIZE and (not x_coords or x_coords[-1] != w - TILE_SIZE):
        x_coords.append(w - TILE_SIZE)

    tile_count = 0
    for y in y_coords:
        for x in x_coords:
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                continue

            tile_filename = output_dir / f"test_{base_name}_tile_{y}_{x}.jpg"
            cv2.imwrite(str(tile_filename), tile)
            tile_count += 1
            
    return tile_count

def main():
    raw_path = Path(RAW_DATA_DIR)
    out_path = Path(TEST_OUTPUT_DIR)
    
    if not raw_path.exists():
        print(f"Error: Directory does not exist: {RAW_DATA_DIR}")
        return

    classes = [d for d in raw_path.iterdir() if d.is_dir()]
    
    if not classes:
        print(f"No classes found in {RAW_DATA_DIR}.")
        return

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    for cls_dir in classes:
        overlap = CLASS_OVERLAP.get(cls_dir.name, DEFAULT_OVERLAP)
        
        images = list(cls_dir.rglob('*.jpg')) + list(cls_dir.rglob('*.png')) + list(cls_dir.rglob('*.tif'))
        
        if not images:
            print(f"Skipping {cls_dir.name} - no images found.")
            continue
            
        random_img = random.choice(images)
        date_identifier = random_img.parent.name
        
        class_out_dir = out_path / cls_dir.name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        tiles_generated = process_and_save_single_image(random_img, class_out_dir, overlap)
        
        print(f"\nClass   : {cls_dir.name}")
        print(f"Image   : {random_img.name} (from {date_identifier})")
        print(f"Overlap : {overlap * 100}%")
        print(f"Result  : Generated {tiles_generated} tiles")

    print("\n[DONE] Check the 'test_prep_results' folder to inspect the generated tiles.")

if __name__ == "__main__":
    main()