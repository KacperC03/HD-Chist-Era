import os
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.absolute()

DATASET_DIR = os.getenv("TILED_DATA_DIR", "tiled_data")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
MODEL_TYPE = "yolov8s-cls.pt" 
EPOCHS = 30                   
IMG_SIZE = 640                
BATCH_SIZE = 4                

def main():
    abs_dataset_dir = Path(DATASET_DIR).absolute()
    
    project_dir = BASE_DIR / "runs" / "classify" / "crop_classification"
    checkpoint_path = project_dir / EXPERIMENT_NAME / "weights" / "last.pt"

    if checkpoint_path.exists():
        print(f"\n[RESUME] Found checkpoint at {checkpoint_path}")
        print(f"Training resumes: {checkpoint_path}")
        
        model = YOLO(str(checkpoint_path))
        
        model.train(
            resume=True,
            project=str(project_dir)
        )
        
    else:
        print(f"\n[START] Checkpoint not found. Starting new training from scratch.")
        model = YOLO(MODEL_TYPE)
        model.train(
            data=str(abs_dataset_dir),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=str(project_dir),
            name=EXPERIMENT_NAME,
            device=0,
            workers=2
        )

if __name__ == '__main__':
    main()