import os
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================= SETTINGS =================
BASE_DIR = Path(__file__).parent.absolute()
DATASET_DIR = os.getenv("TILED_DATA_DIR", "tiled_data")

# Directory where reports will be saved
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ==============================================

def main():
    abs_dataset_dir = Path(DATASET_DIR).absolute()
    test_dir = abs_dataset_dir / "test"
    
    # Path to the BEST weights from your training
    model_path = BASE_DIR / "runs" / "classify" / "crop_classification" / "yolo_model_v1" / "weights" / "best.pt"
    
    # 1. Verify files exist
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Make sure the training completed at least one epoch and the 'weights' folder exists.")
        return
        
    if not test_dir.exists():
        print(f"[ERROR] Test folder does not exist: {test_dir}")
        return

    # 2. Load the model
    print(f"\nLoading the trained model from: {model_path.name}...")
    model = YOLO(str(model_path)) 
    
    class_names = model.names
    
    print("\nStarting evaluation on the test set. This may take a few minutes...\n")
    
    # Variables for tracking statistics
    total_images = 0
    correct_predictions = 0
    class_stats = {cls_name: {"total": 0, "correct": 0} for cls_name in class_names.values()}

    # 3. Main testing loop
    for true_class_folder in test_dir.iterdir():
        if not true_class_folder.is_dir():
            continue
            
        true_class_name = true_class_folder.name
        images = list(true_class_folder.glob("*.jpg")) + list(true_class_folder.glob("*.png")) + list(true_class_folder.glob("*.tif"))
        
        for img_path in images:
            total_images += 1
            class_stats[true_class_name]["total"] += 1
            
            results = model.predict(source=str(img_path), device=0, verbose=False)
            result = results[0] 
            
            predicted_class_id = result.probs.top1
            predicted_class_name = class_names[predicted_class_id]
            
            if predicted_class_name == true_class_name:
                correct_predictions += 1
                class_stats[true_class_name]["correct"] += 1

    # 4. Prepare the final report
    report_lines = []
    report_lines.append("="*50)
    report_lines.append(" TEST SET EVALUATION RESULTS")
    report_lines.append("="*50)
    
    if total_images == 0:
        print("[WARNING] No images found in the test folder!")
        return
        
    accuracy = (correct_predictions / total_images) * 100
    report_lines.append(f"Overall Accuracy : {accuracy:.2f}% ({correct_predictions}/{total_images})\n")
    report_lines.append("Accuracy per class:")
    
    # Prepare data for CSV
    csv_data = [["Class Name", "Total Images", "Correct Predictions", "Accuracy (%)"]]
    
    for cls_name, stats in class_stats.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            report_lines.append(f" - {cls_name.ljust(25)}: {acc:>6.2f}% ({stats['correct']}/{stats['total']})")
            csv_data.append([cls_name, stats["total"], stats["correct"], round(acc, 2)])
        else:
            report_lines.append(f" - {cls_name.ljust(25)}: No test images found")
            csv_data.append([cls_name, 0, 0, "N/A"])
            
    # Add overall stats to CSV
    csv_data.append(["OVERALL", total_images, correct_predictions, round(accuracy, 2)])
    report_lines.append("\n" + "="*50)

    # Convert list of lines to a single string
    report_text = "\n".join(report_lines)
    
    # Print to console
    print(report_text)
    
    # 5. Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = RESULTS_DIR / f"report_{timestamp}.txt"
    csv_filename = RESULTS_DIR / f"metrics_{timestamp}.csv"
    
    # Save TXT
    with open(txt_filename, "w", encoding="utf-8") as txt_file:
        txt_file.write(report_text)
        
    # Save CSV
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
        
    print(f"\n[INFO] Reports successfully saved to:")
    print(f"  TXT: {txt_filename}")
    print(f"  CSV: {csv_filename}\n")

if __name__ == "__main__":
    main()