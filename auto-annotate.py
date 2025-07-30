import torch
from pathlib import Path
import cv2 # OpenCV is used to get image dimensions
import yaml # Used to create the data.yaml file

# --- CONFIGURATION ---
# The confidence score threshold for saving a prediction.
# Detections below this score will be ignored.
CONF_THRESHOLD = 0.4 

# Path to your trained model weights
MODEL_PATH = 'docker_deploy/best.pt'

# Directory of your unlabeled images
IMAGE_DIR = Path('/mnt/e/Photos/TableStream/cardsNeedingAnnotations')

# Directory where the new .txt label files will be saved
LABEL_DIR = Path('/mnt/e/Photos/TableStream/cardsNeedingAnnotationsTestLabels')
# --- END CONFIGURATION ---


# Ensure the output directory exists
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Load your trained model
print("Loading model...")
# Using trust_repo=True is often necessary for custom models
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, trust_repo=True)
print("Model loaded successfully.")

# --- NEW: Generate data.yaml file ---
# This file tells Roboflow what the class indices (like 0, 1, 2) mean.
print("Generating data.yaml file...")
class_names = model.names
data_yaml_content = {
    'names': class_names,
    'nc': len(class_names)
}
yaml_save_path = LABEL_DIR / 'data.yaml'
with open(yaml_save_path, 'w') as f:
    yaml.dump(data_yaml_content, f, sort_keys=False)
print(f"Successfully created data.yaml at: {yaml_save_path}")
# --- END NEW SECTION ---

# Process each image in the directory
print(f"\nProcessing images in: {IMAGE_DIR}")
image_paths = list(IMAGE_DIR.glob('*.jpg')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpeg'))
print(f"Found {len(image_paths)} images to process.")

for image_path in image_paths:
    print(f"  -> Processing {image_path.name}")
    
    # --- MANUAL NORMALIZATION LOGIC ---
    # Load the image with OpenCV to get its exact dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"     [Warning] Could not read image: {image_path.name}. Skipping.")
        continue
    img_height, img_width, _ = img.shape

    # Run inference on the image
    results = model(str(image_path))
    
    # Get predictions with raw pixel coordinates (xmin, ymin, xmax, ymax)
    predictions = results.xyxy[0]

    # Define the full path for the output .txt file
    save_path = LABEL_DIR / f"{image_path.stem}.txt"

    # Open the file to write the filtered and manually normalized labels
    with open(save_path, 'w') as f:
        # Each 'pred' is a tensor: [xmin, ymin, xmax, ymax, confidence, class_id]
        for pred in predictions:
            confidence = pred[4]
            
            # Apply the confidence threshold
            if confidence >= CONF_THRESHOLD:
                # Extract raw pixel coordinates
                x_min, y_min, x_max, y_max = pred[0], pred[1], pred[2], pred[3]
                class_id = int(pred[5])

                # Manually calculate YOLO's normalized format
                box_width = x_max - x_min
                box_height = y_max - y_min
                x_center = x_min + box_width / 2
                y_center = y_min + box_height / 2

                # Normalize the coordinates by the image dimensions
                norm_x = x_center / img_width
                norm_y = y_center / img_height
                norm_w = box_width / img_width
                norm_h = box_height / img_height
                
                # Write the correct YOLO format line to the file
                # <class_index> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
                f.write(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

print("\n✅ Finished generating labels and data.yaml!")
print(f"Labels saved to: {LABEL_DIR}")
print("\nIMPORTANT: When uploading to Roboflow, make sure to include the new 'data.yaml' file along with your images and .txt files.")
