from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
import albumentations as A  # If you need any augmentation
from efficientnet_pytorch import EfficientNet  # Assuming you're using EfficientNet from this library
from typing import List, Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms
import json

yolo5ModelName='best.pt'
classifierModelName='magic_card_classifier_v3.pth'


# Load the class-to-index mapping from the JSON file
with open('class_to_idx.json', 'r') as f:
    mapping = json.load(f)

class_to_idx = mapping['class_to_idx']
idx_to_class = mapping['idx_to_class']
num_classes = len(idx_to_class) 

app = FastAPI()
# Allow all CORS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load your YOLOv5 detection model
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo5ModelName)

# Load your EfficientNet classification model
# classification_model = EfficientNet.from_name('efficientnet-b0')  # Change 'efficientnet-b0' to your variant if needed
# classification_model.load_state_dict(torch.load('efficientnet_card_classifier.pth'))
# classification_model.eval()

if True:
    # Load the EfficientNet model architecture
    classification_model = EfficientNet.from_name('efficientnet-b0')
    # Modify the final fully connected layer (_fc) to match the number of classes
    classification_model._fc = torch.nn.Linear(in_features=1280, out_features=num_classes)
    # Load the state dict from the saved model
    # classification_model.load_state_dict(torch.load(classifierModelName))
    classification_model.load_state_dict(torch.load(classifierModelName, map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    classification_model.eval()
    print(f"Model loaded successfully with {num_classes} output classes.")

# Scryfall endpoint
SCRYFALL_API_URL = 'https://api.scryfall.com/cards/named'

# Helper function to find closest detected box to user click
def find_closest_box(detections, click_x, click_y):
    closest_box = None
    min_distance = float('inf')
    for box in detections:
        x1, y1, x2, y2, conf, class_id = box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        distance = np.sqrt((box_center_x - click_x) ** 2 + (box_center_y - click_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_box = box
    return closest_box

# Helper function to fetch Scryfall card details
def fetch_card_details_by_id(oracle_id: str):
    response = requests.get(f"https://api.scryfall.com/cards/search?q=oracleid:{oracle_id}")
    if response.status_code == 200:
        data = response.json()
        if data['total_cards'] > 0:
            return data['data'][0]  # Return the first result
        else:
            return {"error": "No card found with this Oracle ID"}
    else:
        return {"error": "Failed to fetch card details"}

# Request model for click coordinates
class ClickLocation(BaseModel):
    x: int
    y: int

# FastAPI endpoint for detecting and classifying cards
@app.post("/classify")
async def classify_magic_card(file: UploadFile = File(...), x: float = Form(...), y: float = Form(...)):
    # Read image from the uploaded file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    image = np.array(image)
    

    # Step 1: Use YOLOv5 model to detect cards in the image
    results = yolov5_model(image)
    detections = results.xyxy[0].cpu().numpy()  # Extract detection results in format [x1, y1, x2, y2, confidence, class]

    # Step 3: Print number of detected objects
    num_detections = detections.shape[0]  # Get the number of detections (rows in detections array)
    print(f"Number of detected objects: {num_detections}")

    # Step 4: Loop through detections and print the coordinates of each detected object
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id = detection
        print(f"Object {i + 1}:")
        print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Confidence: {confidence}")
        print(f"  Class ID: {class_id}")

    # Step 2: Find the closest bounding box to the user's click location
    if x and y:
        closest_box = find_closest_box(detections, x*width, y*height)
        if closest_box is None:
            return {"error": "No card detected near click location"}
    else:
        return {"error": "Click location is required"}

    # Extract the image corresponding to the closest box
    x1, y1, x2, y2, _, _ = closest_box
    card_image = image[int(y1):int(y2), int(x1):int(x2)]

    # Ensure the image has 3 channels (RGB)
    if len(card_image.shape) == 2:  # Grayscale (1 channel), convert to 3-channel grayscale
        card_image = cv2.cvtColor(card_image, cv2.COLOR_GRAY2RGB)
    elif card_image.shape[2] == 4:  # If 4 channels (e.g., RGBA), convert to RGB
        card_image = cv2.cvtColor(card_image, cv2.COLOR_RGBA2RGB)


    cv2.imwrite('cropped_card_image.jpg', cv2.cvtColor(card_image, cv2.COLOR_RGB2BGR))  # Save as JPEG

    # Step 3: Preprocess the extracted card image and run it through your classification model
    # Resize and normalize the image for EfficientNet (assuming 224x224 input size)
    card_image = cv2.resize(card_image, (224, 224))  # Example size, adjust as necessary

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Apply preprocessing to the card image
    card_image_tensor = preprocess(card_image).unsqueeze(0)  # Add batch dimension
    #card_image_tensor = torch.from_numpy(card_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    # Run classification
    with torch.no_grad():
        output = classification_model(card_image_tensor)
        print(f"Model output: {output}")
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        print(f"Confidence: {confidence.item()}, Predicted class index: {predicted_class.item()}")
        predicted_class = predicted_class.item()
        

    print(f"predicted_class: {predicted_class}")
    predicted_scryfall_id = idx_to_class.get(str(predicted_class), None)
    print(f"predicted_scryfall_id: {predicted_scryfall_id}")

    if predicted_scryfall_id is None:
        return {"error": "Predicted class index not mapped to any Scryfall ID"}

    # Step 4: Fetch card details from Scryfall
    scryfall_data = fetch_card_details_by_id(predicted_scryfall_id)

    # Return the result
    return {
        "detected_card": predicted_scryfall_id,
        "classification_confidence": confidence.item(),
        "scryfall_data": scryfall_data
    }

@app.get("/")
async def read_root():
    return {"message": "App is running"}
