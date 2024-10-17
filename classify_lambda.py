import torch
import numpy as np
import cv2
import json
import boto3
import os
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from multipart import MultipartDecoder

yolov5_model_name="best.pt"
classifier_model_name="magic_card_classifier_v3.pth"

# Load the class-to-index mapping from the JSON file
with open('class_to_idx.json', 'r') as f:
    mapping = json.load(f)

class_to_idx = mapping['class_to_idx']
idx_to_class = mapping['idx_to_class']
num_classes = len(idx_to_class)

# S3 client for fetching models
s3 = boto3.client('s3')

# Download YOLOv5 and EfficientNet models from S3 to /tmp
def download_models():
    # Assuming you have already uploaded your models to S3
    bucket_name = 'card-classifier'
    
    # YOLOv5 model
    yolo_model_path = f'/tmp/{yolov5_model_name}'
    s3.download_file(bucket_name, yolov5_model_name, yolo_model_path)
    
    # EfficientNet model
    classifier_model_path = f'/tmp/{classifier_model_name}'
    s3.download_file(bucket_name, classifier_model_name, classifier_model_path)
    
    return yolo_model_path, classifier_model_path

# Load models
def load_models(yolo_path, efficientnet_path):
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
    classification_model = EfficientNet.from_name('efficientnet-b0')
    classification_model._fc = torch.nn.Linear(in_features=1280, out_features=num_classes)
    classification_model.load_state_dict(torch.load(efficientnet_path))
    classification_model.eval()
    
    return yolov5_model, classification_model

def parse_multipart_formdata(event):
    content_type = event['headers']['Content-Type']
    body = event['body']

    # # MultipartDecoder requires binary data, so we decode it
    # if event["isBase64Encoded"]:
    #     body = base64.b64decode(body)

    # Use MultipartDecoder to parse form data
    decoder = MultipartDecoder(body, content_type)
    form_data = {}
    for part in decoder.parts:
        content_disposition = part.headers.get(b"Content-Disposition", b"").decode()
        if 'filename' in content_disposition:
            form_data['file'] = part.content  # This is the binary image file
        else:
            name = content_disposition.split('name="')[1].split('"')[0]
            form_data[name] = part.content.decode()  # This is other form data like 'x' and 'y'

    return form_data

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

# Lambda handler function
def lambda_handler(event, context):
    # Download models on the first execution
    yolo_path, classifier_path = download_models()
    
    # Load the models
    yolov5_model, classification_model = load_models(yolo_path, classifier_path)
    
    # Parse the form data from multipart/form-data
    form_data = parse_multipart_formdata(event)

    # Extract the image and click coordinates from the form data
    image_data = form_data['file']
    click_x = float(form_data['x'])
    click_y = float(form_data['y'])

    # Process the image
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    image = np.array(image)

    # YOLOv5 detection
    results = yolov5_model(image)
    detections = results.xyxy[0].cpu().numpy()

    # Step 3: Print number of detected objects
    num_detections = detections.shape[0]  # Get the number of detections (rows in detections array)
    print(f"Number of detected objects: {num_detections}")

    # Find the closest box to the click
    closest_box = find_closest_box(detections, click_x * width, click_y * height)
    if closest_box is None:
        return {"error": "No card detected near click location"}

    # Extract the card image based on the detected bounding box
    x1, y1, x2, y2, _, _ = closest_box
    card_image = image[int(y1):int(y2), int(x1):int(x2)]
    card_image = cv2.cvtColor(card_image, cv2.COLOR_RGB2BGR)
    card_image = cv2.resize(card_image, (224, 224))

    # Preprocess the card image for EfficientNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    card_image_tensor = preprocess(card_image).unsqueeze(0)

    # Classification with EfficientNet
    with torch.no_grad():
        output = classification_model(card_image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_class = predicted_class.item()
    predicted_scryfall_id = idx_to_class.get(str(predicted_class), None)

    # Fetch details from Scryfall
    scryfall_data = fetch_card_details_by_id(predicted_scryfall_id)

    return {
        "statusCode": 200,
        "body": {
            "detected_card": predicted_scryfall_id,
            "classification_confidence": confidence.item(),
            "scryfall_data": scryfall_data
        }
    }
