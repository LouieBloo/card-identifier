import torch
import numpy as np
from PIL import Image
import config

class YoloSlicer:
    def __init__(self):
        """Initializes and loads the YOLOv5 model."""
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=config.YOLO_MODEL_PATH)
        print("YOLOv5 model loaded successfully.")

    def _find_closest_box(self, detections, click_x, click_y):
        """Helper function to find the bounding box closest to a click."""
        closest_box = None
        min_distance = float('inf')
        for box in detections:
            x1, y1, x2, y2, _, _ = box
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            distance = np.sqrt((box_center_x - click_x) ** 2 + (box_center_y - click_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_box = box
        return closest_box

    def slice_card_from_image(self, full_image: Image.Image, x_coord: float, y_coord: float):
        """
        Detects cards in an image and returns the cropped PIL image and
        bounding box closest to the provided coordinates.
        """
        width, height = full_image.size
        # Convert PIL image to numpy array for model
        image_np = np.array(full_image)

        results = self.model(image_np)
        detections = results.xyxy[0].cpu().numpy()

        if detections.shape[0] == 0:
            return None, None

        # Convert normalized coordinates to absolute pixel coordinates
        abs_click_x = x_coord * width
        abs_click_y = y_coord * height

        closest_box = self._find_closest_box(detections, abs_click_x, abs_click_y)

        if closest_box is None:
            return None, None

        x1, y1, x2, y2, _, _ = closest_box
        
        # Crop the image using the original PIL image to preserve quality
        cropped_image = full_image.crop((int(x1), int(y1), int(x2), int(y2)))

        bounding_box = {
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2)
        }
        
        return cropped_image, bounding_box