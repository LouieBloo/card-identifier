# classification/cnn_classifier.py
import torch
import json
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import config

class CNNClassifier:
    def __init__(self):
        """Initializes the EfficientNet model, transforms, and class mappings."""
        with open(config.CLASS_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        self.idx_to_class = mapping['idx_to_class']
        num_classes = len(self.idx_to_class)

        self.model = EfficientNet.from_name('efficientnet-b0')
        self.model._fc = torch.nn.Linear(in_features=1280, out_features=num_classes)
        self.model.load_state_dict(torch.load(config.CNN_CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()
        print("CNN Classifier (EfficientNet) loaded successfully.")

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_card(self, card_image: Image.Image):
        """Classifies a cropped card image and returns top predictions."""
        card_tensor = self.preprocess(card_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(card_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, config.CNN_TOP_K)

        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()

        if not isinstance(top_probs, list):
            top_probs = [top_probs]
            top_indices = [top_indices]

        guesses = []
        for i in range(config.CNN_TOP_K):
            idx = str(top_indices[i])
            oracle_id = self.idx_to_class.get(idx)
            if oracle_id:
                guesses.append({
                    "confidence": top_probs[i],
                    "predicted_id": oracle_id  # This is an oracle_id
                })
        return guesses