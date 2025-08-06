import torch
import json
from PIL import Image
from torchvision import transforms
import timm
import config
from collections import OrderedDict

class CNNClassifierV2:
    def __init__(self):
        """Initializes the EfficientNetV2 model, transforms, and class mappings."""
        # Load the class mapping file to know the number of classes
        with open(config.CLASS_MAPPING_PATH_V2, 'r') as f:
            mapping = json.load(f)
        self.idx_to_class = mapping['idx_to_class']
        num_classes = len(self.idx_to_class)

        # --- Model Initialization ---
        model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        # --- FIX for torch.compile() Mismatch ---
        # Load the checkpoint first into a temporary variable
        # Ensure config.CNN_CLASSIFIER_MODEL_PATH points to your new v2 model
        checkpoint = torch.load(config.CNN_CLASSIFIER_MODEL_PATH_V2, map_location=torch.device('cpu'))

        # The state_dict may be nested inside the checkpoint if you saved the whole dictionary
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Create a new state_dict without the `_orig_mod.` prefix from torch.compile
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                name = k[10:] # remove `_orig_mod.`
            else:
                name = k
            new_state_dict[name] = v
        
        # Load the corrected state_dict
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print(f"CNN Classifier ({model_name}) loaded successfully.")

        # --- Preprocessing Update ---
        # Use the exact same validation transforms from your training script for consistency.
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify_card(self, card_image: Image.Image):
        """Classifies a cropped card image and returns top predictions."""
        # Preprocess the image and add a batch dimension
        card_tensor = self.preprocess(card_image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(card_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)
            # Get the top K predictions
            top_probs, top_indices = torch.topk(probabilities, config.CNN_TOP_K)

        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()

        # Handle the case where topk returns a single item (not in a list)
        if not isinstance(top_probs, list):
            top_probs = [top_probs]
            top_indices = [top_indices]

        guesses = []
        for i in range(len(top_probs)):
            idx = str(top_indices[i])
            oracle_id = self.idx_to_class.get(idx)
            if oracle_id:
                guesses.append({
                    "confidence": top_probs[i],
                    "predicted_id": oracle_id  # This is an oracle_id
                })
        return guesses
