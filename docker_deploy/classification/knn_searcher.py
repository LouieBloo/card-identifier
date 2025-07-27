# classification/knn_searcher.py
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import config
from services.opensearch_client import get_opensearch_client

class KNNSearcher:
    def __init__(self):
        """Initializes the DINOv2 embedding model and OpenSearch client."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.client = get_opensearch_client()
        print(f"k-NN Searcher (DINOv2) loaded successfully, using device: {self.device}.")

    def _generate_embedding(self, image: Image.Image):
        """Generates a DINOv2 vector embedding for a single image."""
        image = image.convert("RGB")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
            #embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def search_by_image(self, card_image: Image.Image):
        """Generates an embedding and performs a k-NN search."""
        embedding = self._generate_embedding(card_image)
        
        query = {
            "size": config.KNN_TOP_K,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding.tolist(),
                        "k": config.KNN_TOP_K
                    }
                }
            }
        }
        
        try:
            response = self.client.search(index=config.OPENSEARCH_INDEX_NAME, body=query)
        except Exception as e:
            print(f"k-NN search failed: {e}")
            return []

        guesses = []
        for hit in response['hits']['hits']:
            guesses.append({
                "confidence": hit['_score'],
                "predicted_id": hit['_source']['scryfall_id'] # This is a scryfall_id
            })
        return guesses