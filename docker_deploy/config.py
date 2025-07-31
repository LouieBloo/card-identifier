
# --- Model File Paths ---
YOLO_MODEL_PATH = 'best.pt'
CNN_CLASSIFIER_MODEL_PATH = 'magic_card_classifier_v8.pth'
CNN_CLASSIFIER_MODEL_PATH_V2 = 'magic_card_classifier_efficientnet_v2.pth'
CLASS_MAPPING_PATH = 'class_to_idx.json'
CLASS_MAPPING_PATH_V2 = 'class_to_idx_v2.json'

# --- OpenSearch Configuration ---
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
# IMPORTANT: Use environment variables for passwords in production
OPENSEARCH_PASSWORD = "YourStrongPassword123!" 
OPENSEARCH_INDEX_NAME = "magic_cards"

# --- Scryfall API ---
SCRYFALL_API_URL = 'https://api.scryfall.com'

# --- Search Parameters ---
KNN_TOP_K = 5  # Number of results to fetch for k-NN search
CNN_TOP_K = 3  # Number of results to fetch for CNN classification