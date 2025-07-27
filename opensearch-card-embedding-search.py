import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch
import time

# UPDATE THIS to the path of the image you want to search for
TARGET_IMAGE_PATH = '/mnt/e/Photos/TableStream/augmented_images/0000419b-0bba-4488-8f7a-6194544ce91e_aug_999999.jpg' 

OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "YourStrongPassword123!" # UPDATE THIS
INDEX_NAME = "magic_cards"
TOP_K = 5 # How many top results to retrieve
# --- END CONFIGURATION ---

def create_opensearch_client():
    """Creates and returns an OpenSearch client instance."""
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client

def create_embedding_model():
    """Loads and returns the DINOv2 model and processor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    return model, processor, device

def generate_embedding(image_path, model, processor, device):
    """Generates a vector embedding for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
        return embedding
    except FileNotFoundError:
        print(f"Error: Test image not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def search_for_similar_images(client, vector_embedding):
    """Performs a k-NN search in OpenSearch."""
    
    # This is the standard k-NN query structure
    query = {
        "size": TOP_K,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector_embedding.tolist(),
                    "k": TOP_K
                }
            }
        }
    }
    
    print("\nExecuting k-NN search...")
    try:
        response = client.search(
            index=INDEX_NAME,
            body=query
        )
        return response
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return None

def main():
    """Main execution function."""
    print(f"Preparing to search for image: {TARGET_IMAGE_PATH}")
    
    # 1. Initialize clients and models
    client = create_opensearch_client()
    model, processor, device = create_embedding_model()
    
    # 2. Generate embedding for the target image
    start_time = time.time()
    target_embedding = generate_embedding(TARGET_IMAGE_PATH, model, processor, device)
    
    if target_embedding is None:
        return # Exit if embedding failed
        
    embed_time = time.time()
    print(f"Embedding generated in {embed_time - start_time:.4f} seconds.")
    
    # 3. Perform the search
    search_response = search_for_similar_images(client, target_embedding)
    search_time = time.time()
    
    if search_response is None:
        return

    print(f"Search completed in {search_time - embed_time:.4f} seconds.")
    
    # 4. Display results
    hits = search_response['hits']['hits']
    
    if not hits:
        print("\nNo results found.")
        return
        
    print(f"\n--- Top {len(hits)} Results ---")
    for i, hit in enumerate(hits):
        score = hit['_score']
        scryfall_id = hit['_source']['scryfall_id']
        print(f"{i+1}. Scryfall ID: {scryfall_id:<40} Score: {score:.4f}")

if __name__ == "__main__":
    main()