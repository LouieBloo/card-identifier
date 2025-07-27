import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from opensearchpy import OpenSearch, helpers
import time
import sys

IMAGE_DIR = '/mnt/e/Photos/TableStream/augmented_images'
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "YourStrongPassword123!"
INDEX_NAME = "magic_cards"
BATCH_SIZE = 500 
# List of valid image file extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

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
    
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base', use_fast=True)
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    return model, processor, device

def get_existing_scryfall_ids(client):
    """
    Efficiently fetches all scryfall_ids currently in the index using the scroll API.
    """
    print("Fetching existing Scryfall IDs from OpenSearch (this may take a moment for large indexes)...")
    existing_ids = set()
    try:
        response = client.search(
            index=INDEX_NAME,
            body={"query": {"match_all": {}}, "_source": ["scryfall_id"], "size": 10000},
            scroll="5m"
        )
        scroll_id = response.get('_scroll_id')
        
        while scroll_id and len(response['hits']['hits']) > 0:
            for hit in response['hits']['hits']:
                existing_ids.add(hit['_source']['scryfall_id'])
            
            # Use a progress indicator for fetching
            sys.stdout.write(f"\rFound {len(existing_ids)} existing IDs...")
            sys.stdout.flush()

            response = client.scroll(scroll_id=scroll_id, scroll='5m')
            scroll_id = response.get('_scroll_id')

    except Exception as e:
        if "index_not_found_exception" in str(e):
             print("\nIndex not found. Starting from scratch.")
        else:
            print(f"\nAn error occurred while fetching existing IDs: {e}")
    
    print(f"\nFinished fetching. Found {len(existing_ids)} total existing IDs.")
    return existing_ids

def generate_actions(image_files, existing_ids, model, processor, device):
    """
    A generator function that yields OpenSearch bulk actions, one by one.
    This is memory-efficient as it doesn't store all actions in a list.
    """
    total_files = len(image_files)
    processed_count = 0
    skipped_count = 0
    
    for image_path in image_files:
        processed_count += 1
        filename = os.path.basename(image_path)
        base_filename = os.path.splitext(filename)[0]
        scryfall_id = base_filename.split('_aug')[0]

        if scryfall_id in existing_ids:
            skipped_count += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs.pooler_output.squeeze().cpu().numpy()
                #embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            yield {
                "_index": INDEX_NAME,
                "_source": {
                    "scryfall_id": scryfall_id,
                    "embedding": embedding.tolist()
                }
            }
        except Exception as e:
            print(f"\n[Warning] Could not process image {image_path}: {e}")
            
        # Update progress indicator
        progress = (processed_count / total_files) * 100
        sys.stdout.write(f"\rProcessing files: {processed_count}/{total_files} ({progress:.2f}%) | Skipped: {skipped_count}")
        sys.stdout.flush()


def main():
    """Main execution function."""
    start_time = time.time()
    
    client = create_opensearch_client()
    model, processor, device = create_embedding_model()
    
    existing_ids = get_existing_scryfall_ids(client)

    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Directory not found at '{IMAGE_DIR}'")
        return

    print("Discovering all image files...")
    all_image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(IMAGE_DIR)
        for file in files if file.lower().endswith(VALID_EXTENSIONS)
    ]
    
    if not all_image_files:
        print("No image files found. Exiting.")
        return
        
    print(f"Found {len(all_image_files)} total image files. Starting processing and indexing.")

    # Use the generator to create actions and helpers.bulk to consume them in batches
    action_generator = generate_actions(all_image_files, existing_ids, model, processor, device)
    
    try:
        success_count = 0
        fail_count = 0
        # The helpers.bulk function can consume a generator directly, which is highly memory efficient.
        for success, info in helpers.streaming_bulk(client, action_generator, chunk_size=BATCH_SIZE, request_timeout=200):
            if not success:
                fail_count += 1
            else:
                success_count += 1
        
        print(f"\n\n--- Indexing Complete ---")
        print(f"Successfully indexed {success_count} new documents.")
        if fail_count > 0:
            print(f"Failed to index {fail_count} documents.")

    except Exception as e:
        print(f"\nAn error occurred during bulk indexing: {e}")

    end_time = time.time()
    print(f"Total script time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()