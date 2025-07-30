import os
import sys
import time
from contextlib import contextmanager

import torch
from opensearchpy import OpenSearch, helpers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import numpy as np # Make sure to import numpy
# --- ⚙️ CONFIGURATION ---
IMAGE_DIR = '/mnt/e/Photos/TableStream/augmented_images'  # UPDATE THIS
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_USER = "admin"
OPENSEARCH_PASSWORD = "YourStrongPassword123!" # UPDATE THIS
INDEX_NAME = "magic_cards"

# --- PERFORMANCE TUNING ---
# Adjust based on your GPU's VRAM. Powers of 2 are common (32, 64, 128).
INFERENCE_BATCH_SIZE = 64 
# Adjust based on your CPU cores. Good starting point is half your core count.
NUM_WORKERS = 4 
# How many documents to send to OpenSearch in a single bulk request.
OPENSEARCH_CHUNK_SIZE = 500 

# List of valid image file extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
# --- END CONFIGURATION ---

def create_opensearch_client():
    """Creates and returns an OpenSearch client instance."""
    return OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
        use_ssl=True, verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False
    )

def standardize_image(img: Image.Image, target_size=(224, 224), fill_color=(0, 0, 0)) -> Image.Image:
    resized_img = img.copy()
    resized_img.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_size[0] - resized_img.width) // 2
    paste_y = (target_size[1] - resized_img.height) // 2
    new_img.paste(resized_img, (paste_x, paste_y))
    return new_img

@contextmanager
def optimize_index_for_bulk(client, index_name):
    """
    A context manager to temporarily disable index replicas and refresh interval
    for faster bulk indexing.
    """
    original_settings = {}
    try:
        print("Temporarily optimizing index settings for bulk import...")
        settings_to_get = ['index.number_of_replicas', 'index.refresh_interval']
        response = client.indices.get_settings(index=index_name, name=",".join(settings_to_get))
        
        # Store original settings
        index_settings = response.get(index_name, {}).get('settings', {})
        original_settings = {
            'number_of_replicas': index_settings.get('index', {}).get('number_of_replicas', '1'),
            'refresh_interval': index_settings.get('index', {}).get('refresh_interval', '1s')
        }

        # Apply optimized settings
        body = {
            "index": {
                "number_of_replicas": "0",
                "refresh_interval": "-1"
            }
        }
        client.indices.put_settings(index=index_name, body=body)
        yield
    finally:
        if original_settings:
            print("\nRestoring original index settings...")
            body = {"index": original_settings}
            client.indices.put_settings(index=index_name, body=body)
        print("Index settings restored.")

def get_existing_scryfall_ids(client, index_name):
    """Efficiently fetches all scryfall_ids currently in the index."""
    print("Fetching existing Scryfall IDs from OpenSearch...")
    existing_ids = set()
    try:
        # Use a simple query and scroll to get all IDs
        query = {"query": {"match_all": {}}, "_source": ["scryfall_id"], "size": 10000}
        for doc in helpers.scan(client, query=query, index=index_name, scroll='5m'):
            existing_ids.add(doc['_source']['scryfall_id'])
            if len(existing_ids) % 10000 == 0:
                sys.stdout.write(f"\rFound {len(existing_ids)} existing IDs...")
                sys.stdout.flush()
    except Exception as e:
        if "index_not_found_exception" in str(e):
            print("Index not found. Starting from scratch.")
        else:
            print(f"\nCould not fetch existing IDs: {e}")
    print(f"\nFinished fetching. Found {len(existing_ids)} total existing IDs.")
    return existing_ids

class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading images from a list of file paths."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Load image and ensure it's RGB
            image = Image.open(image_path).convert("RGB")
            filename = os.path.basename(image_path)
            base_filename = os.path.splitext(filename)[0]
            scryfall_id = base_filename.split('_aug')[0]
            return image, scryfall_id
        except Exception as e:
            # Return None if an image is corrupt, handle in collate_fn
            print(f"\n[Warning] Corrupt image at {image_path}: {e}")
            return None, None

def collate_fn(batch):
    """Custom collate function to filter out None values from failed image loads."""
    # Filter out None entries from the batch
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None
    # Unzip the batch
    images, scryfall_ids = zip(*batch)
    return images, scryfall_ids

def main():
    """Main execution function."""
    start_time = time.time()
    
    # --- 1. Setup ---
    client = create_opensearch_client()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model.eval() # Set model to evaluation mode

    # --- 2. Find new images to process ---
    existing_ids = get_existing_scryfall_ids(client, INDEX_NAME)
    
    print("Discovering all image files...")
    all_image_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(IMAGE_DIR)
        for file in files if file.lower().endswith(VALID_EXTENSIONS)
    ]
    
    # Filter out images that have already been indexed
    # new_image_files = [
    #     fp for fp in all_image_files 
    #     if os.path.splitext(os.path.basename(fp))[0].split('_aug')[0] not in existing_ids
    # ]

    # --- START MODIFICATION ---

    # First, filter for only the un-augmented images ("_aug_999999")
    unaugmented_files = [
        fp for fp in all_image_files
        if os.path.splitext(os.path.basename(fp))[0].endswith('_aug_999999')
    ]
    
    print(f"Found {len(unaugmented_files)} un-augmented images to consider.")

    # Next, filter out images that have already been indexed from that smaller list
    new_image_files = [
        fp for fp in unaugmented_files
        if os.path.splitext(os.path.basename(fp))[0].split('_aug')[0] not in existing_ids
    ]
    
    # --- END MODIFICATION ---

    if not new_image_files:
        print("No new images to index. All done!")
        return
        
    print(f"Found {len(all_image_files)} total images. {len(new_image_files)} are new and will be indexed.")

    # --- 3. Create DataLoader for parallel processing ---
    dataset = ImageDataset(new_image_files)
    data_loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- 4. Process and Index in Batches ---
    actions_buffer = []
    total_indexed = 0
    
    with optimize_index_for_bulk(client, INDEX_NAME):
        # Use tqdm for a clean progress bar
        pbar = tqdm(data_loader, desc="Embedding and Indexing Batches")
        for image_batch, scryfall_id_batch in pbar:
            if image_batch is None: # Skip if the whole batch was corrupt
                continue

            standardized_batch = [standardize_image(img) for img in image_batch]

            # --- BATCHED INFERENCE ---
            with torch.no_grad():
                inputs = processor(images=standardized_batch, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()

            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norm

            # Create bulk actions for the current batch
            for i in range(len(scryfall_id_batch)):
                action = {
                    "_index": INDEX_NAME,
                    "_source": {
                        "scryfall_id": scryfall_id_batch[i],
                        "embedding": normalized_embeddings[i].tolist()
                    }
                }
                actions_buffer.append(action)

            # If buffer is full, send to OpenSearch
            if len(actions_buffer) >= OPENSEARCH_CHUNK_SIZE:
                try:
                    success, _ = helpers.bulk(client, actions_buffer, request_timeout=200)
                    total_indexed += success
                    pbar.set_postfix({"last_chunk_indexed": success, "total_indexed": total_indexed})
                except Exception as e:
                    print(f"\nError during bulk indexing: {e}")
                actions_buffer = [] # Clear the buffer

    # --- 5. Index any remaining actions in the buffer ---
    if actions_buffer:
        print("\nIndexing final batch...")
        try:
            success, _ = helpers.bulk(client, actions_buffer, request_timeout=200)
            total_indexed += success
        except Exception as e:
            print(f"\nError during final bulk indexing: {e}")

    print(f"\n--- Indexing Complete ---")
    print(f"Successfully indexed {total_indexed} new documents.")
    end_time = time.time()
    print(f"Total script time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()