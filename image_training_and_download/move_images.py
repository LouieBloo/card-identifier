import os
import shutil
import requests
import json
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# --- Configuration ---
# Path to the folder containing your augmented images
INPUT_DIR = "/mnt/e/Photos/TableStream/augmented_images"
# Path to the folder where the 'train' and 'val' directories will be created
OUTPUT_DIR = "/mnt/e/Photos/TableStream/training_images"
# The proportion of images to use for training
TRAIN_SPLIT = 0.8
# The maximum number of parallel workers for copying files
MAX_WORKERS = 8 # Adjust based on your CPU cores and disk speed

def get_scryfall_data(cache_path='scryfall_default_cards.json', max_age_seconds=86400):
    """
    Downloads or loads Scryfall bulk data from a local cache.

    Args:
        cache_path (str): The path to store the cached JSON file.
        max_age_seconds (int): The maximum age of the cache file in seconds before re-downloading.
                               Defaults to 1 day.

    Returns:
        list: A list of card data dictionaries, or None if failed.
    """
    # Check if a recent cache file exists
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path)) < max_age_seconds:
        print(f"Loading Scryfall data from local cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # If no recent cache, download the data
    print("Downloading Scryfall bulk data (this may take a moment)...")
    try:
        response = requests.get('https://api.scryfall.com/bulk-data/default_cards')
        response.raise_for_status()
        bulk_data_url = response.json()['download_uri']

        bulk_response = requests.get(bulk_data_url, stream=True)
        bulk_response.raise_for_status()

        # Save the downloaded data to the cache file
        with open(cache_path, 'wb') as f:
            for chunk in bulk_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Scryfall data downloaded and cached at {cache_path}")
        
        # Now load from the just-created cache file
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download Scryfall data: {e}")
        return None

def build_parent_id_map(cards_data):
    """Builds a mapping from a card's print ID to its functional (oracle) ID."""
    print("Building card ID to parent ID map...")
    card_id_to_parent_id = {}
    for card in cards_data:
        card_id = card['id']
        # Use oracle_id if available to group all printings.
        # Otherwise, the card is its own parent (e.g., for tokens, art cards).
        # This fixes the bug where cards without an oracle_id were skipped.
        parent_id = card.get('oracle_id', card_id)
        card_id_to_parent_id[card_id] = parent_id
    print("ID map created.")
    return card_id_to_parent_id

def get_existing_files(root_dir):
    """Scans a directory tree and returns a set of all filenames for fast lookups."""
    print(f"Scanning for existing files in {root_dir}...")
    existing_files = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            existing_files.add(f)
    print(f"Found {len(existing_files)} existing files.")
    return existing_files

def process_image_group(args):
    """
    Handles file copying for a single group of images. Designed for parallel execution.
    """
    parent_id, images, train_split, input_dir, train_dir, val_dir = args
    
    # Create directories for this parent ID
    train_parent_dir = os.path.join(train_dir, parent_id)
    val_parent_dir = os.path.join(val_dir, parent_id)
    os.makedirs(train_parent_dir, exist_ok=True)
    os.makedirs(val_parent_dir, exist_ok=True)

    random.shuffle(images)
    total_images = len(images)
    
    # --- CRITICAL FIX for Data Leakage ---
    # Ensure a clean split between training and validation sets.
    if total_images > 1:
        # Ensure at least one image goes to training, even for small groups
        split_idx = max(1, int(total_images * train_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
    else:
        # If only one image exists, it can only go to training
        train_images = images
        val_images = []

    # Copy training images
    for img in train_images:
        src_path = os.path.join(input_dir, img)
        dest_path = os.path.join(train_parent_dir, img)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)

    # Copy validation images
    for img in val_images:
        src_path = os.path.join(input_dir, img)
        dest_path = os.path.join(val_parent_dir, img)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            
    return 1 # Return 1 for success to count in the progress bar

def move_images_optimized(input_dir, output_dir, train_split=0.8):
    """
    Optimized function to sort images into train/val folders by their functional card identity.
    """
    # Create main training and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Step 1: Get Scryfall data (from cache or download)
    cards_data = get_scryfall_data()
    if not cards_data:
        return

    # Step 2: Build the ID map
    card_id_to_parent_id = build_parent_id_map(cards_data)

    # Step 3: Get a set of all files already in the destination directory
    existing_files = get_existing_files(output_dir)

    # Step 4: Group images by their parent ID, skipping existing ones
    print("Grouping new images by parent ID...")
    images_by_parent_id = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') and filename not in existing_files:
            primary_id = filename.split('_aug')[0]
            # Use the map to find the parent_id; this will now correctly handle all cards
            parent_id = card_id_to_parent_id.get(primary_id)
            
            if parent_id:
                if parent_id not in images_by_parent_id:
                    images_by_parent_id[parent_id] = []
                images_by_parent_id[parent_id].append(filename)
    
    print(f"Found {len(images_by_parent_id)} parent cards with new images to process.")
    if not images_by_parent_id:
        print("No new images to move. Exiting.")
        return

    # Step 5: Prepare arguments for parallel processing
    tasks = [
        (parent_id, images, train_split, input_dir, train_dir, val_dir)
        for parent_id, images in images_by_parent_id.items()
    ]

    # Step 6: Process and copy files in parallel with a progress bar
    print(f"Moving images using up to {MAX_WORKERS} parallel workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_image_group, tasks), total=len(tasks), desc="Processing Cards"))
    
    print(f"\nImage processing complete. Processed {sum(results)} card groups.")

if __name__ == '__main__':
    # Make sure to set your actual input and output directories
    move_images_optimized(INPUT_DIR, OUTPUT_DIR, train_split=TRAIN_SPLIT)
