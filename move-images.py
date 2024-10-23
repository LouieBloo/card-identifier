import os
import requests
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def move_images(input_dir, output_dir, val_split=0.2):
    # Create training and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Step 1: Load Scryfall card data
    print("Downloading Scryfall default_cards data...")
    response = requests.get('https://api.scryfall.com/bulk-data/default_cards')
    if response.status_code != 200:
        print("Failed to download Scryfall data")
        return
    
    bulk_data_url = response.json()['download_uri']
    bulk_response = requests.get(bulk_data_url)
    if bulk_response.status_code != 200:
        print("Failed to download bulk card data")
        return
    
    cards_data = bulk_response.json()
    print("Scryfall data loaded successfully.")

    # Step 2: Build mapping from card IDs to their parent IDs
    card_id_to_parent_id = {}
    for card in cards_data:
        card_id = card['id']
        parent_id = card.get('oracle_id', card_id)
        card_id_to_parent_id[card_id] = parent_id

    print("Card ID to parent ID mapping created.")

    # Dictionary to hold images by their parent ID
    images_by_parent_id = {}

    # Step 3: Group images by their parent ID
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):  # Adjust for other image types if needed
            # Extract the card ID from the filename
            primary_id = filename.split('_aug')[0]
            # Map the card ID to its parent ID
            parent_id = card_id_to_parent_id.get(primary_id, primary_id)  # Default to primary_id if not found
            images_by_parent_id.setdefault(parent_id, []).append(filename)
    
    print("Images grouped by parent ID.")

    # Prepare lists of copy tasks
    train_copy_tasks = []
    val_copy_tasks = []

    count = 0
    goal = len(images_by_parent_id)
    print(f"Total parent IDs to process: {goal}")

    # Step 4: For each parent ID, copy images to the training set and select 20% for validation
    for parent_id, images in images_by_parent_id.items():
        random.shuffle(images)  # Randomize the images
        total_images = len(images)
        val_count = max(1, int(total_images * val_split)) if total_images > 1 else 0

        val_images = images[:val_count]
        train_images = images

        # Create directories for this parent ID
        train_parent_dir = os.path.join(train_dir, parent_id)
        val_parent_dir = os.path.join(val_dir, parent_id)

        os.makedirs(train_parent_dir, exist_ok=True)
        if val_images:
            os.makedirs(val_parent_dir, exist_ok=True)

        # Prepare copy tasks for training images
        for img in train_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(train_parent_dir, img)
            train_copy_tasks.append((src_path, dest_path))

        # Prepare copy tasks for validation images (subset of training images)
        for img in val_images:
            src_path = os.path.join(train_parent_dir, img)  # Since it's already copied to train_dir
            dest_path = os.path.join(val_parent_dir, img)
            val_copy_tasks.append((src_path, dest_path))

        count += 1
        if count % 1000 == 0 or count == goal:
            print(f"{count}/{goal} parent IDs processed")

    # Function to copy files (used for multithreading)
    def copy_file(src_dest_tuple):
        src, dest = src_dest_tuple
        shutil.copy2(src, dest)

    # Use ThreadPoolExecutor to copy training images
    print("Starting to copy training images...")
    total_train_tasks = len(train_copy_tasks)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(copy_file, task): task for task in train_copy_tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10000 == 0 or completed == total_train_tasks:
                print(f"Copied {completed}/{total_train_tasks} training images")

    # Use ThreadPoolExecutor to copy validation images
    print("Starting to copy validation images...")
    total_val_tasks = len(val_copy_tasks)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(copy_file, task): task for task in val_copy_tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10000 == 0 or completed == total_val_tasks:
                print(f"Copied {completed}/{total_val_tasks} validation images")

    print("Image copying complete.")

# Example usage:
move_images('/mnt/d/Photos/TableStream/augmented_images', '/mnt/d/Photos/TableStream/training_images')
