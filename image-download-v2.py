import requests
import os
import glob
import shutil
import random
import albumentations as A
import cv2
import json
from torchvision.datasets import ImageFolder

def download_card_images(output_dir='card_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Scryfall bulk data URL
    bulk_data_url = 'https://api.scryfall.com/bulk-data/default_cards'

    response = requests.get(bulk_data_url)
    data = response.json()
    download_uri = data['download_uri']

    # Download the bulk data
    print('Downloading card data...')
    response = requests.get(download_uri)
    cards = response.json()

    print('Downloading card images...')
    for card in cards:
        # Filter out tokens
        if card.get('layout') == 'token' or card.get('layout') == 'double_faced_token':
            continue  # Skip tokens

        # Check if the card is legal in Commander
        # legalities = card.get('legalities', {})
        # if legalities.get('commander') != 'legal':
        #     continue  # Skip cards not legal in Commander

        if 'image_uris' in card and 'normal' in card['image_uris']:
            image_url = card['image_uris']['normal']
        # Check if 'card_faces' exists (e.g., double-faced cards)
        elif 'card_faces' in card and isinstance(card['card_faces'], list) and 'image_uris'  in card['card_faces'][0]:
            image_url = card['card_faces'][0]['image_uris']['normal']
        else:
            print(f"No image URL found for {card_name}")

        card_name = card['name'].replace('/', '_').replace(' ', '_')
        id = card['id']
        image_path = os.path.join(output_dir, f"{id}.jpg")
        if not os.path.exists(image_path):
            try:
                img_data = requests.get(image_url).content
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                    print(f'Downloaded {card_name}')
            except Exception as e:
                print(f'Failed to download {card_name}: {e}')
        else:
            print(f'{card_name} already exists.')



def augment_images(input_dir='card_images', output_dir='augmented_images', num_augmented=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define augmentation transformations
    # transform = A.Compose([
    #     A.Resize(width=488, height=680),
    #     A.RandomBrightnessContrast(p=0.5),
    #     A.MotionBlur(blur_limit=5, p=0.5),
    #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    #     A.Rotate(limit=90, p=0.3),
    #     A.Blur(blur_limit=(3,11), p=0.75)
    # ])

    transform = A.Compose([
        A.Resize(width=488, height=680),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # A.Rotate(limit=90, p=0.3),
        A.Blur(blur_limit=(3,11), p=0.75)
    ])

    for filename in os.listdir(input_dir):
        # Save the original image with 'aug_99' appended to the filename
        original_output_filename = f"{os.path.splitext(filename)[0]}_aug_99.jpg"
        original_output_path = os.path.join(output_dir, original_output_filename)
        
        if not os.path.exists(original_output_path):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # Perform augmentations and save augmented images
            for i in range(num_augmented):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, augmented_image)
                print(f'Saved augmented image {output_filename}')

            cv2.imwrite(original_output_path, image)  # Save original image
            print(f'Saved original image {original_output_filename}')
        else:
            print(f'{original_output_path} already exists.')

def generate_annotations(image_dir='augmented_images', label_dir='annotations', class_id=0):
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # YOLO format: class x_center y_center width height (normalized)
        annotation = f"{class_id} 0.5 0.5 1.0 1.0\n"
        
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(annotation)
            print(f'Created annotation for {filename}')



def move_images(input_dir, output_dir, train_split=0.8):
    # Create training and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
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
    # If a card is a variation, map it to its parent; else, map to itself
    card_id_to_parent_id = {}
    for card in cards_data:
        card_id = card['id']
        if 'prints_search_uri' in card and 'oracle_id' in card:
            parent_id = card['oracle_id']
        else:
            parent_id = card_id  # Use card ID if no oracle_id is available
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
            if parent_id not in images_by_parent_id:
                images_by_parent_id[parent_id] = []
            images_by_parent_id[parent_id].append(filename)
    
    print("Images grouped by parent ID.")

    # Step 4: For each parent ID, split images into training and validation sets
    for parent_id, images in images_by_parent_id.items():
        random.shuffle(images)  # Randomize the augmentations
        total_images = len(images)
        split_idx = int(len(images) * train_split)
    
        # Ensure at least 1 image in both train and val, especially for small datasets
        if total_images == 3:
            train_images = images[:2]  # 2 images for training
            val_images = images[2:]    # 1 image for validation
        elif total_images > 1:
            # If we have more than 3 images, split based on train_split
            split_idx = max(1, split_idx)  # Ensure at least 1 for training
            val_images = images[split_idx:]
            train_images = images[:split_idx]
        else:
            # Handle edge case with 1 image
            train_images = images
            val_images = []  # No validation images
    
        # Create directories for this parent ID in the training and validation subfolders
        train_parent_dir = os.path.join(train_dir, parent_id)
        val_parent_dir = os.path.join(val_dir, parent_id)
    
        if not os.path.exists(train_parent_dir):
            os.makedirs(train_parent_dir)
        
        if not os.path.exists(val_parent_dir):
            os.makedirs(val_parent_dir)
    
        # Move training images
        for img in train_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(train_parent_dir, img)
            #shutil.move(src_path, dest_path)
            shutil.copy2(src_path, dest_path)
            print(f'Moved {img} to {train_parent_dir}')
    
        # Move validation images
        for img in val_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(val_parent_dir, img)
            #shutil.move(src_path, dest_path)
            shutil.copy2(src_path, dest_path)
            print(f'Moved {img} to {val_parent_dir}')
    
    print("Image moving complete.")




def create_mappings():
    # Define the path to your training/validation dataset (use the root folder where the class folders are stored)
    dataset_path = 'classified_images/train'  # Replace with your actual dataset path
    output_file = 'class_to_idx.json'      # Output file to save the mapping

    # Load the dataset using ImageFolder
    dataset = ImageFolder(root=dataset_path)

    # Get the class-to-index mapping
    class_to_idx = dataset.class_to_idx

    # Reverse the mapping to get index-to-class (Scryfall ID)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save the class-to-index and index-to-class mapping to a JSON file
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }

    # Write the mapping to a JSON file
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Class-to-index mapping saved to {output_file}")



def augment_and_split_images(input_dir='card_images', output_dir='output_data', num_augmented=10, train_split=0.8):
    # Create base directories for training and validation
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Define augmentation transformations
    transform = A.Compose([
        A.Resize(width=488, height=680),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # A.Rotate(limit=90, p=0.3),
        A.Blur(blur_limit=(3,11), p=0.75)
    ])
    
    # Step 1: Download Scryfall card data
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
    
    # Step 2: Build mapping from card IDs to their parent Oracle IDs
    card_id_to_oracle_id = {}
    for card in cards_data:
        card_id = card['id']
        oracle_id = card.get('oracle_id', card_id)  # Use card ID if oracle_id is not available
        card_id_to_oracle_id[card_id] = oracle_id
    
    print("Card ID to Oracle ID mapping created.")
    
    # Dictionary to keep track of the number of images per Oracle ID
    oracle_id_image_counts = {}
    
    # Step 3: Process each image
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust for other image types if needed
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image {filename}. Skipping.")
                continue
            
            # Extract the card ID from the filename
            primary_id = filename.split('_aug')[0]
            # Map the card ID to its Oracle ID
            oracle_id = card_id_to_oracle_id.get(primary_id, primary_id)
            
            # Initialize the image count for this Oracle ID if not already done
            if oracle_id not in oracle_id_image_counts:
                oracle_id_image_counts[oracle_id] = 0
            
            # Check how many images already exist for this Oracle ID
            total_images_for_oracle = oracle_id_image_counts[oracle_id]
            
            # If there are already 50 or more images, reduce num_augmented to 1
            if total_images_for_oracle >= 50:
                current_num_augmented = 1
            else:
                current_num_augmented = num_augmented
                # # Adjust current_num_augmented if adding num_augmented would exceed 50 images
                # if total_images_for_oracle + num_augmented > 50:
                #     current_num_augmented = 50 - total_images_for_oracle
            
            # Create a temporary list to hold augmented images for this card
            augmented_images = []
            
            # Save the original image with '_aug_99' appended to the filename
            original_output_filename = f"{primary_id}_aug_99.jpg"
            augmented_images.append((original_output_filename, image))
            oracle_id_image_counts[oracle_id] += 1
            
            # Perform augmentations and save augmented images
            for i in range(current_num_augmented):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                output_filename = f"{primary_id}_aug_{i}.jpg"
                augmented_images.append((output_filename, augmented_image))
                oracle_id_image_counts[oracle_id] += 1
            
            # Save augmented images into a temporary directory organized by Oracle ID
            temp_oracle_dir = os.path.join(output_dir, 'temp', oracle_id)
            os.makedirs(temp_oracle_dir, exist_ok=True)
            
            for img_name, img_data in augmented_images:
                output_path = os.path.join(temp_oracle_dir, img_name)
                cv2.imwrite(output_path, img_data)
                print(f"Saved {img_name} to {temp_oracle_dir}")
    
    print("Augmentation complete. Starting train/validation split...")
    
    # Step 4: Split images into training and validation sets per Oracle ID
    temp_dir = os.path.join(output_dir, 'temp')
    for oracle_id in os.listdir(temp_dir):
        oracle_dir = os.path.join(temp_dir, oracle_id)
        images = [img for img in os.listdir(oracle_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        total_images = len(images)
        split_idx = int(total_images * train_split)
        
        # Ensure at least 1 image in both train and val, especially for small datasets
        if total_images == 3:
            train_images = images[:2]  # 2 images for training
            val_images = images[2:]    # 1 image for validation
        elif total_images > 1:
            split_idx = max(1, split_idx)  # Ensure at least 1 for training
            train_images = images[:split_idx]
            val_images = images[split_idx:]
        else:
            train_images = images
            val_images = []
        
        # Create directories for this Oracle ID in the training and validation subfolders
        train_oracle_dir = os.path.join(train_dir, oracle_id)
        val_oracle_dir = os.path.join(val_dir, oracle_id)
        os.makedirs(train_oracle_dir, exist_ok=True)
        os.makedirs(val_oracle_dir, exist_ok=True)
        
        # Move training images
        for img in train_images:
            src_path = os.path.join(oracle_dir, img)
            dest_path = os.path.join(train_oracle_dir, img)
            shutil.move(src_path, dest_path)
            print(f"Moved {img} to {train_oracle_dir}")
        
        # Move validation images
        for img in val_images:
            src_path = os.path.join(oracle_dir, img)
            dest_path = os.path.join(val_oracle_dir, img)
            shutil.move(src_path, dest_path)
            print(f"Moved {img} to {val_oracle_dir}")
        
        # Remove the temporary Oracle ID directory
        os.rmdir(oracle_dir)
    
    # Remove the temporary directory
    os.rmdir(temp_dir)
    print("Image splitting complete.")

#download_card_images()
#augment_images('card_images','augmented_images',2)
#move_images('augmented_images', 'classified_images')
#generate_annotations()
#create_mappings()

augment_and_split_images(
    input_dir='card_images',
    output_dir='classified_images',
    num_augmented=4,  # Desired number of augmentations per image
    train_split=0.8    # 80% training, 20% validation
)