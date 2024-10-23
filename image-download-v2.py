import requests
import os
import glob
import shutil
import random
import albumentations as A
import cv2
import json
from torchvision.datasets import ImageFolder

card_images_folder='/mnt/e/Photos/TableStream/card_images'
augmented_images_folder='/mnt/e/Photos/TableStream/augmented_images'
training_dataset_folder='/mnt/e/Photos/TableStream/training_images'


def download_card_images(output_dir=card_images_folder):
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

    # Filter out tokens
    banned_layouts = {'token', 'double_faced_token', 'scheme', 'planar', 'vanguard','emblem','augment','host','art_series'}

    print('Downloading card images...')
    for card in cards:
        # Check if the card's layout is in the banned layouts
        if card.get('layout') in banned_layouts:
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



def augment_images(input_dir=card_images_folder, output_dir=augmented_images_folder, num_augmented=10,num_strict_blurr=5, skip_augmenting_if_exists=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define augmentation transformations
    transform = A.Compose([
        A.Resize(width=488, height=680),  # Resize to a fixed size for consistency
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=0, p=0.7, border_mode=cv2.BORDER_CONSTANT),  # Shift and scale the card
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=0, value=(114, 114, 114), p=0.5),  # Add padding around the card
        A.RandomResizedCrop(height=640, width=640, scale=(0.5, 1.0), p=0.5),  # Simulate the card being smaller
        A.RandomBrightnessContrast(p=0.5),  # Brightness and contrast adjustment
        A.MotionBlur(blur_limit=5, p=0.5),  # Simulate motion blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add some noise
        A.Blur(blur_limit=(3, 11), p=0.75)  # Apply a blur
    ])

    transform_strict_blurr = A.Compose([
        A.Resize(width=488, height=680),  # Resize to a fixed size for consistency
        A.Downscale(scale_min=0.1, scale_max=0.25, p=1.0),
        A.ImageCompression(quality_lower=5, quality_upper=10, p=1.0),
        A.Blur(blur_limit=(3, 11), p=0.75)  # Apply a blur
    ])

    count = 0
    total = len(os.listdir(input_dir))
    for filename in os.listdir(input_dir):
        # Save the original image with 'aug_99' appended to the filename
        original_output_filename = f"{os.path.splitext(filename)[0]}_aug_99.jpg"
        original_output_path = os.path.join(output_dir, original_output_filename)

        if skip_augmenting_if_exists and os.path.exists(original_output_path):
            print(f'Skipping {filename} because {original_output_filename} already exists.')
            continue
        
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
                #print(f'Saved augmented image {output_filename}')

            # Perform strict blur augmentations, starting after regular augmentations
            for j in range(num_strict_blurr):
                augmented_blur = transform_strict_blurr(image=image)
                augmented_image_blur = augmented_blur['image']
                # The new index for the strict blur augmentations starts after the regular ones
                output_filename_blur = f"{os.path.splitext(filename)[0]}_aug_{num_augmented + j}.jpg"
                output_path_blur = os.path.join(output_dir, output_filename_blur)
                cv2.imwrite(output_path_blur, augmented_image_blur)
                #print(f'Saved strict blur augmented image {output_filename_blur}')

            cv2.imwrite(original_output_path, image)  # Save original image
            count = count + 1
            if count % 100 == 0:
                print(f'{count}/{total} original files augmented')
            #print(f'Saved original image {original_output_filename}')
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

    count = 0
    goal = len(images_by_parent_id)
    print(goal)
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
            #train_images = images[:split_idx]
            train_images = images[:]
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
            #print(f'Moved {img} to {train_parent_dir}')
    
        # Move validation images
        for img in val_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(val_parent_dir, img)
            #shutil.move(src_path, dest_path)
            shutil.copy2(src_path, dest_path)
            #print(f'Moved {img} to {val_parent_dir}')

        count = count +1
        print(f"{count}/{goal} copied")
    
    print("Image moving complete.")




def create_mappings():
    # Define the path to your training/validation dataset (use the root folder where the class folders are stored)
    dataset_path = f"{training_dataset_folder}/train"  # Replace with your actual dataset path
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




#download_card_images()
#augment_images(card_images_folder,augmented_images_folder,15,4, True) check this as we just added blurrr
#move_images(augmented_images_folder, training_dataset_folder)
#generate_annotations()
create_mappings()
# augment_and_split_images(
#     input_dir='card_images',
#     output_dir='classified_images',
#     num_augmented=4,  # Desired number of augmentations per image
#     train_split=0.8    # 80% training, 20% validation
# )