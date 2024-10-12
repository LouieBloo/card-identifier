import requests
import os
import glob
import shutil
import random
import albumentations as A
import cv2

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
        legalities = card.get('legalities', {})
        if legalities.get('commander') != 'legal':
            continue  # Skip cards not legal in Commander

        if 'image_uris' in card and 'normal' in card['image_uris']:
            image_url = card['image_uris']['normal']
        # Check if 'card_faces' exists (e.g., double-faced cards)
        elif 'card_faces' in card and isinstance(card['card_faces'], list):
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
    transform = A.Compose([
        A.Resize(width=488, height=680),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(blur_limit=5, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Rotate(limit=90, p=0.3),
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



def split_dataset(images_dir='augmented_images', labels_dir='annotations', output_dir='dataset', train_ratio=0.8):
    images = glob.glob(os.path.join(images_dir, '*.jpg'))
    random.shuffle(images)
    train_count = int(len(images) * train_ratio)
    train_images = images[:train_count]
    val_images = images[train_count:]

    for phase, image_list in [('train', train_images), ('val', val_images)]:
        os.makedirs(os.path.join(output_dir, 'images', phase), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', phase), exist_ok=True)
        for image_path in image_list:
            filename = os.path.basename(image_path)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            shutil.copy(image_path, os.path.join(output_dir, 'images', phase, filename))
            shutil.copy(os.path.join(labels_dir, label_filename), os.path.join(output_dir, 'labels', phase, label_filename))
            print(f'Copied {filename} to {phase} set')


def move_images(input_dir, output_dir, train_split=0.8):
    # Create training and validation directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Dictionary to hold images by their ID (before _aug)
    images_by_id = {}

    # Group images by their primary ID
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):  # You can adjust for other image types if needed
            primary_id = filename.split('_aug')[0]
            if primary_id not in images_by_id:
                images_by_id[primary_id] = []
            images_by_id[primary_id].append(filename)
    
    # For each ID, split the images into training and validation
    for primary_id, images in images_by_id.items():
        random.shuffle(images)  # Randomize the augmentations
        split_idx = int(len(images) * train_split)

        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create directories for this ID in the training and validation subfolders
        train_id_dir = os.path.join(train_dir, primary_id)
        val_id_dir = os.path.join(val_dir, primary_id)

        if not os.path.exists(train_id_dir):
            os.makedirs(train_id_dir)
        
        if not os.path.exists(val_id_dir):
            os.makedirs(val_id_dir)

        # Move training images
        for img in train_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(train_id_dir, img)
            shutil.move(src_path, dest_path)
            print(f'Moved {img} to {train_id_dir}')

        # Move validation images
        for img in val_images:
            src_path = os.path.join(input_dir, img)
            dest_path = os.path.join(val_id_dir, img)
            shutil.move(src_path, dest_path)
            print(f'Moved {img} to {val_id_dir}')

# Example usage:
input_dir = 'augmented_images'  # Your flat directory with all images
output_dir = 'classified_images'  # Your target directory for splitting
move_images(input_dir, output_dir)

# download_card_images()
# augment_images()
# generate_annotations()