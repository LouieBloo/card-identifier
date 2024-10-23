import os
import statistics
import json
import requests

def count_images_in_folders(directory):
    folder_image_count = {}

    # Loop through each folder in the directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        
        if os.path.isdir(folder_path):
            # Count the number of files (images) in the folder
            image_count = len([file for file in os.listdir(folder_path) 
                               if os.path.isfile(os.path.join(folder_path, file))])
            folder_image_count[folder_name] = image_count

    # Sort folders by the number of images, from most to least
    sorted_folders = sorted(folder_image_count.items(), key=lambda x: x[1], reverse=False)

    # Print the results
    for folder, count in sorted_folders:
        print(f"{folder}: {count} images")
    
    return folder_image_count

def calculate_statistics(image_counts):
    counts = list(image_counts.values())
    
    if len(counts) == 0:
        print("No data available.")
        return

    mean = statistics.mean(counts)
    stddev = statistics.stdev(counts)

    print(f"\nMean (average) number of images per folder: {mean:.2f}")
    print(f"Standard deviation of images per folder: {stddev:.2f}")



def check_folders(input_dir):
    output_json_file="folder_counts.json"

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
    # Step 4: Create a dictionary with Oracle IDs and the number of files per folder
    oracle_id_file_count = {oracle_id: len(files) for oracle_id, files in images_by_parent_id.items()}

    # Step 5: Save the Oracle ID and file count data to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(oracle_id_file_count, json_file, indent=4)

    print(f"Oracle ID and file count saved to {output_json_file}")


def removeGarbage():
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

    source_folder = "/mnt/d/Photos/TableStream/card_images"
    augment_folder = "/mnt/d/Photos/TableStream/augmented_images"

    # Step 3: Define banned layouts
    banned_layouts = {
        'token', 'double_faced_token', 'scheme', 'planar', 'vanguard', 
        'emblem', 'augment', 'host', 'art_series'
    }

    # Remove files from the source folder
            

    # Step 4: Collect Scryfall IDs of banned cards
    banned_scryfall_ids = set()  # Using a set for fast lookup
    for card in cards_data:
        if card.get('layout') in banned_layouts:
            scryfall_id = card.get('id')
            banned_scryfall_ids.add(scryfall_id)
            source_file = os.path.join(source_folder, f"{scryfall_id}.jpg")
            if os.path.exists(source_file):
                print(f"Deleting {source_file}")
                os.remove(source_file)

    # Print the number of IDs to delete
    print(f"Collected {len(banned_scryfall_ids)} banned Scryfall IDs.")

    # Step 5: Loop through the augment folder once and delete matching files
    for entry in os.scandir(augment_folder):
        if entry.is_file():
            # Extract the Scryfall ID from the filename before the '_aug_' part
            file_name = entry.name
            scryfall_id = file_name.split('_aug_')[0]

            # Check if the Scryfall ID is in the banned list
            if scryfall_id in banned_scryfall_ids:
                augment_file_path = os.path.join(augment_folder, file_name)
                print(f"Deleting {augment_file_path}")
                os.remove(augment_file_path)

    print("Garbage removal completed.")

def add_to_augment_images(input_dir='/mnt/d/Photos/TableStream/card_images', output_dir='/mnt/d/Photos/TableStream/augmented_images', num_augmented=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    transform = A.Compose([
        A.Resize(width=488, height=680),  # Resize to a fixed size for consistency
        A.Downscale(scale_min=0.1, scale_max=0.25, p=1.0),
        A.ImageCompression(quality_lower=5, quality_upper=10, p=1.0),
        A.Blur(blur_limit=(3, 11), p=0.75)  # Apply a blur
    ])

    count = 0
    total = len(os.listdir(input_dir))
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        # Perform augmentations and save augmented images
        for i in range(num_augmented):
            augmented = transform(image=image)
            augmented_image = augmented['image']
            output_filename = f"{os.path.splitext(filename)[0]}_aug_{i+15}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, augmented_image)
            #print(f'Saved augmented image {output_filename}')

        count = count + 1
        if count % 100 == 0:
            print(f'{count}/{total} original files augmented')
        #print(f'Saved original image {original_output_filename}')

# Path to the training folder
training_folder = 'classified_images/train'  # Adjust this path if necessary
# Count images in each folder
#image_counts = count_images_in_folders(training_folder)

# Calculate and display the statistics
#calculate_statistics(image_counts)

#check_folders('classified_images/train')
#count_images_in_folders(training_folder)

#removeGarbage()
for i in range(15):
    print(i)