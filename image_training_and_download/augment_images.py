import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple

# --- 1. Enhanced Augmentation Pipelines ---

# This pipeline creates a wide variety of realistic webcam-like distortions.
WEBCAM_LIKE_TRANSFORM = A.Compose([
    # --- Base Orientation: Is the card tapped or not? ---
    A.RandomRotate90(p=0.5),

    # --- Geometric Distortions: Simulates camera angle and position ---
    A.Perspective(scale=(0.05, 0.1), pad_val=30, pad_mode=cv2.BORDER_CONSTANT, p=0.8), # Darker padding
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=(-0.4, 0.1), # Tighter scale to make the card larger
        rotate_limit=15,
        p=0.9,
        border_mode=cv2.BORDER_CONSTANT,
        value=(30, 30, 30) # Darker background
    ),

    # --- Blurring and Resolution: Simulates poor focus and low-res cameras ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.8),
        A.MotionBlur(blur_limit=(3, 11), p=0.7),
        A.MedianBlur(blur_limit=(3, 7), p=0.5),
    ], p=1.0),
    A.Downscale(scale_min=0.25, scale_max=0.6, interpolation=cv2.INTER_AREA, p=0.7),

    # --- Lighting and Color: Simulates bad lighting, shadows, and webcam color processing ---
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.9),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.7),

    # --- Digital Artifacts: Simulates compression and noise ---
    A.ImageCompression(quality_lower=20, quality_upper=50, p=0.8),
    A.GaussNoise(var_limit=(10.0, 80.0), p=0.6),

    # --- Final Sizing: Preserve aspect ratio and pad to a square ---
    A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(
        min_height=640,
        min_width=640,
        border_mode=cv2.BORDER_CONSTANT,
        value=(30, 30, 30) # Darker padding color
    ),
])

# NEW: A dedicated pipeline to create a single, heavily blurred image.
HEAVY_BLUR_TRANSFORM = A.Compose([
    A.Blur(blur_limit=(19, 29), p=1.0), # Apply a strong, consistent blur
    A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(
        min_height=640,
        min_width=640,
        border_mode=cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    ),
])


def _process_single_image(
    image_path: Path,
    output_dir: Path,
    num_augmentations: int,
    transform: A.Compose
) -> int:
    """
    Worker function to process one image and generate its augmentations.
    Returns the number of images successfully generated.
    """
    try:
        base_name = image_path.stem

        # --- Save the original image with the special name ---
        original_output_path = output_dir / f"{base_name}_aug_999999.jpg"

        # Read the image once
        image = cv2.imread(str(image_path))
        if image is None:
            # print(f"Warning: Could not read {image_path}, skipping.")
            return 0

        # --- Generate N-1 regular augmentations ---
        # We reserve one slot for the special heavy blur augmentation.
        num_regular_augs = max(0, num_augmentations - 1)
        for i in range(num_regular_augs):
            augmented = transform(image=image)
            augmented_image = augmented['image']

            output_filename = f"{base_name}_aug_{i}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), augmented_image)

        # --- Generate the special heavy blur augmentation ---
        if num_augmentations > 0:
            blurred_augmented = HEAVY_BLUR_TRANSFORM(image=image)
            blurred_image = blurred_augmented['image']
            
            # Use the last available index for the blurred image filename
            blur_index = num_augmentations - 1
            output_filename_blur = f"{base_name}_aug_{blur_index}.jpg"
            output_path_blur = output_dir / output_filename_blur
            cv2.imwrite(str(output_path_blur), blurred_image)

        # Save the original reference image last
        cv2.imwrite(str(original_output_path), image)

        # Return total images created for this one source file
        return num_augmentations + 1

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return 0


def generate_augmentations(
    source_dir: str,
    output_dir: str,
    num_augmentations: int = 20,
    skip_existing: bool = True,
    batch_size: int = 100,
    max_workers: int = None,
    transform_pipeline: A.Compose = WEBCAM_LIKE_TRANSFORM
):
    """
    Generates augmentations for images in a source directory and saves them to an output directory.

    This function processes images in parallel using a thread pool for significant speed improvements.

    Args:
        source_dir (str): Path to the directory containing source images.
        output_dir (str): Path to the directory where augmented images will be saved.
        num_augmentations (int): Number of augmented versions to create for each source image.
        skip_existing (bool): If True, skips processing an image if its '_aug_999999.jpg'
                              version already exists in the output directory.
        batch_size (int): The number of images to group into a single processing batch.
        max_workers (int): The maximum number of threads to use. Defaults to the number of CPU cores.
        transform_pipeline (A.Compose): The Albumentations pipeline to use for augmentation.
                                        Defaults to a realistic webcam simulation.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not source_path.is_dir():
        print(f"Error: Source directory not found at {source_dir}")
        return

    print("Scanning source directory and filtering images...")

    # --- Gather all image paths to be processed ---
    image_files_to_process = []
    all_source_files = list(source_path.glob('*.*'))

    for file_path in tqdm(all_source_files, desc="Filtering existing"):
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            if skip_existing:
                # Check if the reference image exists, which implies it's already done
                check_file = output_path / f"{file_path.stem}_aug_999999.jpg"
                if not check_file.exists():
                    image_files_to_process.append(file_path)
            else:
                image_files_to_process.append(file_path)

    total_images = len(image_files_to_process)
    if total_images == 0:
        print("No new images to process. All images might already be augmented.")
        return

    print(f"Found {total_images} new images to augment.")

    # --- Create batches of image paths ---
    batches = [
        image_files_to_process[i:i + batch_size]
        for i in range(0, total_images, batch_size)
    ]

    total_generated_count = 0

    # --- Process batches in parallel using a ThreadPoolExecutor ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # The main progress bar will track completed batches
        with tqdm(total=len(batches), desc="Processing Batches") as batch_pbar:
            for batch in batches:
                # Submit all tasks for the current batch
                future_to_path = {
                    executor.submit(
                        _process_single_image,
                        img_path,
                        output_path,
                        num_augmentations,
                        transform_pipeline
                    ): img_path for img_path in batch
                }

                # This inner progress bar can optionally track progress within a batch
                # For very fast tasks, it might be too much overhead, but useful for diagnostics
                for future in as_completed(future_to_path):
                    try:
                        total_generated_count += future.result()
                    except Exception as e:
                        img_path = future_to_path[future]
                        print(f"An exception occurred while processing {img_path.name}: {e}")

                # Update the main progress bar after each batch is complete
                batch_pbar.update(1)

    print("\n--- Augmentation Complete ---")
    print(f"Processed {total_images} source images.")
    print(f"Generated a total of {total_generated_count} new image files in '{output_path}'.")


if __name__ == '__main__':
    # =================================================================================
    # This block allows you to run this script directly from the command line.
    # It also serves as an example of how to call the function from another script.
    # =================================================================================

    # --- Configuration ---
    # PLEASE UPDATE THESE PATHS
    SOURCE_IMAGES_FOLDER = '/mnt/e/Photos/TableStream/card_images'
    AUGMENTED_IMAGES_FOLDER = '/mnt/e/Photos/TableStream/augmented_images'

    # --- Check if paths are placeholder, if so, exit with instructions ---
    if 'path/to/your' in SOURCE_IMAGES_FOLDER or 'path/to/your' in AUGMENTED_IMAGES_FOLDER:
        print("="*60)
        print("ERROR: Please update the SOURCE_IMAGES_FOLDER and")
        print("       AUGMENTED_IMAGES_FOLDER variables in this script.")
        print("="*60)
    else:
        print("Starting image augmentation process...")
        generate_augmentations(
            source_dir=SOURCE_IMAGES_FOLDER,
            output_dir=AUGMENTED_IMAGES_FOLDER,
            num_augmentations=24,       # Generate 25 augmented + 1 original per card
            skip_existing=False,
            batch_size=100,             # Process 100 images before updating the main progress bar
            max_workers=os.cpu_count()  # Use all available CPU cores for max speed
        )

