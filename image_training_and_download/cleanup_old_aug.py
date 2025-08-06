import os
import re
from pathlib import Path

def cleanup_old_augmentations(target_dir: str, dry_run: bool = True):
    """
    Deletes specific augmented images from a target directory.

    This script is designed to remove files with names like '..._aug_0.jpg', 
    '..._aug_1.jpg', etc., while preserving files that are either the source 
    reference ('..._aug_999999.jpg') or have non-numeric IDs like MongoDB hashes 
    ('..._aug_60d5f1b2c3d4e5f6g7h8i9j0.jpg').

    Args:
        target_dir (str): The path to the directory containing the images to clean up.
        dry_run (bool): If True, the script will only print the files it intends to 
                        delete without actually deleting them. If False, it will 
                        perform the deletion. Defaults to True for safety.
    """
    target_path = Path(target_dir)
    if not target_path.is_dir():
        print(f"Error: Directory not found at '{target_dir}'")
        return

    # Regex to find filenames with the pattern '_aug_' followed by one or more digits.
    # This specifically targets the numbered augmentations.
    # It captures the number part in a group.
    pattern = re.compile(r"_aug_(\d+)\.jpg$")

    files_to_delete = []
    
    print("Scanning directory for augmentations to delete...")

    for file_path in target_path.glob('*.jpg'):
        match = pattern.search(file_path.name)
        
        # Check if the filename matches the '_aug_NUMBER.jpg' pattern
        if match:
            # Extract the number from the matched group
            aug_number = int(match.group(1))
            
            # We want to delete it ONLY if the number is NOT 999999
            if aug_number != 999999:
                files_to_delete.append(file_path)

    if not files_to_delete:
        print("No numbered augmentations (excluding _aug_999999) found to delete.")
        return

    print("-" * 50)
    if dry_run:
        print(f"DRY RUN: Found {len(files_to_delete)} files that would be deleted.")
        print("To delete these files, run the script with `PERFORM_DELETION = True`.")
        # Print the first 20 files as a sample
        for f in files_to_delete[:20]:
            print(f"  - Would delete: {f.name}")
        if len(files_to_delete) > 20:
            print(f"  - ... and {len(files_to_delete) - 20} more.")
    else:
        print(f"DELETING: Found {len(files_to_delete)} files to remove.")
        deleted_count = 0
        for f in files_to_delete:
            try:
                f.unlink()
                deleted_count += 1
                # To avoid flooding the console, we can print progress periodically
                if deleted_count % 100 == 0:
                    print(f"  ... deleted {deleted_count}/{len(files_to_delete)} files")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        
        print(f"\nDeletion complete. Removed {deleted_count} files.")
    print("-" * 50)


if __name__ == '__main__':
    # =================================================================================
    # Configuration
    # =================================================================================

    # 1. SET THE PATH to the folder you want to clean up.
    AUGMENTED_IMAGES_FOLDER = '/mnt/e/Photos/TableStream/augmented_images'

    # 2. SET THIS TO FALSE TO ACTUALLY DELETE FILES.
    # For safety, this is True by default, which only shows what would be deleted.
    PERFORM_DELETION = True # <-- CHANGE THIS TO True TO DELETE FILES

    # =================================================================================
    
    if 'path/to/your' in AUGMENTED_IMAGES_FOLDER:
        print("="*60)
        print("ERROR: Please update the AUGMENTED_IMAGES_FOLDER variable in this script.")
        print("="*60)
    else:
        if not PERFORM_DELETION:
            print("--- Running in DRY RUN mode. No files will be deleted. ---")
            cleanup_old_augmentations(AUGMENTED_IMAGES_FOLDER, dry_run=True)
        else:
            # Add a final confirmation step to prevent accidental deletion
            confirm = input(
                f"You are about to permanently delete files from '{AUGMENTED_IMAGES_FOLDER}'.\n"
                "This action cannot be undone.\n"
                "Are you sure you want to continue? (yes/no): "
            )
            if confirm.lower() == 'yes':
                print("--- Running in DELETION mode. Files will be removed. ---")
                cleanup_old_augmentations(AUGMENTED_IMAGES_FOLDER, dry_run=False)
            else:
                print("Deletion cancelled by user.")
