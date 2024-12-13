import os

augmented_images_folder='/mnt/e/Photos/TableStream/augmented_images'
training_dataset_folder='/mnt/e/Photos/TableStream/training_images'

# for filename in os.listdir(augmented_images_folder):
#     if filename.endswith("aug_99.jpg"):
#         current_file_name = augmented_images_folder + "/" + filename
#         new_file_name = augmented_images_folder + "/" + filename.replace("aug_99", "aug_999999")
#         os.rename(current_file_name, new_file_name)



for subdir, dirs, files in os.walk(training_dataset_folder):
    for file in files:
        current_file_name = os.path.join(subdir, file)
        if current_file_name.endswith("aug_99.jpg"):
            new_file_name = current_file_name.replace("aug_99", "aug_999999")
            os.rename(current_file_name, new_file_name)