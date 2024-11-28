import os
import pandas as pd
import shutil
import random

# Paths to the dataset and CSV file
dataset_dir = '/home/zxia545/_Code/UT-Austin_projects/csml_final_project/dataset/ham10000/Labeled_Data'
output_base_dir = '/home/zxia545/_Code/UT-Austin_projects/csml_final_project/code_repo/split_data'

# Create output directories for train, validation, and test splits
splits = ['train', 'validation', 'test']
for split in splits:
    for class_name in os.listdir(dataset_dir):
        os.makedirs(os.path.join(output_base_dir, split, class_name), exist_ok=True)
# Define split ratios
train_ratio = 0.75
validation_ratio = 0.1
test_ratio = 0.15

# Iterate over each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_path):
        # Get all image files in the current class directory
        images = [img for img in os.listdir(class_path) if img.endswith('.jpg') or img.endswith('.png')]
        
        # Shuffle the images to ensure randomness
        random.shuffle(images)
        
        # Calculate split indices
        total_images = len(images)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * validation_ratio)
        
        # Split the images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Helper function to copy images to respective split folders
        def copy_images(image_list, split_name):
            for image in image_list:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(output_base_dir, split_name, class_name, image)
                shutil.copy2(src_path, dst_path)
        
        # Copy images to respective directories
        copy_images(train_images, 'train')
        copy_images(val_images, 'validation')
        copy_images(test_images, 'test')

print("Dataset successfully split into train, validation, and test sets.")

