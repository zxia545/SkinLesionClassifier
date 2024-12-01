import os
import random
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Directory paths
input_dir = "/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/train"  # Example: augment the training set
output_dir = "/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/augmented_train"

# Create output directories for augmented images if not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define transformations for data augmentation
# Define the augmentation transforms
augmentation_transforms = transforms.Compose([
    # Convert to tensor first (necessary for tensor-based operations like noise)
    transforms.ToTensor(),

    # Geometric Transformations
    transforms.RandomRotation(degrees=30),  # Rotate randomly between -30° and 30°
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally with a 50% probability
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translate image up to 10% of image size

    # Scaling and Cropping (Zoom in/out slightly and crop to retain original dimensions)
    transforms.RandomResizedCrop(240, scale=(0.8, 1.0)),  # Resize to 240*240 and zoom between 80% and 100%

    # Color Transformations
    transforms.ColorJitter(
        brightness=random.uniform(0.3, 0.5),   # Adjust brightness slightly
        contrast=random.uniform(0.1, 0.2),     # Reduce contrast range
        saturation=random.uniform(0.8, 1.2),   # Moderate saturation range
        hue=0.05        # Small hue shift for subtle changes
    ),


    # Noise Addition (after converting to tensor)
    transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.01 if random.random() > 0.5 else img),  # Gaussian Noise

    # Elastic Deformations (no specific library, could use external ones like `albumentations` for more advanced)
    transforms.Lambda(lambda img: img),  # Placeholder for Elastic Deformations
])

# Augment each image in the input directory
num_augmentations = 5  # Number of augmentations per original image

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if os.path.isdir(class_path):
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path.endswith('.jpg') or img_path.endswith('.png'):
                img = Image.open(img_path).convert('RGB')

                # Apply augmentations multiple times to create new augmented images
                for i in range(num_augmentations):
                    augmented_img = augmentation_transforms(img)
                    augmented_img_pil = transforms.ToPILImage()(augmented_img)
                    augmented_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    augmented_img_pil.save(os.path.join(output_class_path, augmented_img_name))

print("Data augmentation completed.")
