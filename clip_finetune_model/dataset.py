# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, classes, processor, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            classes (list): List of class names.
            processor (CLIPProcessor): Pretrained CLIP processor.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.classes = classes
        self.processor = processor
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            class_dir = os.path.join(self.image_dir, cls)
            if not os.path.isdir(class_dir):
                continue  # Skip if the class directory does not exist
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image tensor.
            label (int): Class label.
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
