import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler

from models import CustomResNet
from train import train_model
from evaluate import evaluate_model
from utils import plot_training_history
import tqdm

# Define data directories
data_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')

# Define mean and std for normalization (using ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    # Geometric Transformations
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),

    # Color Transformations
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    # Convert to Tensor
    transforms.ToTensor(),

    # Noise Addition (apply with 50% probability)
    transforms.RandomApply([
        transforms.Lambda(lambda img: torch.clamp(img + torch.randn_like(img) * 0.01, 0, 1))
    ], p=0.5),

    # Normalize
    transforms.Normalize(mean, std),
])

# # Data augmentation and normalization for training
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(90),
#     transforms.ColorJitter(
#         brightness=0.3,
#         contrast=0.3,
#         saturation=0.3,
#         hue=0.1
#     ),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])


# Validation and Test transforms
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)


# # Compute class weights
# labels = [label for _, label in train_dataset.imgs]
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# # Loss function with class weights
# criterion = nn.CrossEntropyLoss(weight=class_weights)


# Define DataLoaders
batch_size = 16
num_workers = 8  # Adjust based on your system

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

# # Compute sample weights
# sample_weights = [class_weights[label] for _, label in train_dataset.imgs]

# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# # Update DataLoader
# dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

# Class names
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")



# Model Definition
model = CustomResNet(num_classes=num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training the model
num_epochs = 25
model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(
    model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs)

# Plot training history
plot_training_history(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

# Evaluating the model
evaluate_model(model, dataloaders, device, class_names)
