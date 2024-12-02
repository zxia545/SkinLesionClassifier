import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import CustomResNet

# Define data directories
data_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data'
test_dir = os.path.join(data_dir, 'test')

# Define mean and std for normalization (using ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Validation and Test transforms
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load datasets
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
class_names = test_dataset.classes
num_classes = len(class_names)

# Define DataLoader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

# Load the saved model
model_path = 'custom_cnn_model.pth'
model = CustomResNet(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
print(f"Model loaded from '{model_path}'")

# Evaluation function
def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

    # Compute additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}')

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    plot_confusion_matrix(cm, class_names)

    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return total_loss, total_acc, precision, recall, f1, cm

# Plotting Confusion Matrix
def plot_confusion_matrix(cm, class_names, normalize=True, save_path='confusion_matrix_custom_cnn.png'):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to '{save_path}'")
    plt.show()

# Evaluating the model
evaluate_model(model, test_loader, device, class_names)
