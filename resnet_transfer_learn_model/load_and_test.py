import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

# Load saved model
def load_model(model_path, num_classes, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from '{model_path}'")
    return model

# Define evaluation data transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Load datasets
data_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data'
test_dir = os.path.join(data_dir, 'test')
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
class_names = test_dataset.classes
num_classes = len(class_names)

# Define DataLoader
batch_size = 32
num_workers = 4  # Adjust based on your system
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load the model
model_path = 'resnet_transfer_learning_model.pth'
model = load_model(model_path, num_classes, device)

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
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

    # Compute metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return total_loss, total_acc, precision, recall, f1, cm

# Criterion definition
criterion = torch.nn.CrossEntropyLoss()

# Evaluating the model
test_loss, test_acc, test_precision, test_recall, test_f1, cm = evaluate_model(model, test_loader, criterion, device)

# Plotting Confusion Matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix - Test Set', normalize=False, save_path='confusion_matrix_resnet.png'):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to '{save_path}'")
    plt.show()

plot_confusion_matrix(cm, class_names, normalize=True, save_path='confusion_matrix_resnet.png')

# Plotting Test Metrics
def plot_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1):
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [test_loss, test_acc, test_precision, test_recall, test_f1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Test Metrics')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('test_metrics_resnet.png')
    print("Test metrics saved to 'test_metrics_resnet.png'")
    plt.show()

plot_test_metrics(test_loss, test_acc, test_precision, test_recall, test_f1)
