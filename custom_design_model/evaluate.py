import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_confusion_matrix

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0

    all_preds = []
    all_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader['test'].dataset)
    total_acc = running_corrects.double() / len(dataloader['test'].dataset)

    print(f'Test Loss: {total_loss:.4f} Acc: {total_acc:.4f}')

    # Compute additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1:.4f}')

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    plot_confusion_matrix(cm, class_names)

    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
