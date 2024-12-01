# train.py

import time
import copy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10, patience=5):
    """
    Trains the model and returns the best model based on validation accuracy.

    Args:
        model (nn.Module): The model to train.
        dataloaders (dict): Dictionary containing 'train' and 'validation' DataLoaders.
        dataset_sizes (dict): Dictionary containing sizes of 'train' and 'validation' datasets.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        model (nn.Module): The best trained model.
        train_loss_history (list): Training loss history.
        train_acc_history (list): Training accuracy history.
        val_loss_history (list): Validation loss history.
        val_acc_history (list): Validation accuracy history.
    """
    since = time.time()
    writer = SummaryWriter('runs/blip_finetune')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # Lists to store history
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # [batch_size, num_classes]
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # TensorBoard logging
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # Save history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                # Deep copy the model if validation accuracy improves
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print('Early stopping!')
                        model.load_state_dict(best_model_wts)
                        writer.close()
                        time_elapsed = time.time() - since
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best Validation Accuracy: {best_acc:.4f}')
                        return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history
