import time
import copy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import save_model

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    patience = 7
    early_stopping_counter = 0
    best_epoch = 0


    since = time.time()
    writer = SummaryWriter('runs/skin_lesion_experiment_custom_cnn')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # TensorBoard logging
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # Deep copy the model if it has better accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Save history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

        print()

        # # Early Stopping Check
        # if phase == 'val':
        #     if epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_wts = copy.deepcopy(model.state_dict())
        #         early_stopping_counter = 0
        #         best_epoch = epoch
        #     else:
        #         early_stopping_counter += 1
        #         if early_stopping_counter >= patience:
        #             print(f'Early stopping at epoch {epoch+1}')
        #             model.load_state_dict(best_model_wts)
        #             writer.close()
        #             save_model(model, 'custom_resnet_model.pth')
        #             return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed//60)}m {int(time_elapsed%60)}s')
    print(f'Best Validation Accuracy: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()

    # Save the final model
    save_model(model, 'custom_cnn_model.pth')

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history
