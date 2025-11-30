import os.path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils import class_weight

import time
from model_EfficientnetB2 import build_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import train_epochs, plt_loss, plt_accuracy, evaluation_metrics
import numpy as np


def load_data(dataset_path):
    """
     This function loads the dataset and applies random transformations to the training and validation images.
     Input:
      dataset_path : The path to the dataset folder
    """

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), transform=val_transform)
    
    y_train = [ y_train for X_train, y_train in train_dataset]

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    class_names = train_dataset.classes

    print(f'Dataset classes: {class_names}')
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'validate dataset size: {len(val_dataset)}')

    return train_loader, val_loader, y_train


def class_weight_(y):
    """
     This function returns the weights calculated for each class ,using the scikit-learn library, to be added to the loss function.
     Input :
      y: The annotation vector.
    """

    class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y
            )

    weights = [0., 0., 0.]
    M = [n for n in class_weights]
    j = 0
    for k in np.unique(y):
        weights[k] = M[j]
        j += 1
    
    return weights
    
    
def count_parameters(model):
    """
     This function calculates the number of trainable parameters in the model.
    """

    params = [p.numel() for p in model.parameters() if p.requires_grad]
    print(f'Number of trainable parameters: {sum(params)}')
    

def train_model(cnn, train_loader, val_loader, y_train, device, epochs, lr, out):
    """
     In this function we launch the training session.
     Inputs:
      cnn : The CNN model
      train_loader : The training dataset loader
      val_loader : The validation dataset loader
      y_train : The training annotation vector
      device : 'cuda' for GPU or 'cpu' for CPU
      epochs : Number of training epochs
      lr : Learning rate
      out : output path name (should be stored under ./runs folder)
    """
    
    weights = class_weight_(y_train)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device), label_smoothing=0.2)

    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    start_time = time.time()

    train_losses, test_losses, train_correct, test_correct = train_epochs(epochs, cnn, train_loader, criterion, optimizer, val_loader, scheduler, out, device)

    print(f'\nDuration: {(time.time() - start_time) / 60:.0f} minutes')

    return train_losses, test_losses, train_correct, test_correct


def args_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="number of epoch during training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate value")
    parser.add_argument("--dataset-path", type=str, default='../data_mono_clean/', help="dataset path")
    parser.add_argument("--out-dir", type=str, default='exp1', help="output path name should be stored under ./runs folder")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = 'cpu'
        
    print('device: ', device)

    opt = args_opt()
    epochs = opt.epochs
    lr = opt.lr
    dataset_path = opt.dataset_path
    out = opt.out_dir

    # Load the dataset
    train_loader, val_loader, y_train = load_data(dataset_path)

    # Initialize the CNN model
    cnn_last = build_model().to(device)
    
    count_parameters(cnn_last)

    # Lunch the training session
    train_losses, test_losses, train_correct, test_correct = train_model(cnn_last, train_loader, val_loader, y_train, device, epochs, lr, out)

    # Save the loss and accuracy curves
    plt_loss(train_losses, test_losses, './runs/' + out + '/loss.png')
    plt_accuracy(train_correct, test_correct, './runs/' + out + '/accuracy.png')

    # Calculate the following metrics after training : accuracy, precision, recall, f1 score.
    # Save the confusion matrix
    accuracy, precision, recall, fscore = evaluation_metrics('./runs/' + out + '/best_model.pt', val_loader, [0, 1, 2], './runs/' + out + '/', device)
    print(f'accuracy = {accuracy} Precision = {precision} Recall = {recall} fscore = {fscore}')
