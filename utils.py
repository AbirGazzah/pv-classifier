import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

from model_EfficientnetB2 import build_model


def train_epochs(epochs, cnn, train_loader, criterion, optimizer, val_loader, scheduler, exp_name, device):
    """
     This function defines the training loop.
     Inputs:
      epochs: Number of training epochs
      cnn: The CNN model
      train_loader: The training dataset loader
      criterion: The loss function
      optimizer: The optimizer
      val_loader: The validation dataset loader
      scheduler: Learning rate scheduler
      exp_name: output path name
      device: 'cuda' for GPU or 'cpu' for CPU
    """

    best_accuracy = 0
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    print('*************** start training ********************')
    for i in range(epochs):

        trn_corr = 0
        total_loss_train = 0
        total_samples = 0

        cnn.train()
        for X_train, y_train in train_loader:

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            batch_size = y_train.size(0)
            total_samples += batch_size

            # forward propagation
            y_pred = cnn(X_train)

            # Calculate the loss value for each batch
            loss = criterion(y_pred, y_train)
            total_loss_train += loss.item()

            # calculate the number of correct predictions for each batch
            predicted = torch.max(y_pred.data, 1)[1]
            trn_corr += (predicted == y_train).sum().item()

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate the training loss and accuracy for each epoch
        total_loss_train = total_loss_train / len(train_loader)
        train_losses.append(total_loss_train)
        train_accuracy = trn_corr * 100 / total_samples
        train_correct.append(train_accuracy)

        total_samples_val = 0
        total_loss_val = 0
        tst_corr = 0

        cnn.eval()
        with torch.no_grad():
            for X_test, y_test in val_loader:

                X_test = X_test.to(device)
                y_test = y_test.to(device)

                batch_size = y_test.size(0)
                total_samples_val += batch_size

                y_val = cnn(X_test)
                loss_val = criterion(y_val, y_test)
                total_loss_val += loss_val.item()

                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum().item()

        total_loss_val = total_loss_val / len(val_loader)
        test_losses.append(total_loss_val)
        val_accuracy = tst_corr * 100 / total_samples_val
        test_correct.append(val_accuracy)

        # track the highest validation accuracy and update the saved checkpoint accordingly
        if val_accuracy > best_accuracy:
            torch.save(cnn.state_dict(), './runs/' + exp_name + '/best_model.pt')
            best_accuracy = val_accuracy

        # Apply the learning rate scheduling : dynamic adjustment of the learning rate
        scheduler.step(val_accuracy)

        print(f'epoch: {i:2}  loss_train: {total_loss_train:.5f}  loss_val: {total_loss_val:.5f}  train_accuracy: {train_accuracy :.3f}%  val_accuracy: {val_accuracy :.3f}%')

    return train_losses, test_losses, train_correct, test_correct


def evaluation_metrics(model_path, test_loader, classes, cm_path, device):
    """
     This function calculates the following metrics : accuracy, precision, recall, f1 score, and save the confusion matrix.
     Inputs:
      model_path: The CNN model path
      test_loader: The validation dataset loader
      classes: List of class indices
      cm_path: Path to save the Confusion matrix
      device: 'cuda' for GPU or 'cpu' for CPU
    """

    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tst_corr = 0
    total_samples_val = 0
    pred_labels = []
    gtruth_labels = []

    for X_test, y_test in test_loader:
        X_test = X_test.to('cuda')
        y_test = y_test.to('cuda')

        batch_size = y_test.size(0)
        total_samples_val += batch_size

        y_val = model(X_test)

        predicted = torch.max(y_val.data, 1)[1]
        tst_corr += (predicted == y_test).sum().item()

        pred_labels.append(predicted.cpu().numpy())
        gtruth_labels.append(y_test.cpu().numpy())

    accuracy = tst_corr * 100 / total_samples_val
    gtruth_labels = np.concatenate(gtruth_labels)
    pred_labels = np.concatenate(pred_labels)
    precision, recall, fscore, _ = precision_recall_fscore_support(gtruth_labels, pred_labels, average='macro')

    cm = confusion_matrix(gtruth_labels, pred_labels, labels = classes)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix =cm)

    disp_cm.plot()
    plt.title('Confusion matrix')
    plt.savefig(cm_path + 'confusion_matrix.png')
    plt.close()

    return accuracy, precision, recall, fscore

def plt_loss(train_losses, test_losses, loss_path):
    """
     This function plots the loss curves and saves the results.
    """

    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='validation loss')
    plt.title('Loss at each epoch')
    plt.legend()
    plt.savefig(loss_path)
    plt.close()

def plt_accuracy(train_correct, test_correct, accuracy_path):
    """
     This function plots the accuracy curves and saves the results.
    """

    plt.plot(train_correct, label='training accuracy')
    plt.plot(test_correct, label='validation accuracy')
    plt.title('Accuracy at each epoch')
    plt.legend()
    plt.savefig(accuracy_path)
    plt.close()
