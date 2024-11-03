import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def train_routine(
    model,
    train_loader,
    optimizer,
    loss_fn,
    num_epochs,
    verbose=False
):
    # model.to(device)

    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    for epoch in range(num_epochs):

        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for batch in train_loader:
            inputs, labels, _ = batch
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(probabilities, 1)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        # Calculate accuracy and loss for the epoch
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        test_corrects = 0
        test_total = 0

        all_preds = []
        all_labels = []
        all_prob = []

        # with torch.no_grad():  # Disable gradient computation for validation
        #     for test_batch in test_loader:
        #         test_inputs, test_labels, _ = test_batch

        #         test_outputs = model(test_inputs)
        #         test_probabilities = F.softmax(test_outputs)
        #         _, test_preds = torch.max(test_probabilities, 1)

        #         test_loss += loss_fn(test_outputs, test_labels).item() * test_inputs.size(0)
        #         test_corrects += torch.sum(test_preds == test_labels.data)
        #         test_total += test_labels.size(0)

        #         all_preds.extend(test_preds.cpu().numpy())
        #         all_prob.extend(test_probabilities.cpu().numpy())
        #         all_labels.extend(test_labels.cpu().numpy())

        # test_epoch_loss = test_loss / test_total
        # test_epoch_acc = test_corrects.double() / test_total

        # test_losses.append(test_epoch_loss)
        # test_acc.append(test_epoch_acc)
        # all_prob = np.array(all_prob)

        # y_true_one_hot = np.zeros_like(all_prob)
        # y_true_one_hot[np.arange(len(all_labels)), all_labels] = 1
        # brier_score = np.mean(np.sum((all_prob - y_true_one_hot) ** 2, axis=1))


        # print(f'Epoch {epoch+1}/{num_epochs} | Test Acc {test_epoch_acc} | Test Loss {test_epoch_loss} | Brier {brier_score}')
        # , all_preds, all_prob, all_labels
    return model

def get_confusion_matrix(all_labels, all_preds):
  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Draw", "Home", "Away"], yticklabels=["Draw", "Home", "Away"])
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  plt.title(f'Confusion Matrix')
  plt.show()

def get_brier_score(all_labels, all_prob):
  y_true_one_hot = np.zeros_like(all_prob)
  y_true_one_hot[np.arange(len(all_labels)), all_labels] = 1
  brier_score = np.mean(np.sum((all_prob - y_true_one_hot) ** 2, axis=1))
  print(f'| Brier Score {brier_score}')


def calibration_curve(all_labels, all_prob):
  # Binarize the labels for one-vs-rest approach
  y_true_bin = label_binarize(all_labels, classes=[0, 1, 2])

  # Plot precision-recall curve for each class
  plt.figure(figsize=(8, 6))
  for i in range(3):  # Assuming three classes
      precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_prob[:, i])
      plt.plot(recall, precision, label=f'Class {i}')

  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision-Recall Curve (One-vs-Rest)")
  plt.legend()
  plt.show()

