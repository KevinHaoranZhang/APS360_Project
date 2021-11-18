#!/bin/usr/python3
from typing import Tuple

from torch.nn.modules.module import T
import dataprocessor
import simplemodel

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def get_accuracy(model, data_loader):
  correct, total, accuracy = 0, 0, 0
  for images, labels in data_loader:
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    output = model(images)
    # Select maximum score
    score = output.max(1, keepdim=True)[1]
    correct += score.eq(labels.view_as(score)).sum().item()
    total += images.shape[0]
  accuracy = correct / total
  return accuracy

def train(model, training_dataset, validation_dataset, batch_size=64, learning_rate=0.001, num_epochs=1):
  training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
  validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
  # Softmax activation applied internally
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  training_iterations, training_loss, training_accuracy, validation_accuracy = list(), list(), list(), list()

  # training
  for epoch in range(num_epochs):

    for images, labels in iter(training_loader):
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
      # Forward pass, backward pass, and optimize
      output = model(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
    # Calculate the statistics
    training_iterations.append(epoch)
    training_loss.append(float(loss)/batch_size)
    training_accuracy.append(get_accuracy(model, training_loader))
    validation_accuracy.append(get_accuracy(model, validation_loader))
    # Display the statistics
    print(f"epoch number ({epoch+1}) training loss training ({training_loss[epoch]*100.0:.2f}%) accuracy ({training_accuracy[epoch]*100.0:.2f}%) validation accuracy ({validation_accuracy[epoch]*100.0:.2f}%)")
    # Save the current model (checkpoint) to a file
    model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, batch_size, learning_rate, epoch)
    torch.save(model.state_dict(), model_path)

  # Plot the training curve
  plt.title("Training Loss")
  plt.plot(training_iterations, training_loss, label="Training")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.show()

  plt.title("Training Accuracy")
  plt.plot(training_iterations, training_accuracy, label="Training")
  plt.plot(training_iterations, validation_accuracy, label="Validation")
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.legend(loc='best')
  plt.show()

  # Display final training and validation accuracy
  print(f"Final Training Accuracy: {training_accuracy[-1]*100.0:.2f}%")
  print(f"Final Validation Accuracy: {validation_accuracy[-1]*100.0:.2f}%")

if __name__ == "__main__":
    simple_model = simplemodel.Model()
    mnist_train, mnist_val, mnist_test = dataprocessor.get_dataset_digits()
    emnist_train, emnist_val, emnist_test = dataprocessor.get_dataset_letters()
    train_dataset = torch.utils.data.ConcatDataset([mnist_train, emnist_train])
    val_dataset = torch.utils.data.ConcatDataset([mnist_val, emnist_val])
    train(simple_model, train_dataset, val_dataset, batch_size=64, learning_rate=0.01, num_epochs=6)