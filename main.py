#!/bin/usr/python3
from typing import Tuple

from torch.nn.modules.module import T
from torchvision import transforms
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

def train(model, training_loader, validation_loader, batch_size=64, learning_rate=0.001, num_epochs=1):
    # Softmax activation applied internally
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    training_iterations, training_loss, training_accuracy, validation_accuracy = list(), list(), list(), list()

    # training
    for epoch in range(num_epochs):
        for images, labels in iter(training_loader):
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

def get_data_stats(train_loader, val_loader, test_loader):
    label_train_list = [0] * 56
    label_val_list = [0] * 56
    label_test_list = [0] * 56
    for images, labels in iter(train_loader):
        label_train_list[int(labels)] += 1
    for images, labels in iter(val_loader):
        label_val_list[int(labels)] += 1
    for images, labels in iter(test_loader):
        label_test_list[int(labels)] += 1

    label_train_list = [i for i in label_train_list if i != 0]
    label_val_list = [i for i in label_train_list if i != 0]
    label_test_list = [i for i in label_train_list if i != 0]
    lable_list = ["add", "div", "eq", "lb", "rb", "sub", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    print(dict(zip(lable_list, label_train_list)))
    print(dict(zip(lable_list, label_val_list)))
    print(dict(zip(lable_list, label_test_list)))

if __name__ == "__main__":
    simple_model = simplemodel.Model()
    symbols_augmentation = []
    symbols_augmentation.append(dataprocessor.DATA_AUGMENTATION["RESIZE"])
    symbols_augmentation.append(dataprocessor.DATA_AUGMENTATION["INVERT_COLOR"])
    symbols_augmentation.append(dataprocessor.DATA_AUGMENTATION["GRAY_SCALE"])
    symbols_augmentation.append(dataprocessor.DATA_AUGMENTATION["TO_TENSOR"])
    letters_augmentation = []
    letters_augmentation.append(dataprocessor.DATA_AUGMENTATION["HORIZONTAL_FLIP"])
    letters_augmentation.append(dataprocessor.DATA_AUGMENTATION["RIGHT_ROTATION"])
    letters_augmentation.append(dataprocessor.DATA_AUGMENTATION["TO_TENSOR"])
    # dataprocessor.extract_dataset()
    # train_loader, val_loader, test_loader = dataprocessor.get_all_dataset_loaders(1, True,
    #                                                         transform_symbols=transforms.Compose(symbols_augmentation),
    #                                                         transform_letters=transforms.Compose(letters_augmentation))
    train_loader, val_loader, test_loader = dataprocessor.get_classify_dataset_loaders(64,
                                                            transform_symbols=transforms.Compose(symbols_augmentation),
                                                            transform_letters=transforms.Compose(letters_augmentation))
    # train_loader, val_loader, test_loader = dataprocessor.get_digits_dataset_loader(1)
    # train_loader, val_loader, test_loader = dataprocessor.get_letters_dataset_loader(1, transform_letters=transforms.Compose(letters_augmentation))
    # train_loader, val_loader, test_loader = dataprocessor.get_symbols_dataset_loader(1, transform_symbols=transforms.Compose(symbols_augmentation))
    # train_label = []
    # for images, labels in iter(train_loader):
    #     if int(labels) not in train_label:
    #         train_label.append(int(labels))
    # val_label = []
    # for images, labels in iter(val_loader):
    #     if int(labels) not in val_label:
    #         val_label.append(int(labels))
    # test_label = []
    # for images, labels in iter(test_loader):
    #     if int(labels) not in test_label:
    #         test_label.append(int(labels))
    # print(train_label)
    # print(val_label)
    # print(test_label)

    # get_data_stats(train_loader, val_loader, test_loader)
    train(simple_model, train_loader, val_loader, batch_size=64, learning_rate=0.01, num_epochs=6)

