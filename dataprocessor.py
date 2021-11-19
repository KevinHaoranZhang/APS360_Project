import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import torch
import string
import subprocess

# Data augmentation dictonary
DATA_AUGMENTATION = {
    "CENTER_CROP": transforms.CenterCrop(size=0.5),
    "GRAY_SCALE" : transforms.Grayscale(num_output_channels=1),
    "HORIZONTAL_FLIP" : transforms.RandomHorizontalFlip(p=0.5),
    "RANDOM_ROTATION": transforms.RandomRotation(degrees=15),
    "RESIZE": transforms.Resize((28, 28)),
    "TO_TENSOR" : transforms.ToTensor(),
    "VERTICAL_FLIP" : transforms.RandomVerticalFlip(p=0.5),
}

def get_dataset_digits(transform=transforms.Compose([transforms.ToTensor()])):
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)
    mnist_train.targets[mnist_train.targets >= 0] = mnist_train.targets[mnist_train.targets >= 0] + 20
    mnist_test.targets[mnist_test.targets >= 0] = mnist_test.targets[mnist_test.targets >= 0] + 20
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])
    return mnist_train, mnist_val, mnist_test

def get_dataset_letters(transform=transforms.Compose([transforms.ToTensor()])):
    emnist_train = datasets.EMNIST('data', split="letters", train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST('data', split="letters", train=False, download=True, transform=transform)
    emnist_train.targets[emnist_train.targets > 0] = emnist_train.targets[emnist_train.targets > 0] + 29
    emnist_test.targets[emnist_test.targets > 0] = emnist_test.targets[emnist_test.targets > 0] + 29
    emnist_train, emnist_val = torch.utils.data.random_split(emnist_train, [104000, 20800])
    return emnist_train, emnist_val, emnist_test

def get_dataset_symbols(transform=transforms.Compose([transforms.ToTensor()])):
    symbol_train = datasets.ImageFolder("math_symbol_data/training_dataset", transform=transform)
    symbol_val = datasets.ImageFolder("math_symbol_data/validation_dataset", transform=transform)
    symbol_test = datasets.ImageFolder("math_symbol_data/testing_dataset", transform=transform)
    return symbol_train, symbol_val, symbol_test

# Parameters
# batch_size: the batch size for dataloader
# display: to show data samples
# extract_symbol: to extract data from math_symbol_data.zip, first time use -> set to True, later user -> set to False
# transform_digits: data augmentation on digits dataset
# transform_letters: data augmentation on letters dataset
# transform_symbols: data augmentation on symbols dataset
def get_dataset_loaders(batch_size=64, display=False, extract_symbol=False, transform_digits=transforms.Compose([transforms.ToTensor()]), transform_letters=transforms.Compose([transforms.ToTensor()]), transform_symbols=transforms.Compose([transforms.ToTensor()])):
    print(f"[DataProcessing] Initiate: batch size ({batch_size})")
    print(f"[DataProcessing][Data Augmentation][Digits] ({transform_digits})")
    print(f"[DataProcessing][Data Augmentation][Letters] ({transform_letters})")
    print(f"[DataProcessing][Data Augmentation][Symbols] ({transform_symbols})")
    print("[DataProcessing] Loading digits from MNIST dataset")
    mnist_train, mnist_val, mnist_test = get_dataset_digits(transform=transform_digits)
    print("[DataProcessing] Loading letters from EMNIST dataset")
    emnist_train, emnist_val, emnist_test = get_dataset_letters(transform=transform_letters)
    print("[DataProcessing] Loading symbols from Math Symbol dataset")
    if extract_symbol:
        print("[DataProcessing] Unzipping math symbols from Math Symbol dataset")
        subprocess.Popen(["sh", "data_extractor.sh"]).wait()
    symbol_train, symbol_val, symbol_test = get_dataset_symbols(transform=transform_symbols)

    # Display data samples
    if display:
        # Display digits data samples
        digits_label = list(range(20, 30))
        digits_display_dic = {}
        for i in mnist_train:
            if i[1] in digits_label:
                digits_display_dic[i[1]] = i[0]
                digits_label.remove(i[1])
            if len(digits_label) == 0:
                break
        fig = plt.figure(figsize=(10, 5))
        for i in sorted(digits_display_dic):
            ax = fig.add_subplot(2, 10/2, i+1-20, xticks=[], yticks=[])
            plt.imshow(digits_display_dic[i].reshape(28, 28))
            ax.set_title(i - 20)
        plt.show()
        # Display letters data samples
        letters_label = list(range(30, 56))
        letters_display_dic = {}
        alphabet_decoder = list(string.ascii_uppercase)
        for i in emnist_train:
            if i[1] in letters_label:
                letters_display_dic[i[1]] = i[0]
                letters_label.remove(i[1])
            if len(letters_label) == 0:
                break
        fig = plt.figure(figsize=(26, 5))
        for i in sorted(letters_display_dic):
            ax = fig.add_subplot(2, 26/2, i+1-30, xticks=[], yticks=[])
            plt.imshow(letters_display_dic[i].reshape(28, 28))
            ax.set_title(alphabet_decoder[i-30])
        plt.show()
        # Display symbols data samples
        symbols_label = list(range(0, 5))
        symbols_display_dic = {}
        symbols_decoder = ["add", "div", "eq", "lb", "rb", "sub"]
        for i in symbol_train:
            if i[1] in symbols_label:
                symbols_display_dic[i[1]] = i[0]
                symbols_label.remove(i[1])
            if len(symbols_label) == 0:
                break
        fig = plt.figure(figsize=(6, 5))
        for i in sorted(symbols_display_dic):
            ax = fig.add_subplot(2, 6/2, i+1, xticks=[], yticks=[])
            plt.imshow(symbols_display_dic[i].reshape(28, 28))
            ax.set_title(symbols_decoder[i])
        plt.show()

    # Concatenate dataset together into training, validation, and testing datasets
    train_dataset = torch.utils.data.ConcatDataset([mnist_train, emnist_train, symbol_train])
    val_dataset = torch.utils.data.ConcatDataset([mnist_val, emnist_val, symbol_val])
    test_dataset = torch.utils.data.ConcatDataset([mnist_test, emnist_test, symbol_test])
    # Load the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    print("[DataProcessing] Complete")
    print("----------Data Processing Report----------")
    digits_dataset_total = len(mnist_train) + len(mnist_val) + len(mnist_test)
    letters_dataset_total = len(emnist_train) + len(emnist_val) + len(emnist_test)
    symbols_dataset_total = len(symbol_train) + len(symbol_val) + len(symbol_test)
    print(f"Digits dataset: training size ({len(mnist_train)} {len(mnist_train) / digits_dataset_total * 100:.2f}%) validation size ({len(mnist_val)} {len(mnist_val) / digits_dataset_total * 100:.2f}%) testing size ({len(mnist_test)} {len(mnist_test) / digits_dataset_total * 100:.2f}%)")
    print(f"Letters dataset: training size ({len(emnist_train)} {len(emnist_train) / letters_dataset_total * 100:.2f}%) validation size ({len(emnist_val)} {len(emnist_val) / letters_dataset_total * 100:.2f}%) testing size ({len(emnist_test)} {len(emnist_test) / letters_dataset_total * 100:.2f}%)")
    print(f"Symbols dataset: training size ({len(symbol_train)} {len(symbol_train) / symbols_dataset_total * 100:.2f}%) validation size ({len(symbol_val)} {len(symbol_val) / symbols_dataset_total * 100:.2f}%) testing size ({len(symbol_test)} {len(symbol_test) / symbols_dataset_total * 100:.2f}%)")
    digits_label_mapping = dict(zip(range(0, 10), range(20, 30)))
    letters_label_mapping = dict(zip(list(string.ascii_uppercase), range(30, 56)))
    symbols_label_mapping = dict(zip(["add", "div", "eq", "lb", "rb", "sub"], range(0, 7)))
    print(f"Digits Label Mapping {digits_label_mapping}")
    print(f"Letters Label Mapping {letters_label_mapping}")
    print(f"Symbols Label Mapping {symbols_label_mapping}")
    print("------------------------------------------")
    return train_loader, val_loader, test_loader

