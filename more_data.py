#!/bin/usr/python3
from torchvision import datasets
from torchvision import transforms
import torch
from torchvision.utils import save_image
if __name__ == "__main__":
    DATA_AUGMENTATION = {
        "CENTER_CROP": transforms.CenterCrop(size=0.5),
        "GRAY_SCALE" : transforms.Grayscale(num_output_channels=1),
        "HORIZONTAL_FLIP" : transforms.RandomHorizontalFlip(p=1),
        "RIGHT_ROTATION": transforms.RandomRotation(degrees=[89,91]),
        "RESIZE": transforms.Resize((28, 28)),
        "TO_TENSOR" : transforms.ToTensor(),
        "VERTICAL_FLIP" : transforms.RandomVerticalFlip(p=1),
        "INVERT_COLOR": transforms.RandomInvert(p=1),
        "RANDOM_ROTATION": transforms.RandomRotation(degrees=[-15,15]),
        "RANDOM_ROTATION2": transforms.RandomRotation(degrees=[-10,10]),
    }
    symbols_augmentation = []
    symbols_augmentation.append(DATA_AUGMENTATION["HORIZONTAL_FLIP"])
    symbols_augmentation.append(DATA_AUGMENTATION["TO_TENSOR"])
    # Horizontal flip
    counter = 20000
    symbol_dataset = datasets.ImageFolder("math_symbol_data/div_rapper", transform=transforms.Compose(symbols_augmentation))
    symbol_loader = torch.utils.data.DataLoader(symbol_dataset, batch_size=1, num_workers=1, shuffle=True)
    for images, labels in iter(symbol_loader):
        save_image(images, f"math_symbol_data/div_new/{counter}.png")
        counter += 1
    symbols_augmentation = []
    symbols_augmentation.append(DATA_AUGMENTATION["VERTICAL_FLIP"])
    symbols_augmentation.append(DATA_AUGMENTATION["TO_TENSOR"])
    # vertical flip
    counter = 25000
    symbol_dataset = datasets.ImageFolder("math_symbol_data/div_rapper", transform=transforms.Compose(symbols_augmentation))
    symbol_loader = torch.utils.data.DataLoader(symbol_dataset, batch_size=1, num_workers=1, shuffle=True)
    for images, labels in iter(symbol_loader):
        save_image(images, f"math_symbol_data/div_new/{counter}.png")
        counter += 1
    symbols_augmentation = []
    symbols_augmentation.append(DATA_AUGMENTATION["RANDOM_ROTATION"])
    symbols_augmentation.append(DATA_AUGMENTATION["TO_TENSOR"])
    # rotation -15 to 15
    counter = 30000
    symbol_dataset = datasets.ImageFolder("math_symbol_data/div_rapper", transform=transforms.Compose(symbols_augmentation))
    symbol_loader = torch.utils.data.DataLoader(symbol_dataset, batch_size=1, num_workers=1, shuffle=True)
    for images, labels in iter(symbol_loader):
        save_image(images, f"math_symbol_data/div_new/{counter}.png")
        counter += 1
    symbols_augmentation = []
    symbols_augmentation.append(DATA_AUGMENTATION["RANDOM_ROTATION"])
    symbols_augmentation.append(DATA_AUGMENTATION["TO_TENSOR"])
    # rotation -10 to 10
    counter = 35000
    symbol_dataset = datasets.ImageFolder("math_symbol_data/div_rapper", transform=transforms.Compose(symbols_augmentation))
    symbol_loader = torch.utils.data.DataLoader(symbol_dataset, batch_size=1, num_workers=1, shuffle=True)
    for images, labels in iter(symbol_loader):
        save_image(images, f"math_symbol_data/div_new/{counter}.png")
        counter += 1

