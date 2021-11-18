from torchvision import datasets
from torchvision import transforms
import torch

def get_dataset_digits(display=False):
    mnist_train = datasets.MNIST('data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
    mnist_test = datasets.MNIST('data', train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])
    if display:
        for i in mnist_train:
            print(len(i[0]))
    return mnist_train, mnist_val, mnist_test

def get_dataset_letters(display=False):
    emnist_train = datasets.EMNIST('data', split="letters", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
    emnist_test = datasets.EMNIST('data', split="letters", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))
    emnist_train.targets[emnist_train.targets > 0] = emnist_train.targets[emnist_train.targets > 0] + 9
    emnist_train, emnist_val = torch.utils.data.random_split(emnist_train, [104000, 20800])
    if display:
        for i in emnist_train:
            print(i[0])
    return emnist_train, emnist_val, emnist_test