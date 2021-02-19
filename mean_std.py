import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import loaders
from torchvision.datasets import ImageFolder


dataset_names = ('cifar10', 'cifar100', 'mnist', 'stl10', 'svhn', 'imagenet32', 'fashionmnist')

parser = argparse.ArgumentParser(description='Calculate Mean Standard')

parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

args = parser.parse_args()

print(args.dataset)

train_transform = transforms.Compose([transforms.ToTensor()])

if args.dataset == "cifar10":
    train_set = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=train_transform)
    print(train_set.data.shape)
    print(train_set.data.mean(axis=(0, 1, 2))/255)
    print(train_set.data.std(axis=(0, 1, 2))/255)

elif args.dataset == "svhn":
    train_set = torchvision.datasets.SVHN(root='data/svhn', split="train", download=True, transform=train_transform)
    print(train_set.data.shape)
    print(np.mean(train_set.data, axis=(0, 2, 3))/255)
    print(np.std(train_set.data, axis=(0, 2, 3))/255)

elif args.dataset == "cifar100":
    train_set = torchvision.datasets.CIFAR100(root='data/cifar100', train=True, download=True, transform=train_transform)
    print(train_set.data.shape)
    print(np.mean(train_set.data, axis=(0, 1, 2))/255)
    print(np.std(train_set.data, axis=(0, 1, 2))/255)

elif args.dataset == "stl10":
    ########
    train_transform = transforms.Compose([transforms.Resize((32, 32)), train_transform])
    ########
    train_set = torchvision.datasets.STL10(root='data/stl10', split="train", download=True, transform=train_transform)
    train_set.data = torch.from_numpy(train_set.data).permute(0, 2, 3, 1).numpy()
    print(train_set.data.shape)
    print(np.mean(train_set.data, axis=(0, 1, 2))/255)
    print(np.std(train_set.data, axis=(0, 1, 2))/255)

elif args.dataset == "mnist":
    train_set = torchvision.datasets.MNIST(root='data/mnist', train=True, download=True, transform=train_transform)
    print(list(train_set.train_data.size()))
    print(train_set.train_data.float().mean()/255)
    print(train_set.train_data.float().std()/255)

elif args.dataset == "fashionmnist":
    train_set = torchvision.datasets.FashionMNIST(root='data/fashionmnist', train=True, download=True, transform=train_transform)
    print(list(train_set.train_data.size()))
    print(train_set.train_data.float().mean()/255)
    print(train_set.train_data.float().std()/255)

elif args.dataset == "imagenet32":
    train_set = loaders.I1000(root='data/imagenet32', train=True, transform=train_transform)
    #dataset_path = "preprocessing/imagenet32/2012/images"
    #train_path = os.path.join(dataset_path, 'train/box')
    #train_set = ImageFolder(train_path, transform=train_transform)
    print(train_set.train_data.shape)
    print(np.mean(train_set.train_data, axis=(0, 1, 2))/255)
    print(np.std(train_set.train_data, axis=(0, 1, 2))/255)
