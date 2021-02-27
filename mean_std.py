import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import loaders
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


#dataset_names = ('cifar10', 'cifar100', 'mnist', 'stl10', 'svhn', 'imagenet32', 'fashionmnist', 'tinyimagenet')
dataset_names = ('cifar10', 'cifar100', 'mnist', 'stl10', 'svhn', 'imagenet32', 'fashionmnist', 'tinyimagenet')

parser = argparse.ArgumentParser(description='Calculate Mean Standard')

parser.add_argument('-d', '--dataset', metavar='DATA', default='cifar10', choices=dataset_names,
                    help='dataset to be used: ' + ' | '.join(dataset_names) + ' (default: cifar10)')

args = parser.parse_args()

print(args.dataset)

train_transform = transforms.Compose([transforms.ToTensor()])


def get_annotations_map(base_path):
    #valAnnotationsPath = './tiny-imagenet-200/val/val_annotations.txt'
    valAnnotationsPath = os.path.join(base_path, 'val', 'val_annotations.txt')
    valAnnotationsFile = open(valAnnotationsPath, 'r')
    valAnnotationsContents = valAnnotationsFile.read()
    valAnnotations = {}

    for line in valAnnotationsContents.splitlines():
        pieces = line.strip().split()
        valAnnotations[pieces[0]] = pieces[1]

    return valAnnotations


def load_images(path,num_classes):
    #Load images    
    print('Loading ' + str(num_classes) + ' classes')
    X_train=np.zeros([num_classes*500,3,64,64],dtype='uint8')
    y_train=np.zeros([num_classes*500], dtype='uint8')
    trainPath=path+'/train'

    print('loading training images...')
    i=0
    j=0
    annotations={}
    for sChild in os.listdir(trainPath):
        #sChildPath = os.path.join(os.path.join(trainPath,sChild),'images')
        sChildPath = os.path.join(trainPath,sChild)
        annotations[sChild]=j
        for c in os.listdir(sChildPath):
            X=np.array(Image.open(os.path.join(sChildPath,c)))
            if len(np.shape(X))==2:
                X_train[i]=np.array([X,X,X])
            else:
                X_train[i]=np.transpose(X,(2,0,1))
            y_train[i]=j
            i+=1
        j+=1
        if (j >= num_classes):
            break
    print('finished loading training images')

    """
    val_annotations_map = get_annotations_map(path)
    print(val_annotations_map)
    X_test = np.zeros([num_classes*50,3,64,64],dtype='uint8')
    y_test = np.zeros([num_classes*50], dtype='uint8')
    print('loading test images...')
    i = 0
    #testPath=path+'/val/images'
    testPath=path+'/val'
    for sChild in os.listdir(testPath):
        if val_annotations_map[sChild] in annotations.keys():
            sChildPath = os.path.join(testPath, sChild)
            X=np.array(Image.open(sChildPath))
            if len(np.shape(X))==2:
                X_test[i]=np.array([X,X,X])
            else:
                X_test[i]=np.transpose(X,(2,0,1))
            y_test[i]=annotations[val_annotations_map[sChild]]
            i+=1
        else:
            pass
    print('finished loading test images')+str(i)
    """

    #return X_train,y_train,X_test,y_test
    return X_train,y_train

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

elif args.dataset == "tinyimagenet":
    #train_set = loaders.I1000(root='data/tiny-imagenet-200', train=True, transform=train_transform)
    X_train,y_train = load_images('data/tiny-imagenet-200', 200)
    print(X_train.shape)
    print(np.mean(X_train, axis=(0, 2, 3))/255)
    print(np.std(X_train, axis=(0, 2, 3))/255)

    
    """
    train_set = ImageFolder('/mnt/ssd/tiny-imagenet-200', transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=64, drop_last=True)
    total_values = []
    total_batches = 0
    for batch_index, batch_data in enumerate(train_loader):
        #print(batch_data[0].mean(dim=0).mean(dim=1).mean(dim=1).tolist())
        total_values.append(batch_data[0].mean(dim=0).mean(dim=1).mean(dim=1).tolist())
        total_batches += 1
        #get_dims = batch_data[0].size()
        #print(get_dims)
    print(np.array(total_values))
    print(np.array(total_values).shape)
    print(np.mean(np.array(total_values), axis=0))
    means = torch.from_numpy(np.mean(np.array(total_values), axis=0))
    #print(np.std(np.array(total_values), axis=0))
    diff_values = []
    for batch_index, batch_data in enumerate(train_loader):
        #print(batch_data[0].mean(dim=0).mean(dim=1).mean(dim=1).tolist())

        diff_values.append((batch_data[0]-means).sum(dim=0).sum(dim=1).sum(dim=1).tolist())
        total_batches += 1
        get_dims = batch_data[0].size()
        #print(get_dims)
    print(np.array(diff_values))
    print(get_dims)
    """

    #dataset_path = "preprocessing/imagenet32/2012/images"
    #train_path = os.path.join(dataset_path, 'train/box')
    #train_set = ImageFolder(train_path, transform=train_transform)

    #print(train_set.train_data.shape)
    #print(np.mean(train_set.train_data, axis=(0, 1, 2))/255)
    #print(np.std(train_set.train_data, axis=(0, 1, 2))/255)

