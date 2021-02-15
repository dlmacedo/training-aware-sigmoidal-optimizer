import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import random


def getSVHN(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=TF
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=TF,
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


#def worker_init(worker_id):
#    random.seed(1000)


def getCIFAR10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


#def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
def getCIFAR100(batch_size, TF, data_root='/tmp/public_dataset/pytorch', TTF=None, train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF),
            datasets.CIFAR100(root=data_root, train=True, download=True, transform=TF, target_transform=TTF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            #datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF),
            datasets.CIFAR100(root=data_root, train=False, download=True, transform=TF, target_transform=TTF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

"""
def getSTL10(batch_size, TF, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    #data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    data_root = os.path.expanduser(os.path.join(data_root, 'stl10'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)

    TF = transforms.Compose([transforms.Resize((32, 32)), TF])

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='train', download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10(
                root=data_root, split='unlabeled', download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(list(range(10000))), **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
"""


def getTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        #train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=None, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, input_TF, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'cifar100':
        #_, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
        _, test_loader = getCIFAR100(batch_size=batch_size, TF=input_TF, data_root=dataroot, TTF=lambda x: 0, num_workers=1)
    #elif data_type == 'stl10':
    #    _, test_loader = getSTL10(batch_size=batch_size, TF=input_TF, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'Imagenet_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun_resize':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'LSUN_resize'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'fooling_images':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'fooling_images'))
        testsetout = datasets.ImageFolder(dataroot, transform=transforms.Compose([transforms.Resize((32, 32)), input_TF]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'gaussian_noise':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'gaussian_noise'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'uniform_noise':
        dataroot = os.path.expanduser(os.path.join(dataroot, 'uniform_noise'))
        testsetout = datasets.ImageFolder(dataroot, transform=input_TF)
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    return test_loader
