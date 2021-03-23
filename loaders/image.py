import os
import random
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.utils.data as data
#import torchsample.transforms as tstf
import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC

# ImageNet32x32 dataset.
class I1000(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            self.train_data = np.load(self.root + '/imagenet32_train.npz')['data']
            self.train_labels = np.load(self.root + '/imagenet32_train.npz')['labels']
            #self.train_data = self.train_data.transpose((0, 2, 3, 1))
            self.train_data = self.train_data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        else:
            self.test_data = np.load(self.root + '/imagenet32_val.npz')['data']
            self.test_labels = np.load(self.root + '/imagenet32_val.npz')['labels']
            #self.test_data = self.test_data.transpose((0, 2, 3, 1))
            self.test_data = self.test_data.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class ImageLoader:
    def __init__(self, args):

        self.args = args
        self.mnist = False

        if args.dataset == "mnist":
            self.mnist = True
            self.normalize = transforms.Normalize((0.1307,), (0.3081,))
            self.train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "data/mnist"
            self.trainset_for_train = torchvision.datasets.MNIST(
                root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.MNIST(
                root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.MNIST(
                root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "fashionmnist":
            self.mnist = True
            self.normalize = transforms.Normalize((0.2860,), (0.3530,))
            self.train_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "data/fashionmnist"
            self.trainset_for_train = torchvision.datasets.FashionMNIST(
                root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.FashionMNIST(
                root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.FashionMNIST(
                root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "cifar10":
            #self.args.iteractions_per_epoch = 785
            self.normalize = transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            #self.dataset_path = "data/cifar10"
            self.dataset_path = "/mnt/ssd/cifar10"
            self.trainset_for_train = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "cifar100":
            #self.args.iteractions_per_epoch = 785
            self.normalize = transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            #self.dataset_path = "data/cifar100"
            self.dataset_path = "/mnt/ssd/cifar100"
            self.trainset_for_train = torchvision.datasets.CIFAR100(
                root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.CIFAR100(
                root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.CIFAR100(
                root=self.dataset_path, train=False, download=True, transform=self.inference_transform)

        elif args.dataset == "imagenet32":
            def subtract_one(target):
                return target-1
            #norm_mean = [0.48109809447859192, 0.45747185440340027, 0.40785506971129742]
            #norm_std = [0.26040888585626459, 0.25321260169837184, 0.26820634393704579]
            self.normalize = transforms.Normalize((0.481, 0.457, 0.407), (0.260, 0.253, 0.268))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            #self.dataset_path = "data/imagenet32"
            self.dataset_path = "/mnt/ssd/imagenet32"
            self.trainset_for_train = I1000(
                root=self.dataset_path, train=True, transform=self.train_transform, target_transform=subtract_one)
            self.trainset_for_infer = I1000(
                root=self.dataset_path, train=True, transform=self.inference_transform, target_transform=subtract_one)
            self.val_set = I1000(
                root=self.dataset_path, train=False, transform=self.inference_transform, target_transform=subtract_one)
            """
            #self.dataset_path = "preprocessing/imagenet32/2012/images"
            self.dataset_path = "/mnt/ssd/imagenet32/2012/images"
            self.train_path = os.path.join(self.dataset_path, 'train/box')
            self.val_path = os.path.join(self.dataset_path, 'val/box')
            self.trainset_for_train = ImageFolder(
                self.train_path, transform=self.train_transform, target_transform=subtract_one)
            self.trainset_for_infer = ImageFolder(
                self.train_path, transform=self.inference_transform, target_transform=subtract_one)
            self.val_set = ImageFolder(
                self.val_path, transform=self.inference_transform, target_transform=subtract_one)
            """

        elif args.dataset == "svhn":
            self.normalize = transforms.Normalize((0.437, 0.443, 0.472), (0.198, 0.201, 0.197))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "data/svhn"
            self.trainset_for_train = torchvision.datasets.SVHN(
                root=self.dataset_path, split="train", download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.SVHN(
                root=self.dataset_path, split="train", download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.SVHN(
                root=self.dataset_path, split="test", download=True, transform=self.inference_transform)

        elif args.dataset == "stl10":
            #self.args.iteractions_per_epoch = 79
            self.normalize = transforms.Normalize((0.446, 0.439, 0.406), (0.260,  0.256, 0.271))
            self.train_transform = transforms.Compose([
                transforms.Resize(32), #### Important!!!
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.Resize(32), #### Important!!!
                transforms.ToTensor(),
                self.normalize,
            ])
            ##########################################################
            #self.inference_transform = transforms.Compose([transforms.Resize(32), self.inference_transform])
            ##########################################################
            self.dataset_path = "data/stl10"
            self.trainset_for_train = torchvision.datasets.STL10(
                root=self.dataset_path, split="train", download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.STL10(
                root=self.dataset_path, split="train", download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.STL10(
                root=self.dataset_path, split="test", download=True, transform=self.inference_transform)

        elif args.dataset == "imagenet2012":
            if args.model_name.startswith('inception'):
                size = (299, 299)
            else:
                size = (224, 256)

            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size[0]),  # 224 , 299
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.Resize(size[1]),  # 256
                transforms.CenterCrop(size[0]),  # 224 , 299
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "/mnt/ssd/imagenet_scripts/2012/images"
            self.train_path = os.path.join(self.dataset_path, 'train')
            self.val_path = os.path.join(self.dataset_path, 'val')
            self.trainset_for_train = ImageFolder(self.train_path, transform=self.train_transform)
            self.trainset_for_infer = ImageFolder(self.train_path, transform=self.inference_transform)
            self.val_set = ImageFolder(self.val_path, transform=self.inference_transform)


        elif args.dataset == "tinyimagenet200":
            #self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            self.normalize = transforms.Normalize((0.480, 0.448, 0.397), (0.276, 0.269, 0.282))
            self.train_transform = transforms.Compose([
                transforms.Resize(32), #### Important!!!
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.Resize(32), #### Important!!!
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "/mnt/ssd/tiny-imagenet-200"
            self.train_path = os.path.join(self.dataset_path, 'train')
            self.val_path = os.path.join(self.dataset_path, 'val')
            self.trainset_for_train = ImageFolder(self.train_path, transform=self.train_transform)
            self.trainset_for_infer = ImageFolder(self.train_path, transform=self.inference_transform)
            self.val_set = ImageFolder(self.val_path, transform=self.inference_transform)


    def get_loaders(self):

        first_partition_sampler, second_partition_sampler = self._create_partition_samplers()
        trainset_first_partition_loader_for_train = DataLoader(
            self.trainset_for_train, batch_size=self.args.batch_size, num_workers=self.args.workers,
            sampler=first_partition_sampler, worker_init_fn=self._worker_init)
            #sampler=first_partition_sampler, worker_init_fn=self._worker_init, drop_last=True)
        trainset_second_partition_loader_for_train = DataLoader(
            self.trainset_for_train, batch_size=self.args.batch_size, num_workers=self.args.workers,
            sampler=second_partition_sampler, worker_init_fn=self._worker_init)
            #sampler=second_partition_sampler, worker_init_fn=self._worker_init, drop_last=True)
        trainset_first_partition_loader_for_infer = DataLoader(
            self.trainset_for_infer, batch_size=self.args.batch_size, num_workers=self.args.workers,
            sampler=first_partition_sampler, worker_init_fn=self._worker_init)
            #sampler=first_partition_sampler, worker_init_fn=self._worker_init, drop_last=True)
        trainset_second_partition_loader_for_infer = DataLoader(
            self.trainset_for_infer, batch_size=self.args.batch_size, num_workers=self.args.workers,
            sampler=second_partition_sampler, worker_init_fn=self._worker_init)
            #sampler=second_partition_sampler, worker_init_fn=self._worker_init, drop_last=True)
        valset_loader = DataLoader(
            self.val_set, batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=True, worker_init_fn=self._worker_init)

        print("\n##################################################")
        print("TRAINSET FIRST PARTITION LOADER SIZE: ====>>>> ",
              len(trainset_first_partition_loader_for_train.sampler))
        print("TRAINSET SECOND PARTITION LOADER SIZE: ====>>>> ",
              len(trainset_second_partition_loader_for_train.sampler))
        print("TRAINSET TOTAL LOADER SIZE: ====>>>> ",
              len(trainset_first_partition_loader_for_train.sampler) + len(trainset_second_partition_loader_for_train.sampler))
        print("VALIDSET LOADER SIZE: ====>>>> ",
              len(valset_loader.sampler))
        print("##################################################")

        return (trainset_first_partition_loader_for_train, trainset_second_partition_loader_for_train,
                trainset_first_partition_loader_for_infer, trainset_second_partition_loader_for_infer,
                valset_loader, self.normalize)

    def _create_partition_samplers(self):

        first_partition_indexes = []
        second_partition_indexes = []
        number_of_first_partition_indexes = {}
        number_of_second_partition_indexes = {}

        print("\nPreparing samplers...")

        """
        if self.args.number_of_first_partition_examples_per_class != 0:
            for index in tqdm(range(len(self.trainset_for_train))):
                _, label = self.trainset_for_train[index]
                if self.mnist:
                    label = label.item()
                if label not in number_of_first_partition_indexes:
                    number_of_first_partition_indexes[label] = 0
                if number_of_first_partition_indexes[label] < self.args.number_of_first_partition_examples_per_class:
                    first_partition_indexes.append(index)
                    number_of_first_partition_indexes[label] += 1
                elif self.args.number_of_second_partition_examples_per_class == 0:
                    second_partition_indexes.append(index)
        
        if self.args.number_of_second_partition_examples_per_class != 0:
            for index in tqdm(reversed(range(len(self.trainset_for_train)))):
                _, label = self.trainset_for_train[index]
                if self.mnist:
                    label = label.item()
                if label not in number_of_second_partition_indexes:
                    number_of_second_partition_indexes[label] = 0
                if number_of_second_partition_indexes[label] < self.args.number_of_second_partition_examples_per_class:
                    second_partition_indexes.append(index)
                    number_of_second_partition_indexes[label] += 1
                elif self.args.number_of_first_partition_examples_per_class == 0:
                    first_partition_indexes.append(index)
        """

        #"""
        first_partition_indexes = list(range(len(self.trainset_for_train)))
        second_partition_indexes = []
        #"""

        print("\n##################################################")
        print("NUMBER OF FIRST PARTITION EXAMPLES PER CLASS:", number_of_first_partition_indexes)
        print("NUMBER OF SECOND PARTITION EXAMPLES PER CLASS:", number_of_second_partition_indexes)
        print("FIRST PARTITION TOTAL NUMBER OF EXAMPLES:", len(first_partition_indexes))
        print("SECOND PARTITION TOTAL NUMBER OF EXAMPLES:", len(second_partition_indexes))
        print("LENGTH OF TRAINSET PARTITIONS UNION:", len(set(first_partition_indexes) | set(second_partition_indexes)))
        print("LENGTH OF TRAINSET PARTITIONS INTERSECTION:", len(set(first_partition_indexes) & set(second_partition_indexes)))
        print("##################################################")

        trainset_first_partition_sampler = SubsetRandomSampler(first_partition_indexes)
        trainset_second_partition_sampler = SubsetRandomSampler(second_partition_indexes)

        #SequentialSampler
        #SequentialSampler

        return trainset_first_partition_sampler, trainset_second_partition_sampler

    def _worker_init(self, worker_id):
        random.seed(self.args.base_seed)

class BatchNormalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor
