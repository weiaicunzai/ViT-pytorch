import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, ConcatDataset
from config import DataSetConfig
from datasets.dataset import CRC, ExtendedCRC


logger = logging.getLogger(__name__)


def get_loader(args):

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.6, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == 'ECRC':
        config = DataSetConfig(args.dataset)
        if args.cv == 1:
            train_path1 = config.fold2
            train_path2 = config.fold3
            test_path = config.fold1
            #trainset = ConcatDataset([fold2, fold3])
            #testset = fold1
        elif args.cv == 2:
            train_path1 = config.fold1
            train_path2 = config.fold3
            test_path = config.fold2
            #trainset = ConcatDataset([fold1, fold3])
            #testset = fold2
        elif args.cv == 3:
            train_path1 = config.fold1
            train_path2 = config.fold2
            test_path = config.fold3
            #testset = fold2

        train1 = ExtendedCRC(train_path1, transform=transform_train)
        train2 = ExtendedCRC(train_path2, transform=transform_train)
        trainset = ConcatDataset([train1, train2])
        testset = ExtendedCRC(test_path, transform=transform_test, return_path=True)

    elif args.dataset == 'CRC':

        config = DataSetConfig(args.dataset)
        if args.cv == 1:
            train_path1 = config.fold2
            train_path2 = config.fold3
            test_path = config.fold1
            #trainset = ConcatDataset([fold2, fold3])
            #testset = fold1
        elif args.cv == 2:
            train_path1 = config.fold1
            train_path2 = config.fold3
            test_path = config.fold2
            #trainset = ConcatDataset([fold1, fold3])
            #testset = fold2
        elif args.cv == 3:
            train_path1 = config.fold1
            train_path2 = config.fold2
            test_path = config.fold3
            #testset = fold2

        train1 = CRC(train_path1, transform=transform_train)
        train2 = CRC(train_path2, transform=transform_train)
        trainset = ConcatDataset([train1, train2])
        testset = CRC(test_path, transform=transform_test, return_path=True)

    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    #if args.local_rank == 0:
    #    torch.distributed.barrier()

    #train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    #train_sampler = SequentialSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              #sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=4,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
