"""Code for getting the data loaders."""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_loaders(args, dataset=None):
    """Get data loaders for required dataset."""
    if dataset is None:
        dataset = args.dataset
    return get_loaders_eval(dataset, args)


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10()
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)

    train_sampler, valid_sampler = None, None

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=8, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes

def _data_transforms_cifar10():
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    valid_transform = transforms.Compose([transforms.ToTensor()])

    return train_transform, valid_transform
