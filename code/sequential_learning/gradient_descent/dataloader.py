import os
import torch
import torchvision
import numpy as np


# SET ALL SEEDS
def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# GET DATALOADERS (NON-PARALLEL)
def get_dataloaders_cifar10(batch_size,
                            num_workers=0,
                            root='../../data',
                            validation_fraction=0.1,
                            train_transforms=None,
                            test_transforms=None):
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()

    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    # Load training data.
    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transforms,
        download=True
    )

    # Load validation data.
    valid_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=test_transforms
    )

    # Load test data.
    test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.

    # Get samplers.
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    # Get data loaders.
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=valid_sampler
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader
