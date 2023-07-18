import torch
import torchvision
import numpy as np
import os


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_dataloaders_cifar10_distributed(batch_size,
                                        rank,
                                        world_size,
                                        num_workers=0,
                                        root='../../data',
                                        validation_fraction=0.1,
                                        train_transforms=None,
                                        test_transforms=None,
                                        valid_batch_size=128):
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()
    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transforms,
        download=True
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=test_transforms
    )

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.

    # Split into training and validation dataset according to specified validation fraction.
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    # Sampler that restricts data loading to a subset of the dataset.
    # Especially useful in conjunction with torch.nn.parallel.DistributedDataParallel.
    # Each process can pass a DistributedSampler instance as a DataLoader sampler,
    # and load a subset of the original dataset that is exclusive to it.

    # Get samplers.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # Get dataloaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        drop_last=True,
        sampler=valid_sampler
    )
    if rank == 0:
        print("train_loader number of batches: ", len(train_loader))
        print("valid_loader number of batches: ", len(valid_loader))
    return train_loader, valid_loader


def get_dataloaders_cifar10_distributed_full_validation(batch_size,
                                                        rank,
                                                        world_size,
                                                        num_workers=0,
                                                        root='../../data',
                                                        validation_fraction=0.1,
                                                        train_transforms=None,
                                                        test_transforms=None,
                                                        valid_batch_size=128):
    if train_transforms is None:
        train_transforms = torchvision.transforms.ToTensor()
    if test_transforms is None:
        test_transforms = torchvision.transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transforms,
        download=True
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=test_transforms,
        download=True
    )

    # Perform index-based train-validation split of original training data.
    total = len(train_dataset)  # Get overall number of samples in original training data.
    idx = list(range(total))  # Make index list.
    np.random.shuffle(idx)  # Shuffle indices.
    vnum = int(validation_fraction * total)  # Determine number of validation samples from validation split.
    train_indices, valid_indices = idx[vnum:], idx[0:vnum]  # Extract train and validation indices.

    # Split into training and validation dataset according to specified validation fraction.
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(valid_dataset, valid_indices)

    # Sampler that restricts data loading to a subset of the dataset.
    # Especially useful in conjunction with torch.nn.parallel.DistributedDataParallel.
    # Each process can pass a DistributedSampler instance as a DataLoader sampler,
    # and load a subset of the original dataset that is exclusive to it.

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    # Get dataloaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=train_sampler
    )

    # give the full validation set to all workers
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        num_workers=num_workers,
        sampler=valid_sampler,
        drop_last=True
    )

    if rank == 0:
        print("train_loader number of batches: ", len(train_loader))
        print("valid_loader number of batches: ", len(valid_loader))

    return train_loader, valid_loader
