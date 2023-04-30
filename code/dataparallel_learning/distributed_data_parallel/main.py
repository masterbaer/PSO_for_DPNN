# IMPORTS
import torch
import torchvision
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# From local helper modules:
from dataloader import get_dataloaders_cifar10_ddp, set_all_seeds
from helper_train import train_model_ddp
from helper_evaluation import compute_accuracy_ddp
import Alexnet


def main():
    world_size = int(os.getenv("SLURM_NPROCS"))  # Get overall number of processes.
    rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
    print("Rank, world size, device count:", rank, world_size, torch.cuda.device_count())

    # Check if distributed package is available.
    # Check if NCCL backend is available.
    if rank == 0:
        if dist.is_available():
            print("Distributed package available...[OK]")  # Distributed package available?
        if dist.is_nccl_available():
            print("NCCL backend available...[OK]")  # NCCL backend available?

    # On each host with N GPUs, spawn up N processes, while ensuring that
    # each process individually works on a single GPU from 0 to N-1.

    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")
    port = "29500"
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = port

    # Initialize DDP.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)  # Initialize DDP.
    if dist.is_initialized():  # Check initialization.
        print("Process group initialized successfully...[OK]")
    print(dist.get_backend(), "backend used...[OK]")  # Check used backend.

    seed = 0  # Set random seed.
    b = 256  # Set batch size.
    e = 40  # Set number of epochs to be trained.

    # Set all seeds to chosen random seed.
    set_all_seeds(seed)

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.CenterCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Get distributed dataloaders for training and validation data on all ranks.
    train_loader, valid_loader = get_dataloaders_cifar10_ddp(batch_size=b,
                                                             root='../../data',
                                                             train_transforms=train_transforms,
                                                             test_transforms=test_transforms)

    # Get dataloader for test data.
    # Final testing is only done on root.
    if rank == 0:
        test_dataset = torchvision.datasets.CIFAR10(
            root="../../data",
            train=False,
            transform=test_transforms
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=b,
            shuffle=False
        )

    model = Alexnet.AlexNet(num_classes=10).cuda()  # Create model and move it to GPU with id rank.
    ddp_model = DDP(model)  # Wrap model with DDP.
    optimizer = torch.optim.SGD(ddp_model.parameters(), momentum=0.9, lr=0.01)
    # Train model.
    train_model_ddp(model=ddp_model, num_epochs=e, train_loader=train_loader,
                    valid_loader=valid_loader, optimizer=optimizer)

    # Test final model on root.
    if rank == 0:
        test_acc = compute_accuracy_ddp(ddp_model, test_loader)  # Compute accuracy on test data.
        print(f'Test accuracy {test_acc :.2f}%')

    dist.destroy_process_group()  # Clean up: Eliminate process group.


# MAIN STARTS HERE.
if __name__ == '__main__':
    main()
