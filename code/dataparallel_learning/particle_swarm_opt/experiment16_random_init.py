import sys

import torch
import torchvision
from mpi4py import MPI

from dataloader import set_all_seeds, get_dataloaders_cifar10_distributed, \
    get_dataloaders_cifar10_distributed_full_validation
from parallel_average_pull import AveragePull
from model import NeuralNetwork


def evaluate_model(model, test_data_loader):
    #  Compute Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(test_data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(test_data_loader):  # Loop over batches in data.
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        predictions = model(vinputs)  # Calculate model output.
        _, predicted = torch.max(predictions, dim=1)  # Determine class with max. probability for each sample.
        num_examples += vlabels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == vlabels).sum()  # Update overall number of correct predictions.
        loss = loss_fn(predictions, vlabels)
        test_loss = test_loss + loss.item()

    loss_per_batch = test_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    set_all_seeds(0)
    b = 256  # Set batch size.

    average_pull_weight = float(sys.argv[1])
    print("average_pull_weight: ", average_pull_weight)

    if rank == 0:
        print(f"batchsize = {b}")

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    if rank == 0:
        print(f'Using {device} device.')

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

    # GET PYTORCH DATALOADERS FOR TRAINING AND VALIDATION DATASET.
    train_loader, valid_loader = get_dataloaders_cifar10_distributed_full_validation(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        num_workers=0,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        rank=rank,
        world_size=world_size
    )

    test_loader = None
    if rank == 0:
        test_dataset = torchvision.datasets.CIFAR10(
            root='../../data',
            train=False,
            transform=test_transforms,
            download=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=b,
            shuffle=False
        )

    set_all_seeds(rank)

    num_classes = 10

    image_shape = None
    # Check loaded data.
    for images, labels in train_loader:
        image_shape = images.shape
        break

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)  # keep in cpu

    pso = AveragePull(model=model, inertia_weight=0.0,
                 average_pull_weight=average_pull_weight, max_iterations=5000, train_loader=train_loader,
                 valid_loader=valid_loader, learning_rate=0.01, device=device, rank=rank, world_size=world_size,
                 comm=comm, step=5,  init_strat="random")

    pso.optimize(evaluate=True, output1=f"experiment16_loss_{average_pull_weight}.pt",
                 output2=f"experiment16_accuracy_{average_pull_weight}.pt")
    # trained_models = pso.optimize()

    if rank == 0:
        # final loss on test set
        model = model.to(device)
        loss, accuracy = evaluate_model(model, test_loader)
        print("final test loss: ", loss)
        print("final test accuracy: ", accuracy)
