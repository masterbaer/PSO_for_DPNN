import numpy as np
import torch
import torchvision
from numpy.random import random_sample
from torchvision import transforms

from dataloader import get_dataloaders_mnist, get_dataloaders_cifar10
from model import NeuralNetwork
from PSO import PSO


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
    accuracy = correct_pred.float() / num_examples
    return loss_per_batch, accuracy.item()


if __name__ == '__main__':
    b = 256  # Set batch size.

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')

    cifar_10_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        num_workers=0,
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms
    )

    num_classes = 10

    image_shape = None
    # Check loaded data.
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        image_shape = images.shape
        break

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)

    number_of_random_tries = 10
    best_inertial_weight = 0
    best_social_weight = 0
    best_cognitive_weight = 0
    best_loss = float("inf")

    for i in range(number_of_random_tries):
        inertial_weight = (2 - 0) * random_sample() + 0
        social_weight = (2 - 0) * random_sample() + 0
        cognitive_weight = (2 - 0) * random_sample() + 0

        print("current values: (inertial,social,cognitive) = ", inertial_weight, social_weight, cognitive_weight)
        pso = PSO(model=model, num_particles=10, inertia_weight=inertial_weight,
                  social_weight=social_weight, cognitive_weight=cognitive_weight, min_param_value=-1,
                  max_param_value=1, max_iterations=100, train_loader=valid_loader, device=device)

        pso.optimize(visualize=False, evaluate=False)

        loss, accuracy = evaluate_model(model, test_loader)
        print("final test loss: ", loss)
        print("final test accuracy: ", accuracy)

        if loss == "nan":
            print("ALERT, this should never happen. best_loss is nan!")
            continue

        if loss < best_loss:
            best_loss = loss
            best_inertial_weight = inertial_weight
            best_social_weight = social_weight
            best_cognitive_weight = cognitive_weight
            print("better solution found, new best loss: ", best_loss)

        print("best hyperparameters so far: ",
              f"inertial_weight = {best_inertial_weight}, social_weight={best_social_weight}, "
              f"cognitive_weight = {best_cognitive_weight}")

    print("best hyperparameters: ", f"inertial_weight = {best_inertial_weight}, social_weight={best_social_weight}, "
                                    f"cognitive_weight = {best_cognitive_weight}")
