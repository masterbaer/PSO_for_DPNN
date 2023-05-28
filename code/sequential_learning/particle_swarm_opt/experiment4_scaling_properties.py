import torch
import torchvision
from matplotlib import pyplot as plt
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
    return loss_per_batch, accuracy


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

    smallest_particle_size = 1
    largest_particle_size = 30

    loss_per_particlenumber = torch.zeros(largest_particle_size - smallest_particle_size + 1)
    accuracy_per_particlenumber = torch.zeros(largest_particle_size - smallest_particle_size + 1)
    test_loss_per_particlenumber = torch.zeros(largest_particle_size - smallest_particle_size + 1)
    test_accuracy_per_particlenumber = torch.zeros(largest_particle_size - smallest_particle_size + 1)

    for number_of_particles_used in range(largest_particle_size - smallest_particle_size + 1):
        model.zero_grad()

        pso = PSO(model=model, num_particles=smallest_particle_size + number_of_particles_used, inertia_weight=0.9,
                  social_weight=0.5, cognitive_weight=0.8, min_param_value=-1,
                  max_param_value=1, max_iterations=100, train_loader=valid_loader, device=device)
        gbest_loss, best_seen_accuracy = pso.optimize(visualize=False, evaluate=False)

        loss_per_particlenumber[number_of_particles_used] = gbest_loss
        accuracy_per_particlenumber[number_of_particles_used] = best_seen_accuracy

        test_loss, test_accuracy = evaluate_model(model, test_loader)
        test_loss_per_particlenumber[number_of_particles_used] = test_loss
        test_accuracy_per_particlenumber[number_of_particles_used] = test_accuracy

    torch.save(loss_per_particlenumber, f"loss_per_particlenumber_{smallest_particle_size}_to_{largest_particle_size}.pt")
    torch.save(accuracy_per_particlenumber, f"accuracy_per_particlenumber_{smallest_particle_size}_to_{largest_particle_size}.pt")
    torch.save(test_loss_per_particlenumber, f"test_loss_per_particlenumber_{smallest_particle_size}_to_{largest_particle_size}.pt")
    torch.save(test_accuracy_per_particlenumber, f"test_accuracy_per_particlenumber_{smallest_particle_size}_to_{largest_particle_size}.pt")

    print("losses: ", loss_per_particlenumber.numpy())
    print("accuracies:", accuracy_per_particlenumber.numpy())
    print("test-losses: ", test_loss_per_particlenumber.numpy())
    print("test-accuracies: ", test_accuracy_per_particlenumber.numpy())
