"""

Testing the social-only model in combination with gradients.
Also, try out an additional linear inertial decay

"""

import torch
import torchvision

from dataloader import get_dataloaders_cifar10
from model import NeuralNetwork
from PSO_social_only_with_gradients import PSOWithGradientsOnlySocial
from helperfunctions import evaluate_model, evaluate_ensemble

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

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)  # keep in cpu
    #pso = PSOWithGradientsOnlySocial(model=model, num_particles=20, inertia_weight=0.9,
    #                       social_weight=0.5, max_iterations=1000, train_loader=train_loader,
    #                       valid_loader=valid_loader, learning_rate=0.01, device=device)
    pso = PSOWithGradientsOnlySocial(model=model, num_particles=20, inertia_weight=0.0,
                           social_weight=0.5, max_iterations=1000, train_loader=train_loader,
                           valid_loader=valid_loader, learning_rate=0.01, device=device)

    trained_models = pso.optimize(evaluate=True, output1="experiment7_loss.pt", output2="experiment7_accuracy.pt")

    # final loss on test set
    model = model.to(device)
    loss, accuracy = evaluate_model(model, test_loader, device)
    print("final test loss: ", loss)
    print("final test accuracy: ", accuracy)

    ensemble_accuracy = evaluate_ensemble(trained_models, test_loader, device)
    print("final ensemble accuracy: ", ensemble_accuracy)
