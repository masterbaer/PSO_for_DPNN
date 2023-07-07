import torch
import torchvision

from dataloader import get_dataloaders_cifar10
from model import NeuralNetwork, PSOPINNNet
from PSO_BP_CD import PSO_BP_CD
from helperfunctions import evaluate_model, evaluate_ensemble


# This optimizer does changes to PSO-BP-CD. https://arxiv.org/pdf/2202.01943.pdf

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

    # A small sanity check disabling all PSO and only leaving the Gradients on one particle.

    # model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    # pso = PSOWithGradients(model=model, num_particles=1, inertia_weight=0.0,
    #                       social_weight=0.0, cognitive_weight=0.0, max_iterations=1000, train_loader=train_loader,
    #                       valid_loader=valid_loader, learning_rate=0.01, device=device)

    # This one trains rather well. We do not use a learning rate scheduler here and only get slightly lower performance
    # than SGD with a LR-Scheduler.
    # Reference: # Epoch: 005/040 with batchsize 175 | Train: 0.54 | Validation: 0.50, using SGD with LR-Scheduler
    # Here:  final test accuracy:  0.41749998927116394

    # Hyperparameters from PSO-PINN on first experiment:
    # num_particles = 50 , inertia_weight = 0.9, social_weight=0.5, cognitive_weight=0.08, learning_rate = 0.005
    # This one fails to train. Using inertia decay as well makes this approach more viable.

    # Hyperparameters frmo PSO-PINN for hard problems:
    # num_particles = 50 , inertia_weight = 0.99, social_weight=0.5, cognitive_weight=0.08, learning_rate = 0.005
    # This configuration also fails to train. Using inertia decay as well makes this approach more viable.

    # model = PSOPINNNet(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    # pso = PSOWithGradients(model=model, num_particles=50, inertia_weight=0.99,
    #                       social_weight=0.5, cognitive_weight=0.08, max_iterations=1000, train_loader=train_loader,
    #                       valid_loader=valid_loader, learning_rate=0.005, device=device)

    # Trying the PSO-PINN configuration with a larger net. Instead of 50 particles we use 20.
    # This configuration is rather weird. The losses all get way too high with inertia 0.99 using the standard PSO_BP_CD
    # To overcome this problem we also let the inertia decay.
    # We could also instead adapt the initial velocity but the paper did not mention how it was chosen.

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)  # keep in cpu
    # inertia 0.99 in paper but we get error messages
    pso = PSO_BP_CD(model=model, num_particles=20, inertia_weight=0.0,
                           social_weight=0.5, cognitive_weight=0.08, max_iterations=1000, train_loader=train_loader,
                           valid_loader=valid_loader, learning_rate=0.005, device=device)

    trained_models = pso.optimize(evaluate=True, output1="experiment5_loss.pt", output2="experiment5_accuracy.pt")

    # final loss on test set
    model = model.to(device)
    loss, accuracy = evaluate_model(model, test_loader, device)
    print("final test loss: ", loss)
    print("final test accuracy: ", accuracy)

    ensemble_accuracy = evaluate_ensemble(trained_models, test_loader, device)
    print("final ensemble accuracy: ", ensemble_accuracy)
