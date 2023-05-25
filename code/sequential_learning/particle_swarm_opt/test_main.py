import torch
import torchvision
from torchvision import transforms

from dataloader import get_dataloaders_mnist, get_dataloaders_cifar10
from model import NeuralNetwork
from PSO import PSO

if __name__ == '__main__':
    b = 256  # Set batch size.

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        num_workers=0
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

    pso = PSO(model=model, num_particles=10, inertia_weight=0.9,
              social_weight=0.5, cognitive_weight=0.8, min_param_value=-1,
              max_param_value=1, max_iterations=200, train_loader=valid_loader, device=device)

    pso.optimize(visualize=True)

    # for i, (vinputs, vlabels) in enumerate(train_loader):
    #    num_weights_nn = sum(p.numel() for p in model.parameters())
    #    print("number of weights: ", num_weights_nn)
    #    print(model(vinputs))
    #    break
