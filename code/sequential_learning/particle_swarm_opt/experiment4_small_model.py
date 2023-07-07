import torch
import torchvision

from dataloader import get_dataloaders_cifar10, get_dataloaders_cifar10_half_training_batch_size
from model import NeuralNetwork
from helperfunctions import evaluate_model
from PSO import PSO

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

    pso = PSO(model=model, num_particles=20, inertia_weight=0.0,
              social_weight=0.5, cognitive_weight=0.8, max_iterations=200, train_loader=train_loader,
              valid_loader=valid_loader, device=device)
    pso.optimize(evaluate=False, output1="experiment4_loss.pt", output2="experiment4_accuracy.pt")

    model = model.to(device)  # for evaluation use in gpu
    loss, accuracy = evaluate_model(model, test_loader, device)

    # final loss on test set

    print("final test loss: ", loss)
    print("final test accuracy: ", accuracy)
