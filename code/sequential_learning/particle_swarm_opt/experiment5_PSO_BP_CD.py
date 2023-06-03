import torch
import torchvision

from dataloader import get_dataloaders_cifar10
from model import NeuralNetwork, PSOPINNNet
from PSO_BP_CD import PSOWithGradients


# This model is from PSO-PINN. https://arxiv.org/pdf/2202.01943.pdf
# The gradient is used as another velocity component and the social and cognitivce coefficients
# decay with 1/n where n is the number of iterations.

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

    #model = PSOPINNNet(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    #pso = PSOWithGradients(model=model, num_particles=50, inertia_weight=0.99,
    #                       social_weight=0.5, cognitive_weight=0.08, max_iterations=1000, train_loader=train_loader,
    #                       valid_loader=valid_loader, learning_rate=0.005, device=device)

    # Trying the PSO-PINN configuration with a larger net. Instead of 50 particles we use 20.
    # This configuration is rather weird. The losses all get way too high with inertia 0.99 using the standard PSO_BP_CD
    # To overcome this problem we also let the inertia decay.
    # We could also instead adapt the initial velocity but the paper did not mention how it was chosen.

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes) # keep in cpu
    pso = PSOWithGradients(model=model, num_particles=20, inertia_weight=0.99,
              social_weight=0.5, cognitive_weight=0.08, max_iterations=1000, train_loader=train_loader,
                           valid_loader=valid_loader, learning_rate=0.005, device=device)

    global_best_loss, global_best_accuracy = pso.optimize(evaluate=False)

    # final loss on test set
    model = model.to(device)
    loss, accuracy = evaluate_model(model, test_loader)
    print("final test loss: ", loss)
    print("final test accuracy: ", accuracy)
