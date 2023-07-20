###########
# IMPORTS #
###########

import torch
import torchvision
from dataloader import get_dataloaders_cifar10, set_all_seeds
from helper_train import train_model
import result_plotter
from sequential_learning.gradient_descent.model import NeuralNetwork


def evaluate_model(model, data_loader, device):
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(data_loader):  # Loop over batches in data.
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ############
    # SETTINGS #
    ############

    seed = 0  # Set random seed.
    #b = 256 * 4  # Set batch size.
    b = 256

    # Get device used for training, e.g., check via torch.cuda.is_available().
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

    ###########
    # DATASET #
    ###########

    # Using transforms on your data allows you to transform it from its
    # source state so that it is ready for training/validation/testing.

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
        train_transforms=cifar_10_transforms,
        test_transforms=cifar_10_transforms,
        num_workers=0
    )

    learning_rate = 0.1

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break
    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    train_generator = iter(train_loader)
    valid_loss_list = []
    valid_accuracy_list = []

    for iteration in range(5000):
        try:
            train_inputs, train_labels = next(train_generator)
        except StopIteration:
            train_generator = iter(train_loader)
            train_inputs, train_labels = next(train_generator)
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)

        outputs = model(train_inputs)
        loss_fn = torch.nn.CrossEntropyLoss()
        model.zero_grad()
        loss = loss_fn(outputs, train_labels)
        loss.backward()

        for param_current in model.parameters():
            param_current.data.sub_(param_current.grad * learning_rate)

        if iteration % 20 == 0:
            loss, accuracy = evaluate_model(model, valid_loader, device)
            valid_loss_list.append(loss)
            valid_accuracy_list.append(accuracy)
            print(f"accuracy after {iteration+1} iterations: {accuracy}")

    torch.save(valid_loss_list, f'simple_gd_loss_{learning_rate}_{b}.pt')
    torch.save(valid_accuracy_list, f'simple_gd_accuracy_{learning_rate}_{b}.pt')

