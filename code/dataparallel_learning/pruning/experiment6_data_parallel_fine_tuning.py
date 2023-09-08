"""
Finetuning.
"""

import torch
import torchvision

from dataloader import set_all_seeds, get_dataloaders_cifar10
from model import NeuralNetwork, CombinedNeuralNetwork

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

if __name__ == '__main__':

    seed = 0
    set_all_seeds(seed)

    b = 256
    e = 30

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

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

    learning_rate = 0.01

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break

    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    combined_model.load_state_dict(torch.load("ex6_combined_dataparallel_model.pt"))
    optimizer_combined = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
    test_accuracy_list_combined = []

    _, test_accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print(f"ensemble accuracy before finetuning: ", test_accuracy_combined)

    test_accuracy_list_combined.append(test_accuracy_combined)

    for epoch in range(e):

        combined_model.train()

        for i, (inputs, labels) in enumerate(train_loader):

            #train the combined model
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_combined.zero_grad()
            outputs_combined = combined_model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_combined, labels)
            loss.backward()
            optimizer_combined.step()

        combined_model.eval()

        _, test_accuracy = evaluate_model(combined_model, test_loader, device)
        print(f"combined model : test accuracy after {epoch+1} epochs: {test_accuracy}")

        test_accuracy_list_combined.append(test_accuracy)


    combined_model.eval()

    torch.save(test_accuracy_list_combined, 'ex6_combined_test_accuracy_finetune.pt')
