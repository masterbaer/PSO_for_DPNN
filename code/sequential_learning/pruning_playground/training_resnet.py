import torch
import torchvision
from torchvision import transforms as transforms

from dataloader import get_dataloaders_cifar10, set_all_seeds
from model import NeuralNetwork, LeNet


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

    seed = 3
    b = 256
    e = 40
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

    # https://github.com/soapisnotfat/pytorch-cifar10/blob/master/main.py
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=b,
        root="../../data",
        validation_fraction=0.1,
        train_transforms=train_transform,
        test_transforms=test_transform,
        num_workers=0
    )

    learning_rate = 0.01

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        print(image_shape)
        break

    #model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model = ResNet(num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)

    valid_loss_list = []
    valid_accuracy_list = []

    for epoch in range(e):

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()

        valid_loss, valid_accuracy = evaluate_model(model, valid_loader, device)
        print(f"validation accuracy after {epoch+1} epochs: {valid_accuracy}")

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        #scheduler.step(valid_accuracy_list[-1])

    torch.save(model.state_dict(), f"model_{seed}.pt")

    torch.save(valid_loss_list, f'loss_{seed}.pt')
    torch.save(valid_accuracy_list, f'accuracy_{seed}.pt')