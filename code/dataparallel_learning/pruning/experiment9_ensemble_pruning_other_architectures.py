"""

We try and compare the ensemble-pruning approach on 3 more architectures with varying architectures using only fully
connected layers.

"""
import sys

from mpi4py import MPI
import torch
import torchvision
from torch import nn

from dataloader import set_all_seeds, get_dataloaders_cifar10


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


def combine_models(model0, model1, model2, model3, combined_model, first_layer_name, last_layer_name):
    for (l0_name, l0), (l1_name, l1), (l2_name, l2), (l3_name, l3), (l_combined_name, l_combined) in zip(
            model0.named_modules(),
            model1.named_modules(),
            model2.named_modules(),
            model3.named_modules(),
            combined_model.named_modules()):

        if isinstance(l0, torch.nn.Linear):

            if l0_name == first_layer_name:
                # not last layer. Stack the layers on top of each other.
                number_of_neurons = l0.weight.shape[0]  # [1] gives the number of inputs
                l_combined.weight.data[0:number_of_neurons, :] += l0.weight.data
                l_combined.weight.data[number_of_neurons:2 * number_of_neurons, :] += l1.weight.data
                l_combined.weight.data[2 * number_of_neurons:3 * number_of_neurons, :] += l2.weight.data
                l_combined.weight.data[3 * number_of_neurons:4 * number_of_neurons, :] += l3.weight.data
                l_combined.bias.data[0:number_of_neurons] += l0.bias.data
                l_combined.bias.data[number_of_neurons:2 * number_of_neurons] += l1.bias.data
                l_combined.bias.data[2 * number_of_neurons:3 * number_of_neurons] += l2.bias.data
                l_combined.bias.data[3 * number_of_neurons:4 * number_of_neurons] += l3.bias.data

            elif l0_name == last_layer_name:
                # last layer. Add the layers and divide the parameters by 4 to average them.

                number_of_inputs = l0.weight.shape[1]

                l_combined.weight.data[:, 0:number_of_inputs] += l0.weight.data
                l_combined.weight.data[:, number_of_inputs:2 * number_of_inputs] += l1.weight.data
                l_combined.weight.data[:, 2 * number_of_inputs:3 * number_of_inputs] += l2.weight.data
                l_combined.weight.data[:, 3 * number_of_inputs:4 * number_of_inputs] += l3.weight.data
                l_combined.bias.data[:] += l0.bias.data
                l_combined.bias.data[:] += l1.bias.data
                l_combined.bias.data[:] += l2.bias.data
                l_combined.bias.data[:] += l3.bias.data
                # divide weights and biases by 4
                l_combined.weight.data = l_combined.weight.data / 4
                l_combined.bias.data = l_combined.bias.data / 4
            #

            else:
                # all other layers. Now we have to consider the number of inputs as well.
                number_of_neurons = l0.weight.shape[0]
                numer_of_inputs = l0.weight.shape[1]
                l_combined.weight.data[0:number_of_neurons, :numer_of_inputs] += l0.weight.data
                l_combined.weight.data[number_of_neurons:2 * number_of_neurons,
                numer_of_inputs:2 * numer_of_inputs] += l1.weight.data
                l_combined.weight.data[2 * number_of_neurons:3 * number_of_neurons,
                2 * numer_of_inputs:3 * numer_of_inputs] += l2.weight.data
                l_combined.weight.data[3 * number_of_neurons:4 * number_of_neurons,
                3 * numer_of_inputs:4 * numer_of_inputs] += l3.weight.data
                l_combined.bias.data[0:number_of_neurons] += l0.bias.data
                l_combined.bias.data[number_of_neurons:2 * number_of_neurons] += l1.bias.data
                l_combined.bias.data[2 * number_of_neurons:3 * number_of_neurons] += l2.bias.data
                l_combined.bias.data[3 * number_of_neurons:4 * number_of_neurons] += l3.bias.data


class NN_Linear(nn.Module):
    # one hidden layer
    def __init__(self, input_dim, output_dim, hidden_size, hidden_number):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList()
        for i in range(hidden_number - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))

        for layer in self.layers:
            x = nn.functional.relu(layer(x))

        x = self.fc2(x)
        return x


if __name__ == '__main__':

    hidden_number = sys.argv[1]
    hidden_number = int(hidden_number)

    hidden_size = sys.argv[2]
    hidden_size = int(hidden_size)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    seed = rank
    set_all_seeds(seed)

    b = 256
    e = 60

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

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

    model = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=hidden_size,
                      hidden_number=hidden_number)
    combined_model = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                               hidden_size=hidden_size * 4, hidden_number=hidden_number)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    combined_test_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(e):

        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs
            labels = labels

            # train the local model
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        state_dict = model.state_dict()
        state_dict = comm.gather(state_dict, root=0)

        if rank == 0:

            # build the large combined model
            for param in combined_model.parameters():
                param.data = torch.zeros_like(param.data)

            model0 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=hidden_size,
                               hidden_number=hidden_number)
            model1 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=hidden_size,
                               hidden_number=hidden_number)
            model2 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=hidden_size,
                               hidden_number=hidden_number)
            model3 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=hidden_size,
                               hidden_number=hidden_number)
            model0.load_state_dict(state_dict[0])
            model1.load_state_dict(state_dict[1])
            model2.load_state_dict(state_dict[2])
            model3.load_state_dict(state_dict[3])

            combine_models(model0, model1, model2, model3, combined_model, "fc1", "fc2")

        model.eval()
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracy_list.append(test_accuracy)

        if rank == 0:
            combined_model.eval()
            _, combined_test_accuracy = evaluate_model(combined_model, test_loader, device)
            combined_test_accuracy_list.append(combined_test_accuracy)

            print(
                f"test accuracy after {epoch + 1} epochs: {test_accuracy, combined_test_accuracy}")


    torch.save(model.state_dict(), f"ex9_model_{hidden_number}_{hidden_size}_{rank}.pt")
    torch.save(test_accuracy_list, f'ex9_test_accuracy_{hidden_number}_{hidden_size}_{rank}.pt')

    if rank == 0:
        torch.save(combined_test_accuracy_list, f'ex9_combined_test_accuracy_{hidden_number}_{hidden_size}.pt')
        torch.save(combined_model.state_dict(), f"ex9_combined_model_{hidden_number}_{hidden_size}.pt")
