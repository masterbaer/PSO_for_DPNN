"""
We have a look at data-parallel training of ensembles to see if the ensemble accuracy can be reached faster.
We expect individual models to be worse, the combined model to be worse as well.
The hope is that a round of finetuning can recover the accuracy, and hopefully, reach the ensemble-accuracy as if
all models got all data.
"""
import sys
import time

from mpi4py import MPI
import torch
import torchvision

from dataloader import set_all_seeds, get_dataloaders_cifar10, get_dataloaders_cifar10_distributed
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    seed = rank
    set_all_seeds(seed)

    b = 256
    e = 120

    dataset = sys.argv[1]
    if rank == 0:
        print("partitioning:", dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    set_all_seeds(seed)  # Set all seeds to chosen random seed.

    cifar_10_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = None
    valid_loader = None
    test_loader = None

    if dataset == "dataparallel":
        # GET PYTORCH DATALOADERS FOR TRAINING, TESTING, AND VALIDATION DATASET.
        train_loader, valid_loader = get_dataloaders_cifar10_distributed(
            batch_size=b,
            root="../../data",
            validation_fraction=0.1,
            train_transforms=cifar_10_transforms,
            test_transforms=cifar_10_transforms,
            num_workers=0,
            world_size=world_size,
            rank=rank
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root='../../data',
            train=False,
            transform=cifar_10_transforms,
            download=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=b,
            shuffle=False
        )
        print("using a data partition on each worker")

    if dataset == "full":
        train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
            batch_size=b,
            root="../../data",
            validation_fraction=0.1,
            train_transforms=cifar_10_transforms,
            test_transforms=cifar_10_transforms,
            num_workers=0
        )
        print("using the full data set on each worker")


    learning_rate = 0.01

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    for param in combined_model.parameters():
        param.data = torch.zeros_like(param.data)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    combined_test_accuracy_list = []
    test_accuracy_list = []
    time_list = []
    time_per_epoch = 0.0

    for epoch in range(e):

        start_time = time.perf_counter()
        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # train the local model
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.perf_counter()
        time_per_epoch += end_time-start_time
        time_list.append(time_per_epoch)


        state_dict = model.state_dict()
        state_dict = comm.gather(state_dict, root=0)

        if rank == 0:

            # build the large combined model
            for param in combined_model.parameters():
                param.data = torch.zeros_like(param.data)

            model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
            model0.load_state_dict(state_dict[0])
            model1.load_state_dict(state_dict[1])
            model2.load_state_dict(state_dict[2])
            model3.load_state_dict(state_dict[3])


            for (l0_name, l0), (l1_name, l1), (l2_name, l2), (l3_name, l3), (l_combined_name, l_combined) in zip(
                    model0.named_modules(),
                    model1.named_modules(),
                    model2.named_modules(),
                    model3.named_modules(),
                    combined_model.named_modules()):

                if isinstance(l0, torch.nn.Linear):

                    if l0_name == "fc1":
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

                    elif l0_name == "fc4":
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

        model.eval()
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracy_list.append(test_accuracy)

        if rank == 0:
            combined_model.eval()
            _, combined_test_accuracy = evaluate_model(combined_model, test_loader, device)
            combined_test_accuracy_list.append(combined_test_accuracy)

            print(f"test accuracy after {epoch+1} epochs: {test_accuracy, combined_test_accuracy}")

    torch.save(model.state_dict(), f"ex6_{dataset}_model_{rank}.pt")
    torch.save(test_accuracy_list, f'ex6_test_accuracy_{dataset}_{rank}.pt')

    if rank == 0:
        torch.save(combined_test_accuracy_list, f'ex6_combined_{dataset}_test_accuracy.pt')
        torch.save(combined_model.state_dict(), f"ex6_combined_{dataset}_model.pt")
        print("time per epoch: ", time_per_epoch/e)
        torch.save(time_list, f"ex6_time_list_{dataset}.pt")
