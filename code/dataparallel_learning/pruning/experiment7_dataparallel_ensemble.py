"""
We compare data-parallel ensembles to synchronous SGD by combining and pruning the models every k iterations
(instead of averaging).
A similar approach was done by Sun et al. (https://arxiv.org/pdf/1606.00575.pdf) with the difference that they used
knowledge distillation instead of pruning.
They synchronize every 4000 batches for cifar-10 (instead of every 16 batches for model averaging).

We expect the accuracy to drop at each synchronization step as it does after pruning without finetuning.

"""
import time

from mpi4py import MPI
import torch
import torchvision
import torch_pruning as tp

from dataloader import set_all_seeds, get_dataloaders_cifar10, get_dataloaders_cifar10_distributed
from model import NeuralNetwork, CombinedNeuralNetwork

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

    b = 64 # batch size per rank
    e = 60
    sync_frequency = 50  # how many batches until sync is required
    finetune_length = 10 # number of batches to finetune the combined model locally

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.

    cifar_10_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

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



    learning_rate = 0.01

    num_classes = 10
    image_shape = None
    # take first image as example input for torch-pruning
    example_inputs = None
    for images, labels in train_loader:
        example_inputs = images
        image_shape = images.shape
        break

    model = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    test_accuracy_list = []
    time_list = []
    batch_num = 0
    start_time = time.perf_counter()

    for epoch in range(e):

        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            batch_num += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            # train the local model
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_num % sync_frequency == 0:
                #  synchronize using ensemble-pruning

                # prune locally

                sparsity = 0.75
                imp = tp.importance.MagnitudeImportance(p=2)
                pruner = tp.pruner.MagnitudePruner(
                    model,
                    example_inputs,
                    importance=imp,
                    ch_sparsity=sparsity,
                    root_module_types=[torch.nn.Linear],
                    ignored_layers=[model.fc4],
                )
                pruner.step()


                state_dict = model.state_dict()
                state_dict = comm.gather(state_dict, root=0)

                # combine
                combined_state_dict = None

                if rank == 0:
                    # build the combined model
                    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3],
                                                           num_classes).to(device)
                    for param in combined_model.parameters():
                        param.data = torch.zeros_like(param.data)

                    model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
                    model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
                    model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
                    model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
                    model0.load_state_dict(state_dict[0])
                    model1.load_state_dict(state_dict[1])
                    model2.load_state_dict(state_dict[2])
                    model3.load_state_dict(state_dict[3])

                    combine_models(model0, model1, model2, model3, combined_model, first_layer_name="fc1",
                                   last_layer_name="fc4")
                    combined_state_dict = combined_model.state_dict()

                # distribute the combined model state dict
                combined_state_dict = comm.bcast(combined_state_dict, root=0)

                # load the combined state dict
                model.load_state_dict(combined_state_dict)

                # finetune the new model

                for fine_tune_batch in range(finetune_length):

                    fine_tune_inputs, fine_tune_labels = next(iter(train_loader))
                    finetune_inputs = fine_tune_inputs.to(device)
                    fine_tune_labels = fine_tune_labels.to(device)

                    optimizer.zero_grad()
                    fine_tune_outputs = model(fine_tune_inputs)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    fine_tune_loss = loss_fn(fine_tune_outputs, fine_tune_labels)
                    fine_tune_loss.backward()
                    optimizer.step()


        model.eval()
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracy_list.append(test_accuracy)

        cumulate_epoch_time = time.perf_counter()
        cumulative_time_passed = cumulate_epoch_time - start_time
        time_list.append(cumulative_time_passed)

        if rank == 0:
            print(f"test accuracy after {epoch+1} epochs: {test_accuracy}, time: {cumulative_time_passed}")

    torch.save(test_accuracy_list, f'ex7_test_accuracy_{rank}.pt')

    if rank == 0:
        torch.save(time_list, "ex7_cumulative_times.pt")
