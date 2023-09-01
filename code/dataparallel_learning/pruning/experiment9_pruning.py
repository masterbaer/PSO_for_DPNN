"""
Try out 75% pruning to get models of original size on these architectures.
"""

import torch
import torchvision
import torch_pruning as tp
from matplotlib import pyplot as plt
from torch import nn

from dataloader import set_all_seeds, get_dataloaders_cifar10
from model import NeuralNetwork, CombinedNeuralNetwork, AdaptiveCombinedNeuralNetwork

def evaluate_model(model, data_loader, device = "cpu"):
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

    seed = 0
    set_all_seeds(seed)

    b = 256
    e = 20

    set_all_seeds(seed)

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
    example_inputs = None
    for images, labels in train_loader:
        example_inputs = images
        image_shape = images.shape
        break

    number_hidden = 7
    size_hidden = 64

    model0_state_dict = torch.load(f"ex9_model_{number_hidden}_{size_hidden}_0.pt")
    model1_state_dict = torch.load(f"ex9_model_{number_hidden}_{size_hidden}_1.pt")
    model2_state_dict = torch.load(f"ex9_model_{number_hidden}_{size_hidden}_2.pt")
    model3_state_dict = torch.load(f"ex9_model_{number_hidden}_{size_hidden}_3.pt")

    """
    model0_state_dict = torch.load("ex1_model_0.pt")
    model1_state_dict = torch.load("ex1_model_1.pt")
    model2_state_dict = torch.load("ex1_model_2.pt")
    model3_state_dict = torch.load("ex1_model_3.pt")
    """

    sparsity = 0.75
    accuracy_prune_combine_finetune = []


    model0 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=size_hidden,
                      hidden_number=number_hidden)
    model1 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=size_hidden,
                      hidden_number=number_hidden)
    model2 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=size_hidden,
                      hidden_number=number_hidden)
    model3 = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes, hidden_size=size_hidden,
                      hidden_number=number_hidden)
    """
    model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    """

    model0.load_state_dict(model0_state_dict)
    model1.load_state_dict(model1_state_dict)
    model2.load_state_dict(model2_state_dict)
    model3.load_state_dict(model3_state_dict)

    _, accuracy_0 = evaluate_model(model0, test_loader)
    _, accuracy_1 = evaluate_model(model1, test_loader)
    _, accuracy_2 = evaluate_model(model2, test_loader)
    _, accuracy_3 = evaluate_model(model3, test_loader)

    print("accuracy of model 0: ", accuracy_0)


    ensemble_model_without_pruning = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                               hidden_size=4*size_hidden, hidden_number=number_hidden)

    for param in ensemble_model_without_pruning.parameters():
        param.data = torch.zeros_like(param.data)

    combine_models(model0, model1, model2, model3, ensemble_model_without_pruning,
                   first_layer_name="fc1", last_layer_name="fc2")
                   
    _, ensemble_acc = evaluate_model(ensemble_model_without_pruning, test_loader)

    """
    ensemble_model_without_pruning = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)

    for param in ensemble_model_without_pruning.parameters():
        param.data = torch.zeros_like(param.data)

    combine_models(model0, model1, model2, model3, ensemble_model_without_pruning,
                   first_layer_name="fc1", last_layer_name="fc4")
    
    _, ensemble_acc = evaluate_model(ensemble_model_without_pruning, test_loader)
    """

    imp0 = tp.importance.MagnitudeImportance(p=2)
    pruner0 = tp.pruner.MagnitudePruner(
        model0,
        example_inputs,
        importance=imp0,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[model0.fc2],
    )
    pruner0.step()

    imp1 = tp.importance.MagnitudeImportance(p=2)
    pruner1 = tp.pruner.MagnitudePruner(
        model1,
        example_inputs,
        importance=imp1,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[model1.fc2],
    )
    pruner1.step()

    imp2 = tp.importance.MagnitudeImportance(p=2)
    pruner2 = tp.pruner.MagnitudePruner(
        model2,
        example_inputs,
        importance=imp2,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[model2.fc2],
    )
    pruner2.step()

    imp3 = tp.importance.MagnitudeImportance(p=2)
    pruner3 = tp.pruner.MagnitudePruner(
        model3,
        example_inputs,
        importance=imp3,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[model3.fc2],
    )
    pruner3.step()


    # combine the pruned models
    combined_model = NN_Linear(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                               hidden_size=size_hidden, hidden_number=number_hidden)
    """
    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    """

    for param in combined_model.parameters():
        param.data = torch.zeros_like(param.data)

    combine_models(model0,model1,model2,model3,combined_model, first_layer_name="fc1", last_layer_name="fc2")

    combined_model.eval()

    _, acc = evaluate_model(combined_model, test_loader)
    print("ensemble accuracy after pruning: ", acc)
    accuracy_prune_combine_finetune.append(acc)

    # finetune

    optimizer_combined = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(e):

        combined_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer_combined.zero_grad()
            outputs_combined = combined_model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_combined, labels)
            loss.backward()
            optimizer_combined.step()

        print("epoch", epoch + 1)
        _, acc = evaluate_model(combined_model, test_loader)
        accuracy_prune_combine_finetune.append(acc)
        print(f"accuracy after {epoch + 1} epochs: {acc}")



    plt.plot([acc-accuracy_0 for acc in accuracy_prune_combine_finetune], label="train-prune-combine-finetune pipeline", marker="o")

    #plt.plot([accuracy_0], label="model 0", marker="s")
    #plt.plot([accuracy_1], label="model 1", marker="s")
    #plt.plot([accuracy_2], label="model 2", marker="s")
    #plt.plot([accuracy_3], label="model 3", marker="s")

    plt.plot([ensemble_acc - accuracy_0], label="full ensemble improvement", marker="s")
    plt.axhline(y=0, color='b', linestyle=':')

    plt.legend()
    plt.xlabel("finetuning epochs")
    plt.ylabel("accuracy improvement compared to model 0")
    plt.savefig(f"ex9_finetune_accuracies_{number_hidden}_{size_hidden}.png")

