"""
Compare the finetunings on a single model and the pruned model (before and after combining).
Use a sparsity of 75% to get the original model size.
"""

import torch
import torchvision
import torch_pruning as tp
from matplotlib import pyplot as plt

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

if __name__ == '__main__':

    seed = 0
    set_all_seeds(seed)

    b = 256
    e = 50

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

    model0_state_dict = torch.load("ex1_model_0.pt")
    model1_state_dict = torch.load("ex1_model_1.pt")
    model2_state_dict = torch.load("ex1_model_2.pt")
    model3_state_dict = torch.load("ex1_model_3.pt")

    # sparsities from 0.0 to 0.9
    sparsity = 0.75

    # prune then combine
    model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)

    model0.load_state_dict(model0_state_dict)
    model1.load_state_dict(model1_state_dict)
    model2.load_state_dict(model2_state_dict)
    model3.load_state_dict(model3_state_dict)

    pruned_model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    pruned_model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    pruned_model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    pruned_model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)

    pruned_model0.load_state_dict(model0_state_dict)
    pruned_model1.load_state_dict(model1_state_dict)
    pruned_model2.load_state_dict(model2_state_dict)
    pruned_model3.load_state_dict(model3_state_dict)



    imp0 = tp.importance.MagnitudeImportance(p=2)
    pruner0 = tp.pruner.MagnitudePruner(
        pruned_model0,
        example_inputs,
        importance=imp0,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[pruned_model0.fc4],
    )
    pruner0.step()

    imp1 = tp.importance.MagnitudeImportance(p=2)
    pruner1 = tp.pruner.MagnitudePruner(
        pruned_model1,
        example_inputs,
        importance=imp1,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[pruned_model1.fc4],
    )
    pruner1.step()

    imp2 = tp.importance.MagnitudeImportance(p=2)
    pruner2 = tp.pruner.MagnitudePruner(
        pruned_model2,
        example_inputs,
        importance=imp2,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[pruned_model2.fc4],
    )
    pruner2.step()

    imp3 = tp.importance.MagnitudeImportance(p=2)
    pruner3 = tp.pruner.MagnitudePruner(
        pruned_model3,
        example_inputs,
        importance=imp3,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[pruned_model3.fc4],
    )
    pruner3.step()

    hidden_1 = pruned_model0.fc1.bias.shape[0] + pruned_model1.fc1.bias.shape[0] + pruned_model2.fc1.bias.shape[0] + pruned_model3.fc1.bias.shape[0]
    hidden_2 = pruned_model0.fc2.bias.shape[0] + pruned_model1.fc2.bias.shape[0] + pruned_model2.fc2.bias.shape[0] + pruned_model3.fc2.bias.shape[0]
    hidden_3 = pruned_model0.fc3.bias.shape[0] + pruned_model1.fc3.bias.shape[0] + pruned_model2.fc3.bias.shape[0] + pruned_model3.fc3.bias.shape[0]

    # combine the pruned models
    combined_model_first_prune = AdaptiveCombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes,
                                                   hidden_1, hidden_2, hidden_3)
    for param in combined_model_first_prune.parameters():
        param.data = torch.zeros_like(param.data)

    for (l0_name, l0), (l1_name, l1), (l2_name, l2), (l3_name, l3), (l_combined_name, l_combined) in zip(
            pruned_model0.named_modules(),
            pruned_model1.named_modules(),
            pruned_model2.named_modules(),
            pruned_model3.named_modules(),
            combined_model_first_prune.named_modules()):

        if isinstance(l0, torch.nn.Linear):

            if l0_name == "fc1":
                # first layer
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

    # combine then prune
    combined_state_dict = torch.load("ex1_combined_model.pt")
    combined_model_first_combine = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    combined_model_first_combine.load_state_dict(combined_state_dict)
    imp = tp.importance.MagnitudeImportance(p=2)
    pruner_combined = tp.pruner.MagnitudePruner(
        combined_model_first_combine,
        example_inputs,
        importance=imp,
        ch_sparsity=sparsity,
        root_module_types=[torch.nn.Linear],
        ignored_layers=[combined_model_first_combine.fc4],
    )
    pruner_combined.step()

    print(combined_model_first_combine.fc2.weight.shape)
    print(combined_model_first_prune.fc2.weight.shape)
    print(model0.fc2.weight.shape)

    optimizer_first_combine = torch.optim.SGD(combined_model_first_combine.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_first_prune = torch.optim.SGD(combined_model_first_prune.parameters(), lr=learning_rate, momentum=0.9)
    optimizer_model0 = torch.optim.SGD(model0.parameters(), lr=learning_rate, momentum=0.9)

    accuracies_model0 = []
    accuracies_first_prune = []
    accuracies_first_combine = []

    _, acc0 = evaluate_model(model0, test_loader)
    _, acc_first_combine = evaluate_model(combined_model_first_combine, test_loader)
    _, acc_first_prune = evaluate_model(combined_model_first_prune, test_loader)
    accuracies_model0.append(acc0)
    accuracies_first_combine.append(acc_first_combine)
    accuracies_first_prune.append(acc_first_prune)

    combined_model_first_combine.train()
    combined_model_first_prune.train()
    model0.train()

    # finetune
    for epoch in range(e):

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer_model0.zero_grad()
            outputs_combined = model0(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_combined, labels)
            loss.backward()
            optimizer_model0.step()

            optimizer_first_prune.zero_grad()
            outputs_first_prune = combined_model_first_prune(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_first_prune, labels)
            loss.backward()
            optimizer_first_prune.step()

            optimizer_first_combine.zero_grad()
            outputs_first_combine = combined_model_first_combine(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs_first_combine, labels)
            loss.backward()
            optimizer_first_combine.step()

        print("epoch", epoch + 1)

        # evaluate after each epoch
        _, acc_0 = evaluate_model(model0, test_loader)
        _, acc_first_prune = evaluate_model(combined_model_first_prune, test_loader)
        _, acc_first_combine = evaluate_model(combined_model_first_combine, test_loader)
        print(f"model0: {acc_0}, combine_first: {acc_first_combine}, prune_first: {acc_first_prune}")
        accuracies_model0.append(acc_0)
        accuracies_first_prune.append(acc_first_prune)
        accuracies_first_combine.append(acc_first_combine)

    plt.plot(accuracies_model0, label="model 0 finetuning", marker="o")
    plt.plot(accuracies_first_prune, label="prune-combine finetuning", marker="o")
    plt.plot(accuracies_first_combine, label="combine-prune finetuning", marker="o")

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracies")
    plt.savefig("ex5_accuracies.png")
