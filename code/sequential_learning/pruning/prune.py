import copy

import numpy as np
import torch
import torchvision
from torch.nn.utils import prune
import torch_pruning as tp

from sequential_learning.pruning.dataloader import set_all_seeds, get_dataloaders_cifar10
from sequential_learning.pruning.model import NeuralNetwork, CombinedNeuralNetwork

# we try pytorch's pruning in this file
# for pruning with a threshhold see
# https://saturncloud.io/blog/how-to-prune-weights-less-than-a-threshold-in-pytorch/

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


def evaluate_model_output_averaging(models, data_loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(data_loader):  # Loop over batches in data.
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)

        predictions = [model(vinputs) for model in models]
        aggregated_predictions = torch.stack(predictions).mean(dim=0)

        _, predicted = torch.max(aggregated_predictions,
                                 dim=1)  # Determine class with max. probability for each sample.
        num_examples += vlabels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == vlabels).sum()  # Update overall number of correct predictions.
        loss = loss_fn(aggregated_predictions, vlabels)
        test_loss = test_loss + loss.item()

    loss_per_batch = test_loss / number_of_batches
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_per_batch, accuracy


def apply_layerwise_unstructured_l1_pruning(layer):
    # use model.apply(apply_layerwise_pruning) to prune
    if isinstance(layer, torch.nn.Linear):
        prune.l1_unstructured(layer, name='weight', amount=0.4)


def apply_layerwise_structured_ln_pruning_(layer):
    # use model.apply(apply_layerwise_pruning) to prune
    if isinstance(layer, torch.nn.Linear):
        # dim = 1 disconnects one input from all neurons,
        # dim = 0 disconnects one neuron.
        # See https://towardsdatascience.com/how-to-prune-neural-networks-with-pytorch-ebef60316b91 .
        prune.ln_structured(layer, name='weight', amount=0.75, n=2, dim=0)


def apply_layerwise_structured_ln_pruning_without_last_layer(model, amount=0.5, n=2, dim=0):
    # dim = 0 corresponds to pruning neurons, dim=1 prunes the input connections
    # use model.apply(apply_layerwise_pruning) to prune
    for name, layer in model.named_modules():
        print(name, layer)
        if isinstance(layer, torch.nn.Linear) and name != "fc4":
            prune.ln_structured(layer, name='weight', amount=0.5, n=2, dim=0)


# use global l1 unstructured pruning
# see https://stackoverflow.com/questions/70346398/how-does-pytorch-l1-norm-pruning-works
def apply_global_unstructured_pruning(parameters_to_prune, amount=0.4):
    # e.g.
    # parameters_to_prune = [
    #         (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Linear, model0.modules())
    #     ]

    # see https://towardsdatascience.com/how-to-prune-neural-networks-with-pytorch-ebef60316b91
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


# the mask is removed and the new parameters are assigned (explicit 0 instead of mask)
def remove_pruned_weights(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            if prune.is_pruned(layer):
                prune.remove(layer, 'weight')


if __name__ == '__main__':

    seed = 0
    b = 256
    e = 40
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    print(f'Using {device} device.')
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

    num_classes = 10
    image_shape = None
    for images, labels in train_loader:
        image_shape = images.shape
        break

    model0 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model1 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model2 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    model3 = NeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)

    model0.load_state_dict(torch.load("simple_model_0.pt"))
    model1.load_state_dict(torch.load("simple_model_1.pt"))
    model2.load_state_dict(torch.load("simple_model_2.pt"))
    model3.load_state_dict(torch.load("simple_model_3.pt"))

    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()

    # _, accuracy0 = evaluate_model(model0, test_loader, device)
    # print(accuracy0)

    # model0.apply(apply_layerwise_unstructured_l1_pruning) # pruning every fully connected layer individiually

    # parameters_to_prune = [
    #    (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Linear, model0.modules())
    # ]
    # apply_global_unstructured_pruning(parameters_to_prune)

    # model0.apply(apply_layerwise_structured_ln_pruning_without_last_layer)

    # TODO activate pruning
    # apply_layerwise_structured_ln_pruning_without_last_layer(model0)
    # remove_pruned_weights(model0)

    # apply_layerwise_structured_ln_pruning_without_last_layer(model1)
    # remove_pruned_weights(model1)

    # apply_layerwise_structured_ln_pruning_without_last_layer(model2)
    # remove_pruned_weights(model2)

    # apply_layerwise_structured_ln_pruning_without_last_layer(model3)
    # remove_pruned_weights(model3)

    # _, accuracy0 = evaluate_model(model0, test_loader, device)
    # print(accuracy0)

    for i, layer in enumerate(model0.modules()):
        number_of_zeros = 0
        number_of_params = 0

        # number_of_zero_neurons = 0
        # number_of_neurons = 0
        if isinstance(layer, torch.nn.Linear):
            # print(i, layer)

            for param in layer.parameters():
                print(param.shape)
                # count number of zeros
                # https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
                number_of_zeros += param.numel() - param.nonzero().size(0)
                number_of_params += param.numel()

            print(f"{number_of_zeros} / {number_of_params} are zero")

    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes).to(device)
    for param in combined_model.parameters():
        param.data = torch.zeros_like(param.data)

    print(combined_model.fc4.weight.shape)
    print(model0.fc4.weight.shape)
    print(model0.fc4.weight.shape[0])

    # combined_model.fc1.weight.data[0:512, :] += model0.fc1.weight.data
    # combined_model.fc1.weight.data[512:1024, :] += model1.fc1.weight.data
    # combined_model.fc1.weight.data[1024:1536, :] += model2.fc1.weight.data
    # combined_model.fc1.weight.data[1536:2048, :] += model3.fc1.weight.data
    # combined_model.fc1.bias.data[0:512] += model0.fc1.bias.data
    # combined_model.fc1.bias.data[512:1024] += model1.fc1.bias.data
    # combined_model.fc1.bias.data[1024:1536] += model2.fc1.bias.data
    # combined_model.fc1.bias.data[1536:2048] += model3.fc1.bias.data

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


    #combined_model.eval()
    #_, accuracy0 = evaluate_model(model0, test_loader, device)
    #_, accuracy1 = evaluate_model(model1, test_loader, device)
    #_, accuracy2 = evaluate_model(model2, test_loader, device)
    #_, accuracy3 = evaluate_model(model3, test_loader, device)
    #_, accuracy_combined = evaluate_model(combined_model, test_loader, device)

    # sanity check
    #_, accuracy_output_averaged = evaluate_model_output_averaging([model0, model1, model2, model3], test_loader, device)
    #print(accuracy0, accuracy1, accuracy2, accuracy3, accuracy_output_averaged, accuracy_combined)

    # using no pruning:
    # 0.5656999945640564 0.5669999718666077 0.5879999995231628 0.5681999921798706 0.6137999892234802 0.6162999868392944
    # changing the order leads to exact results --> copying was done correctly!

    # (re-) apply pruning on the large model

    apply_layerwise_structured_ln_pruning_without_last_layer(combined_model, amount=0.5)


    combined_model.eval()
    _, accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print(accuracy_combined)


    for i, layer in enumerate(combined_model.modules()):
        number_of_zeros = 0
        number_of_params = 0

        # number_of_zero_neurons = 0
        # number_of_neurons = 0
        if isinstance(layer, torch.nn.Linear):
            #print(dict(layer.named_buffers())["weight_mask"].shape)
            #print(list(layer.named_buffers()))
            #continue
            # print(i, layer)

            for param in layer.parameters():
                print(param.shape)
                # count number of zeros
                # https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
                number_of_zeros += param.numel() - param.nonzero().size(0)
                number_of_params += param.numel()

            print(f"{number_of_zeros} / {number_of_params} are zero")




    # TODO remove the pruned neurons from the big model (beware, pytorch does not prune the biases in ln_structured)
    # only the incoming weights are set to 0 (or masked)

    # TODO: Compare Output Averaging vs. this approach when finetuning
