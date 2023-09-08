import copy

import numpy as np
import torch
import torchvision
from torch.nn.utils import prune
import torch_pruning as tp
from torchvision import transforms

from sequential_learning.pruning_playground.dataloader import set_all_seeds, get_dataloaders_cifar10
from sequential_learning.pruning_playground.model import NeuralNetwork, CombinedNeuralNetwork, LeNet, CombinedLeNet


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


def combine_models_LeNet(model0, model1, model2, model3, combined_model, first_layer_name="None", last_layer_name="fc3"):
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

        if isinstance(l0, torch.nn.Conv2d):
            # single model:
            #weight: torch.Size([6, 3, 5, 5]) (out,in,kernelx,kernely)
            #bias: torch.Size([6])

            # combined model:
            # weight: torch.Size([24, 3, 5, 5])
            # bias: torch.Size([24])

            output_feature_number = l0.weight.shape[0] # output channels of single model
            input_feature_number = l0.weight.shape[1] # input channels of single model
            print(l0.weight.shape[0], l_combined.weight.shape[0])
            print(l0.weight.shape[1], l_combined.weight.shape[1])
            print(l0.weight.shape[2], l_combined.weight.shape[2])
            print(l0.weight.shape[3], l_combined.weight.shape[3])

            if l0.weight.shape[1] == l_combined.weight.shape[1]:
                # the input is equal e.g. at the very first conv2d.
                l_combined.weight.data[0:output_feature_number,:,:,:] += l0.weight.data
                l_combined.weight.data[output_feature_number:2*output_feature_number, :, :, :] += l1.weight.data
                l_combined.weight.data[2*output_feature_number:3*output_feature_number, :, :, :] += l2.weight.data
                l_combined.weight.data[3*output_feature_number:4*output_feature_number, :, :, :] += l3.weight.data

                l_combined.bias.data[0:output_feature_number] = l0.bias.data
                l_combined.bias.data[output_feature_number:2 * output_feature_number] += l1.bias.data
                l_combined.bias.data[2 * output_feature_number:3 * output_feature_number] += l2.bias.data
                l_combined.bias.data[3 * output_feature_number:4 * output_feature_number] += l3.bias.data
            else:
                # there 4 times as many input channels
                l_combined.weight.data[0:output_feature_number, 0:input_feature_number, :, :] += l0.weight.data
                l_combined.weight.data[output_feature_number:2 * output_feature_number, input_feature_number:2*input_feature_number, :, :] += l1.weight.data
                l_combined.weight.data[2 * output_feature_number:3 * output_feature_number, 2*input_feature_number:3*input_feature_number, :, :] += l2.weight.data
                l_combined.weight.data[3 * output_feature_number:4 * output_feature_number, 3*input_feature_number:4*input_feature_number, :, :] += l3.weight.data

                l_combined.bias.data[0:output_feature_number] = l0.bias.data
                l_combined.bias.data[output_feature_number:2 * output_feature_number] += l1.bias.data
                l_combined.bias.data[2 * output_feature_number:3 * output_feature_number] += l2.bias.data
                l_combined.bias.data[3 * output_feature_number:4 * output_feature_number] += l3.bias.data

            # there is third case here because convolutions are not used as the last layer


if __name__ == '__main__':

    seed = 0
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

    num_classes = 10
    image_shape = None
    example_inputs = None
    for images, labels in train_loader:
        example_inputs = images
        image_shape = images.shape
        break

    model0 = LeNet(num_classes).to(device)
    model1 = LeNet(num_classes).to(device)
    model2 = LeNet(num_classes).to(device)
    model3 = LeNet(num_classes).to(device)

    model0.load_state_dict(torch.load("model_0.pt"))
    model1.load_state_dict(torch.load("model_1.pt"))
    model2.load_state_dict(torch.load("model_2.pt"))
    model3.load_state_dict(torch.load("model_3.pt"))

    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()

    for i, layer in enumerate(model0.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            print(layer.weight.shape)
            print(layer.bias.shape)
            print(layer.bias)
            break


    # _, acc = evaluate_model_output_averaging([model0,model1,model2,model3],test_loader, device)
    # print(acc)
    # 0.6994999647140503

    combined_model = CombinedLeNet().to(device)
    for param in combined_model.parameters():
        param.data = torch.zeros_like(param.data)

    combine_models_LeNet(model0,model1,model2,model3,combined_model)

    #_, acc = evaluate_model(combined_model,test_loader, device)
    #print(acc)
    # 0.6994999647140503

    combined_model.to("cpu")

    # prune

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner_combined = tp.pruner.MagnitudePruner(
        combined_model,
        example_inputs,
        importance=imp,
        ch_sparsity=0.75,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
        ignored_layers=[combined_model.fc3],
    )
    pruner_combined.step()

    # finetune


    learning_rate = 0.01
    optimizer = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True)

    valid_loss_list = []
    valid_accuracy_list = []

    combined_model.to(device)

    for epoch in range(e):

        combined_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = combined_model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        combined_model.eval()

        valid_loss, valid_accuracy = evaluate_model(combined_model, valid_loader, device)
        print(f"validation accuracy after {epoch+1} epochs: {valid_accuracy}")

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        scheduler.step(valid_accuracy_list[-1])

    combined_model.eval()
    _, accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print("accuracy_combined")


    #