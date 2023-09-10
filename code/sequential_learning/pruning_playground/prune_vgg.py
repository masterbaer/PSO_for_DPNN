import copy

import numpy as np
import torch
import torchvision
from torch.nn.utils import prune
import torch_pruning as tp
from torchvision import transforms

from sequential_learning.pruning_playground.dataloader import set_all_seeds, get_dataloaders_cifar10
from sequential_learning.pruning_playground.model import NeuralNetwork, CombinedNeuralNetwork, LeNet, CombinedLeNet, \
    VGG, CombinedVGG  #,CustomBatchNorm, CombinedVGGCustomBatch,


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

def evaluate_model_output_averaging_four_models(mode0,model1,model2,model3, data_loader, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(data_loader):  # Loop over batches in data.
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)

        predictions = [model0(vinputs), model1(vinputs), model2(vinputs), model3(vinputs)]
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

def isLeafLayer(layer):
    return isinstance(layer, torch.nn.Linear) or isinstance(layer,torch.nn.Conv2d) or isinstance(layer,torch.nn.BatchNorm2d)

"""
def combine_models_VGG_BatchNorm(model0, model1, model2, model3, combined_model, first_layer_name="None", last_layer_name="classifier"):

    iter0 = iter(model0.named_modules())
    iter1 = iter(model1.named_modules())
    iter2 = iter(model2.named_modules())
    iter3 = iter(model3.named_modules())
    iter_combined = iter(combined_model.named_modules())

    while True:

        # get next layer

        try:
            (l_combined_name, l_combined) = next(iter_combined)
            (l0_name, l0) = next(iter0)
            (l1_name, l1) = next(iter1)
            (l2_name, l2) = next(iter2)
            (l3_name, l3) = next(iter3)
        except StopIteration:
            break

        # find next leaf layer for the individual models
        while not (isLeafLayer(l0)):
            try:
                (l0_name, l0) = next(iter0)
                (l1_name, l1) = next(iter1)
                (l2_name, l2) = next(iter2)
                (l3_name, l3) = next(iter3)
            except StopIteration:
                break
        # find next leaf layer for the combined model
        while not (isLeafLayer(l_combined)):
            try:
                (l_combined_name, l_combined) = next(iter_combined)
            except StopIteration:
                break

        #print(l0)
        #print(l1)
        #print(l2)
        #print(l3)
        #print(l_combined)


        if isinstance(l_combined, torch.nn.Linear):

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
                #print("last layer linear")
                number_of_inputs = l0.weight.shape[1]
                #print(l0.weight.shape[0], l_combined.weight.shape[0])
                #print(l0.weight.shape[1], l_combined.weight.shape[1])

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
                #print("inner layer linear")
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

        if isinstance(l_combined, torch.nn.Conv2d):
            # single model:
            #weight: torch.Size([6, 3, 5, 5]) (out,in,kernelx,kernely)
            #bias: torch.Size([6])

            # combined model:
            # weight: torch.Size([24, 3, 5, 5])
            # bias: torch.Size([24])

            output_feature_number = l0.weight.shape[0] # output channels of single model
            input_feature_number = l0.weight.shape[1] # input channels of single model
            #print(l0.weight.shape[0], l_combined.weight.shape[0])
            #print(l0.weight.shape[1], l_combined.weight.shape[1])
            #print(l0.weight.shape[2], l_combined.weight.shape[2])
            #print(l0.weight.shape[3], l_combined.weight.shape[3])

            if l0.weight.shape[1] == l_combined.weight.shape[1]:
                #print("first conv2d")
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
                #print("inner conv2d")
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

        if isinstance(l_combined, torch.nn.BatchNorm2d):


            # copy weight, bias, running_mean and running_var by copying everything
            #custom_bn = CustomBatchNorm(l_combined.num_features)

            l_combined.weight.data = l0.weight.data
            l_combined.bias.data = l0.bias.data
            l_combined.running_mean.data = l0.running_mean.data
            l_combined.running_var.data = l0.running_var.data

            # find next leaf layer for the combined model
            (l_combined_name, l_combined) = next(iter_combined)
            while not (isLeafLayer(l_combined)):
                try:
                    (l_combined_name, l_combined) = next(iter_combined)
                except StopIteration:
                    break
            l_combined.weight.data = l1.weight.data
            l_combined.bias.data = l1.bias.data
            l_combined.running_mean.data = l1.running_mean.data
            l_combined.running_var.data = l1.running_var.data

            # find next leaf layer for the combined model
            (l_combined_name, l_combined) = next(iter_combined)
            while not (isLeafLayer(l_combined)):
                try:
                    (l_combined_name, l_combined) = next(iter_combined)
                except StopIteration:
                    break
            l_combined.weight.data = l2.weight.data
            l_combined.bias.data = l2.bias.data
            l_combined.running_mean.data = l2.running_mean.data
            l_combined.running_var.data = l2.running_var.data

            # find next leaf layer for the combined model
            (l_combined_name, l_combined) = next(iter_combined)
            while not (isLeafLayer(l_combined)):
                try:
                    (l_combined_name, l_combined) = next(iter_combined)
                except StopIteration:
                    break
            l_combined.weight.data = l3.weight.data
            l_combined.bias.data = l3.bias.data
            l_combined.running_mean.data = l3.running_mean.data
            l_combined.running_var.data = l3.running_var.data


            # replace this layer with a CustomBatchNorm-layer!
            #setattr(combined_model, l_combined_name, custom_bn)
"""

def combine_models_VGG(model0, model1, model2, model3, combined_model, first_layer_name="None",
                                 last_layer_name="classifier"):
    iter0 = iter(model0.named_modules())
    iter1 = iter(model1.named_modules())
    iter2 = iter(model2.named_modules())
    iter3 = iter(model3.named_modules())
    iter_combined = iter(combined_model.named_modules())

    while True:

        # get next layer

        try:
            (l_combined_name, l_combined) = next(iter_combined)
            (l0_name, l0) = next(iter0)
            (l1_name, l1) = next(iter1)
            (l2_name, l2) = next(iter2)
            (l3_name, l3) = next(iter3)
        except StopIteration:
            break

        # find next leaf layer for the individual models
        while not (isLeafLayer(l0)):
            try:
                (l0_name, l0) = next(iter0)
                (l1_name, l1) = next(iter1)
                (l2_name, l2) = next(iter2)
                (l3_name, l3) = next(iter3)
            except StopIteration:
                break
        # find next leaf layer for the combined model
        while not (isLeafLayer(l_combined)):
            try:
                (l_combined_name, l_combined) = next(iter_combined)
            except StopIteration:
                break

        if isinstance(l_combined, torch.nn.Linear):

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
                # print("last layer linear")
                number_of_inputs = l0.weight.shape[1]
                # print(l0.weight.shape[0], l_combined.weight.shape[0])
                # print(l0.weight.shape[1], l_combined.weight.shape[1])

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
                # print("inner layer linear")
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

        if isinstance(l_combined, torch.nn.Conv2d):
            # single model:
            # weight: torch.Size([6, 3, 5, 5]) (out,in,kernelx,kernely)
            # bias: torch.Size([6])

            # combined model:
            # weight: torch.Size([24, 3, 5, 5])
            # bias: torch.Size([24])

            output_feature_number = l0.weight.shape[0]  # output channels of single model
            input_feature_number = l0.weight.shape[1]  # input channels of single model
            # print(l0.weight.shape[0], l_combined.weight.shape[0])
            # print(l0.weight.shape[1], l_combined.weight.shape[1])
            # print(l0.weight.shape[2], l_combined.weight.shape[2])
            # print(l0.weight.shape[3], l_combined.weight.shape[3])

            if l0.weight.shape[1] == l_combined.weight.shape[1]:
                # print("first conv2d")
                # the input is equal e.g. at the very first conv2d.
                l_combined.weight.data[0:output_feature_number, :, :, :] += l0.weight.data
                l_combined.weight.data[output_feature_number:2 * output_feature_number, :, :, :] += l1.weight.data
                l_combined.weight.data[2 * output_feature_number:3 * output_feature_number, :, :, :] += l2.weight.data
                l_combined.weight.data[3 * output_feature_number:4 * output_feature_number, :, :, :] += l3.weight.data

                l_combined.bias.data[0:output_feature_number] = l0.bias.data
                l_combined.bias.data[output_feature_number:2 * output_feature_number] += l1.bias.data
                l_combined.bias.data[2 * output_feature_number:3 * output_feature_number] += l2.bias.data
                l_combined.bias.data[3 * output_feature_number:4 * output_feature_number] += l3.bias.data
            else:
                # print("inner conv2d")
                # there 4 times as many input channels
                l_combined.weight.data[0:output_feature_number, 0:input_feature_number, :, :] += l0.weight.data
                l_combined.weight.data[output_feature_number:2 * output_feature_number,
                input_feature_number:2 * input_feature_number, :, :] += l1.weight.data
                l_combined.weight.data[2 * output_feature_number:3 * output_feature_number,
                2 * input_feature_number:3 * input_feature_number, :, :] += l2.weight.data
                l_combined.weight.data[3 * output_feature_number:4 * output_feature_number,
                3 * input_feature_number:4 * input_feature_number, :, :] += l3.weight.data

                l_combined.bias.data[0:output_feature_number] = l0.bias.data
                l_combined.bias.data[output_feature_number:2 * output_feature_number] += l1.bias.data
                l_combined.bias.data[2 * output_feature_number:3 * output_feature_number] += l2.bias.data
                l_combined.bias.data[3 * output_feature_number:4 * output_feature_number] += l3.bias.data

            # there is third case here because convolutions are not used as the last layer

        if isinstance(l_combined, torch.nn.BatchNorm2d):

            l_combined.weight.data = torch.cat([l0.weight.data,l1.weight.data,l2.weight.data,l3.weight.data])
            l_combined.bias.data = torch.cat([l0.bias.data,l1.bias.data,l2.bias.data,l3.bias.data])
            l_combined.running_mean.data = torch.cat([l0.running_mean.data,l1.running_mean.data,l2.running_mean.data,l3.running_mean.data])
            l_combined.running_var.data = torch.cat([l0.running_var.data,l1.running_var.data,l2.running_var.data,l3.running_var.data])


if __name__ == '__main__':

    seed = 0
    b = 256
    e = 40
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device.
    device = "cpu"
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

    model0 = VGG("VGG11", num_classes).to(device)
    model1 = VGG("VGG11", num_classes).to(device)
    model2 = VGG("VGG11", num_classes).to(device)
    model3 = VGG("VGG11", num_classes).to(device)

    model0.load_state_dict(torch.load("vgg11_0.pt"))
    model1.load_state_dict(torch.load("vgg11_1.pt"))
    model2.load_state_dict(torch.load("vgg11_2.pt"))
    model3.load_state_dict(torch.load("vgg11_3.pt"))

    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()

    #for layer in model0.modules():
    #    if isinstance(layer, torch.nn.BatchNorm2d):
    #        print(layer)
    #        print(layer.weight.shape) # [64] learnable scale parameter
    #        print(layer.bias.shape) # [64] learnable bias
    #        print(layer.running_mean.shape) # [64] # mean
    #        print(layer.running_var.shape) # [64] # var
    #        # all 4 parameters are needed
    #        exit()
    #exit()

    #_, acc = evaluate_model(model0, test_loader, device)
    #print(acc)
    #0.8589999675750732
    #_, acc = evaluate_model(model1, test_loader, device)
    #print(acc)
    # 0.8570999503135681
    #_, acc = evaluate_model(model2, test_loader, device)
    #print(acc)
    # 0.8579999804496765
    #_, acc = evaluate_model(model3, test_loader, device)
    #print(acc)
    # 0.8580999970436096

    #model0.to("cpu")
    #model1.to("cpu")
    #model2.to("cpu")
    #model3.to("cpu")
    #_, acc = evaluate_model_output_averaging_four_models(model0,model1,model2,model3,test_loader, "cpu")
    #print(acc)
    # 0.8837000131607056


    model0.to("cpu")
    model1.to("cpu")
    model2.to("cpu")
    model3.to("cpu")
    #combined_model = CombinedVGGCustomBatch("VGG11", num_classes).to("cpu")
    combined_model = CombinedVGG("VGG11", num_classes).to("cpu")
    for param in combined_model.parameters():
        param.data = torch.zeros_like(param.data)

    #combine_models_VGG_BatchNorm(model0,model1,model2,model3,combined_model)
    combine_models_VGG(model0,model1,model2,model3,combined_model)

    #del model0
    #del model1
    #del model2
    #del model3

    combined_model.eval()


    #_, acc = evaluate_model(combined_model,test_loader, "cpu")
    #_, acc = evaluate_model(combined_model, valid_loader, "cpu")
    #print(acc)
    # 0.8837000131607056 test loader using 4 batchnorms sequentially on part of the data
    # 0.8837000131607056 test loader using a big batchnormlayer

    # prune

    imp = tp.importance.MagnitudeImportance(p=2)
    pruner_combined = tp.pruner.MagnitudePruner(
        combined_model,
        example_inputs,
        importance=imp,
        ch_sparsity=0.75,
        root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
        ignored_layers=[combined_model.classifier],
    )
    pruner_combined.step()



    #for lcomb,l0 in zip(combined_model.modules(),model0.modules()):
    #    if isLeafLayer(l0):
    #        print(l0)
    #        print(lcomb)
    # exactly the same architecture as expected

    # finetune
    del model0
    del model1
    del model2
    del model3
    del imp
    del pruner_combined

    learning_rate = 0.01
    optimizer = torch.optim.SGD(combined_model.parameters(), lr=learning_rate, momentum=0.9)

    test_loss_list = []
    test_accuracy_list = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


        test_loss, test_accuracy = evaluate_model(combined_model, test_loader, device)
        print(f"test accuracy after {epoch+1} epochs: {test_accuracy}")

        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

    combined_model.eval()
    _, accuracy_combined = evaluate_model(combined_model, test_loader, device)
    print(accuracy_combined)

    """
    test accuracy after 1 epochs: 0.7907999753952026
    test accuracy after 2 epochs: 0.7976999878883362
    test accuracy after 3 epochs: 0.7861999869346619
    test accuracy after 4 epochs: 0.8216999769210815
    test accuracy after 5 epochs: 0.8289999961853027
    test accuracy after 6 epochs: 0.8210999965667725
    test accuracy after 7 epochs: 0.8208000063896179
    test accuracy after 8 epochs: 0.8193999528884888
    test accuracy after 9 epochs: 0.8263999819755554
    test accuracy after 10 epochs: 0.8111000061035156
    test accuracy after 11 epochs: 0.8278999924659729
    test accuracy after 12 epochs: 0.8315999507904053
    test accuracy after 13 epochs: 0.8294999599456787
    test accuracy after 14 epochs: 0.8086000084877014
    test accuracy after 15 epochs: 0.8283999562263489
    test accuracy after 16 epochs: 0.8260999917984009
    test accuracy after 17 epochs: 0.8348000049591064
    test accuracy after 18 epochs: 0.8384999632835388
    test accuracy after 19 epochs: 0.8452000021934509
    test accuracy after 20 epochs: 0.8407999873161316
    test accuracy after 21 epochs: 0.8330999612808228
    test accuracy after 22 epochs: 0.8359999656677246
    test accuracy after 23 epochs: 0.840399980545044
    test accuracy after 24 epochs: 0.8463999629020691
    test accuracy after 25 epochs: 0.8486999869346619
    test accuracy after 26 epochs: 0.8486999869346619
    test accuracy after 27 epochs: 0.8463000059127808
    test accuracy after 28 epochs: 0.8453999757766724
    test accuracy after 29 epochs: 0.8477999567985535
    test accuracy after 30 epochs: 0.847000002861023
    test accuracy after 31 epochs: 0.8496999740600586
    test accuracy after 32 epochs: 0.8364999890327454
    test accuracy after 33 epochs: 0.8420999646186829
    test accuracy after 34 epochs: 0.8466999530792236
    test accuracy after 35 epochs: 0.8445999622344971
    test accuracy after 36 epochs: 0.8411999940872192
    test accuracy after 37 epochs: 0.8429999947547913
    test accuracy after 38 epochs: 0.840999960899353
    test accuracy after 39 epochs: 0.851099967956543
    test accuracy after 40 epochs: 0.8551999926567078
    0.8551999926567078

    # not finished yet, the accuracy keeps increasing
    # a lot of finetuning epochs needed here, maybe the ensemble accuracy will not be reached
    """