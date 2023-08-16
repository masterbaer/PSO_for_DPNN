"""
Here we test if the models can improve with finetuning.
"""
import torch
import torchvision
import torch_pruning as tp
from matplotlib import pyplot as plt

from dataloader import set_all_seeds, get_dataloaders_cifar10
from model import NeuralNetwork, CombinedNeuralNetwork

def get_number_of_zero_params(model):
    number_of_zeros = 0
    number_of_params = 0

    for i, layer in enumerate(model.modules()):
        number_of_zeros_local = 0
        number_of_params_local = 0

        if isinstance(layer, torch.nn.Linear):

            for param in layer.parameters():
                # count number of zeros
                # https://discuss.pytorch.org/t/how-to-count-the-number-of-zero-weights-in-a-pytorch-model/13549/2
                number_of_zeros_local += param.numel() - param.nonzero().size(0)
                number_of_params_local += param.numel()

        number_of_zeros += number_of_zeros_local
        number_of_params += number_of_params_local

    return number_of_zeros, number_of_params

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
    e = 30

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

    combined_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    combined_state_dict = torch.load("ex1_combined_model.pt")
    combined_model.load_state_dict(combined_state_dict)

    combined_model_scratch = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    combined_scratch_state_dict = torch.load("ex1_combined_model_scratch.pt")
    combined_model_scratch.load_state_dict(combined_scratch_state_dict)

    pruned_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
    pruned_model.load_state_dict(combined_state_dict)

    # sparsities from 0.0 to 0.9
    sparsities = [i/10 for i in range(0, 10)]

    accuracy_sparsity_list = []
    accuracy_sparsity_list_scratch = []
    finetune_after_ten_list = []
    finetune_after_ten_list_scratch = []
    finetune_after_thirty_list = []
    finetune_after_thirty_list_scratch = []

    for i, sparsity in enumerate(sparsities):

        #for i, layer in enumerate(combined_model.modules()):
        #    if isinstance(layer, torch.nn.Linear):
        #        print(layer.bias.shape, layer.weight.shape)
        #print("------")

        pruned_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
        pruned_model.load_state_dict(combined_state_dict)

        pruned_scratch_model = CombinedNeuralNetwork(image_shape[1] * image_shape[2] * image_shape[3], num_classes)
        pruned_scratch_model.load_state_dict(combined_scratch_state_dict)

        pruned_model.eval()
        pruned_scratch_model.eval()

        # do pruning here
        # see https://github.com/VainF/Torch-Pruning/wiki/4.-High%E2%80%90level-Pruners
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_combined = tp.pruner.MagnitudePruner(
            pruned_model,
            example_inputs,
            importance=imp,
            ch_sparsity=sparsity,
            root_module_types=[torch.nn.Linear],
            ignored_layers=[pruned_model.fc4],
        )
        pruner_combined.step()

        imp_scratch = tp.importance.MagnitudeImportance(p=2)
        pruner_combined_scratch = tp.pruner.MagnitudePruner(
            pruned_scratch_model,
            example_inputs,
            importance=imp,
            ch_sparsity=sparsity,
            root_module_types=[torch.nn.Linear],
            ignored_layers=[pruned_scratch_model.fc4],
        )
        pruner_combined_scratch.step()

        _, acc = evaluate_model(pruned_model, test_loader)
        print("ensemble accuracy after pruning: ", acc)

        _, acc_scratch = evaluate_model(pruned_scratch_model, test_loader)
        print("scratch accuracy after pruning: ", acc_scratch)

        accuracy_sparsity_list.append(acc)
        accuracy_sparsity_list_scratch.append(acc_scratch)

        # finetune

        optimizer_combined = torch.optim.SGD(pruned_model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer_combined_scratch = torch.optim.SGD(pruned_scratch_model.parameters(), lr=learning_rate,
                                                     momentum=0.9)
        for epoch in range(e):

            # model0.train()
            pruned_model.train()
            pruned_scratch_model.train()

            for i, (inputs, labels) in enumerate(train_loader):
                optimizer_combined.zero_grad()
                outputs_combined = pruned_model(inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs_combined, labels)
                loss.backward()
                optimizer_combined.step()

                optimizer_combined_scratch.zero_grad()
                outputs_combined_scratch = pruned_scratch_model(inputs)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss_combined_scratch = loss_fn(outputs_combined_scratch, labels)
                loss_combined_scratch.backward()
                optimizer_combined_scratch.step()

            if epoch + 1 == 10:
                _, acc_finetune_ten = evaluate_model(pruned_model, test_loader)
                print("finetune accuracy after 10 epochs: ", acc_finetune_ten)

                _, acc_scratch_finetune_ten = evaluate_model(pruned_scratch_model, test_loader)
                print("finetune accuracy after 10 epochs scratch model: ", acc_scratch_finetune_ten)

                finetune_after_ten_list.append(acc_finetune_ten)
                finetune_after_ten_list_scratch.append(acc_scratch_finetune_ten)

        # evaluate after full training:
        _, acc_finetune_last = evaluate_model(pruned_model, test_loader)
        print("finetune accuracy after full finetuning: ", acc_finetune_last)

        _, acc_scratch_finetune_last = evaluate_model(pruned_scratch_model, test_loader)
        print("finetune accuracy after full finetuning scratch model: ", acc_scratch_finetune_last)

        finetune_after_thirty_list.append(acc_finetune_last)
        finetune_after_thirty_list_scratch.append(acc_scratch_finetune_last)



    plt.plot(sparsities, accuracy_sparsity_list, label="prune ensemble w.o. finetuning", marker="o")
    plt.plot(sparsities, accuracy_sparsity_list_scratch, label="prune large model w.o. finetuning", marker="o")
    plt.plot(sparsities, finetune_after_ten_list, label="prune ensemble with finetuning 10 epochs", marker="o")
    plt.plot(sparsities, finetune_after_thirty_list, label="prune ensemble with finetuning 30 epochs", marker="o")
    plt.plot(sparsities, finetune_after_ten_list_scratch, label="prune large model with finetuning 10 epochs", marker="o")
    plt.plot(sparsities, finetune_after_thirty_list_scratch, label="prune large model with finetuning 30 epochs", marker="o")

    accuracies_0 = torch.load("ex1_test_accuracy0.pt")
    plt.plot([0.75], [accuracies_0[-1]], label="model 0", marker="s")

    plt.legend()
    plt.xlabel("sparsities")
    plt.ylabel("accuracy")
    plt.savefig("ex3_accuracies.png")
