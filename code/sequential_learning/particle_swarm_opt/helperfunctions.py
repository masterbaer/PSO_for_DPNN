import torch
from torch import nn


def evaluate_model(model, test_data_loader, device):
    #  Compute Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    number_of_batches = len(test_data_loader)
    correct_pred, num_examples = 0, 0

    for i, (vinputs, vlabels) in enumerate(test_data_loader):  # Loop over batches in data.
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


def evaluate_ensemble(models, test_data_loader, device):
    #  Compute Loss
    correct_pred, num_examples = 0, 0

    for i, (inputs, labels) in enumerate(test_data_loader):  # Loop over batches in data.
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = []  # shape (particles, batchsize, classes)
        for model in models:
            model.to(device)
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                output = model(inputs)
            predictions.append(output)
            model.to("cpu")
        aggregated_predictions = torch.stack(predictions).mean(dim=0)  # (calc mean --> shape (batchsize,classes) )

        _, predicted = torch.max(aggregated_predictions,
                                 dim=1)  # Determine class with max. probability for each sample.
        num_examples += labels.size(0)  # Update overall number of considered samples.
        correct_pred += (predicted == labels).sum()  # Update overall number of correct predictions.

    accuracy = (correct_pred.float() / num_examples).item()
    return accuracy


def evaluate_position_single_batch(model, inputs, labels, device):
    #  Compute Loss
    model = model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    predictions = model(inputs)  # Calculate model output.
    _, predicted = torch.max(predictions, dim=1)  # Determine class with max. probability for each sample.
    num_examples = labels.size(0)  # Update overall number of considered samples.
    correct_pred = (predicted == labels).sum()  # Update overall number of correct predictions.
    loss_total = loss_fn(predictions, labels).item()

    # loss_per_sample = loss / number_of_samples
    accuracy = (correct_pred.float() / num_examples).item()
    return loss_total, accuracy


def reset_all_weights(model: nn.Module) -> None:
    """
    copied from: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/11

    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
