import torch


# COMPUTE ACCURACY (NON-PARALLEL)
def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):  # Loop over batches in data.
            features = features.to(device)
            targets = targets.float().to(device)
            logits = model(features)  # Calculate model output.
            _, predicted = torch.max(logits, dim=1)  # Determine class with max. probability for each sample.
            num_examples += targets.size(0)  # Update overall number of considered samples.
            correct_pred += (predicted == targets).sum()  # Update overall number of correct predictions.
    return correct_pred.float() / num_examples * 100  # Return accuracy in percent
