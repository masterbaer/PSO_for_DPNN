# COMPUTE ACCURACY (DATA-PARALLEL)
import torch


def compute_accuracy_ddp(model, data_loader):
    with torch.no_grad():  # Context-manager that disables gradient calculation to reduce memory consumption.

        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_samples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            # Convert dataset to CUDA tensors.
            features = features.cuda()
            targets = targets.float().cuda()
            #
            # Calculate predictions of current model on features of input data.
            logits = model(features)  # Calculate model prediction.
            _, predicted_labels = torch.max(logits, 1)  # Get class with the highest score.
            num_samples += targets.size(0)  # Accumulate overall number of samples.
            correct_pred += (predicted_labels == targets).sum()  # Accumulate number of correctly predicted samples.

    # Return accuracy as percentage of correctly predicted samples.
    return correct_pred.float() / num_samples * 100  # Return accuracy.


# COMPUTE NUMBER OF CORRECTLY PREDICTED + OVERALL NUMBER OF SAMPLES (DATA-PARALLEL)
def get_right_ddp(model, data_loader):
    with torch.no_grad():  # Context-manager that disables gradient calculation to reduce memory consumption.

        # Initialize number of correctly predicted samples + overall number of samples.
        correct_pred, num_samples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.float().cuda()

            # Calculate predictions of current model on features of input data.
            logits = model(features)  # Calculate model prediction.
            _, predicted_labels = torch.max(logits, dim=1)  # Get class with the highest score.

            num_samples += targets.size(0)  # Accumulate overall number of samples.
            correct_pred += (predicted_labels == targets).sum()  # Accumulate number of correctly predicted samples.

        return correct_pred, num_samples
