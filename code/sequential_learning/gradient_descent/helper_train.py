import torch
import time

from helper_evaluation import compute_accuracy


# TRAIN MODEL (NON-PARALLEL)
def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None):
    start = time.perf_counter()

    # Initialize history lists for loss, training accuracy, and validation accuracy.
    loss_history, train_acc_history, valid_acc_history = [], [], []

    # ACTUAL TRAINING STARTS HERE.
    for epoch in range(num_epochs):  # Loop over epochs.

        model.train()  ## Set model to training mode.

        #  Thus, layers like dropout, batchnorm etc. which behave differently on
        #  train and test procedures know what is going on and can behave accordingly.

        for batch_idx, (features, targets) in enumerate(train_loader):  # Loop over mini batches.

            # CONVERT DATASET TO USED DEVICE.
            features = features.to(device)
            targets = targets.to(device)
            #
            # FORWARD & BACKWARD PASS
            logits = model(features)  # Forward pass: Apply model to samples to calculate output.
            loss = torch.nn.functional.cross_entropy(logits,
                                                     targets)  # cross-entropy loss for multiclass classification
            optimizer.zero_grad()  # Zero out gradients from former step.
            loss.backward()  # Backward pass: Compute gradients of loss w.r.t weights with backpropagation
            optimizer.step()  # Update model parameters via optimizer object, i.e. perform single optimization step.
            loss_history.append(loss.item())  # Logging.

            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        # VALIDATION STARTS HERE.
        #
        # Set model to evaluation mode.
        model.eval()

        with torch.no_grad():  # Context-manager that disables gradient calculation to reduce memory consumption.

            # COMPUTE ACCURACY OF CURRENT MODEL PREDICTIONS ON TRAINING + VALIDATION DATASETS.
            train_acc = compute_accuracy(model, train_loader, device=device)  # Compute accuracy on training data.
            valid_acc = compute_accuracy(model, valid_loader, device=device)  # Compute accuracy on validation data.

            print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f} '
                  f'| Validation: {valid_acc :.2f}')

            train_acc_history.append(train_acc.item())  # Append training accuracy to history list
            valid_acc_history.append(valid_acc.item())  # Append validation accuracy to history list.

        elapsed = (time.perf_counter() - start) / 60  # Measure training time per epoch.
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None: scheduler.step(valid_acc_history[-1])

    elapsed = (time.perf_counter() - start) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    # FINAL TESTING STARTS HERE.
    #
    test_acc = compute_accuracy(model, test_loader, device=device)  # Compute accuracy on test data.
    print(f"Test accuracy {test_acc :.2f}")

    return loss_history, train_acc_history, valid_acc_history
