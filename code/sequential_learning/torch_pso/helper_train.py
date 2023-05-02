import torch
import time

from helper_evaluation import compute_accuracy


# TRAIN MODEL (NON-PARALLEL)
def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer, criterion,
                device, logging_interval=50,
                scheduler=None):
    start = time.perf_counter()

    # Initialize history lists for loss, training accuracy, and validation accuracy.
    loss_history, train_acc_history, valid_acc_history = [], [], []

    # ACTUAL TRAINING STARTS HERE.
    for epoch in range(num_epochs):  # Loop over epochs.

        model.train()  # Set model to training mode.

        #  Thus, layers like dropout, batchnorm etc. which behave differently on
        #  train and test procedures know what is going on and can behave accordingly.

        for batch_idx, (features, targets) in enumerate(train_loader):  # Loop over mini batches.

            # CONVERT DATASET TO USED DEVICE.
            features = features.to(device)
            targets = targets.to(device)

            # The closure method is needed when the function has to be reevaluated multiple times,
            # here for evaluating each particles' loss. See https://pytorch.org/docs/stable/optim.html .

            # For implementation details of optim.step(closure) see torch_pso.optim.GenericPSO.step()
            # and torch_pso.optim.ParticleSwarmOptimizer.Particle.step().

            # The method optim.step(closure) with the particle swarm optimizer first takes one step on each
            # particle (and calculates each particles' loss after the step). Then the new loss on the best particle
            # is calculated again (even though it was already calculated) and returned.
            # Hence, there are num_particles + 1 (in an ideal scenario only num_particles) forward passes.

            def closure():
                optimizer.zero_grad()  # we will not use gradients for the update but change the parameters via PSO
                predictions = model(features)
                loss = criterion(predictions, targets)
                # loss.backward() # no backward pass used
                return loss

            loss = optimizer.step(closure)  # The loss of the best particle AFTER every particle took one step.
            # Needs num_particles + 1 forward passes.

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

        # The scheduler is not present for PSO since the learning rate is incorporated in the algorithm.
        # if scheduler is not None:
        #     scheduler.step(valid_acc_history[-1])

    elapsed = (time.perf_counter() - start) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    # FINAL TESTING STARTS HERE.
    #
    test_acc = compute_accuracy(model, test_loader, device=device)  # Compute accuracy on test data.
    print(f"Test accuracy {test_acc :.2f}")

    return loss_history, train_acc_history, valid_acc_history
