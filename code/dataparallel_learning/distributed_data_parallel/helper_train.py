import torch
import time

from result_plotter import save_plots
from helper_evaluation import get_right_ddp


def train_model_ddp(model, num_epochs, train_loader, valid_loader, optimizer):
    start = time.perf_counter()  # Measure training time.
    rank = torch.distributed.get_rank()  # Get local process ID (= rank).
    world_size = torch.distributed.get_world_size()  # Get overall number of processes.

    if rank == 0:  # Initialize history lists on root.
        loss_history, train_acc_history, valid_acc_history = [], [], []

        # ACTUAL TRAINING STARTS HERE.
    for epoch in range(num_epochs):  # Loop over epochs.

        train_loader.sampler.set_epoch(epoch)  # Set current epoch for distributed dataloader.

        model.train()  # Set model to training mode.

        for batch_idx, (features, targets) in enumerate(train_loader):  # Loop over mini batches.

            # Convert dataset to GPU device.
            features = features.cuda()
            targets = targets.cuda()

            # FORWARD & BACKWARD PASS
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Perform single optimization step (parameter update).
            #
            # LOGGING
            torch.distributed.all_reduce(loss)  # Sum up process-local mini-mini-batch losses.
            loss /= world_size  # Divide by number of processes.

            if rank == 0:
                # Append loss to history list.
                loss_history.append(loss.item())  # Append averaged loss to history.
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Averaged Loss: {loss:.4f}')

        # VALIDATION STARTS HERE.

        # Set model to evaluation mode.
        model.eval()  # Set model to evaluation mode.

        #
        with torch.no_grad():  # Context-manager that disables gradient calculation.
            # Get numbers of correctly classified samples and overall samples in training and validation dataset.
            right_train, num_train = get_right_ddp(model, train_loader)
            right_valid, num_valid = get_right_ddp(model, valid_loader)
            # Convert to torch tensors for collective communication.
            num_train = torch.Tensor([num_train]).cuda()
            num_valid = torch.Tensor([num_valid]).cuda()
            # Sum up numbers over all processes.
            torch.distributed.all_reduce(right_train)
            torch.distributed.all_reduce(right_valid)
            torch.distributed.all_reduce(num_train)
            torch.distributed.all_reduce(num_valid)
            # Calculate global training and validation accuracy.
            train_acc = right_train.item() / num_train.item()
            valid_acc = right_valid.item() / num_valid.item()

            if rank == 0:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Train: {train_acc :.2f} '
                      f'| Validation: {valid_acc :.2f}')
                train_acc_history.append(train_acc)
                valid_acc_history.append(valid_acc)

                # Append accuracy values to corresponding history lists.

        elapsed = (time.perf_counter() - start) / 60  # Measure training time elapsed after each epoch.

        Elapsed = torch.Tensor([elapsed]).cuda()

        # Calculate average training time elapsed after each epoch over all processes,
        torch.distributed.all_reduce(Elapsed)
        Elapsed /= world_size

        # Note that torch.distributed collective communication functions will only
        # work with torch tensors, i.e., floats, ints, etc. must be converted before!
        #
        if rank == 0:
            print('Time elapsed:', Elapsed.item(), 'min')

    # Print process-averaged training time after each epoch.
    elapsed = (time.perf_counter() - start) / 60  # Measure total training time.
    Elapsed = torch.Tensor([elapsed]).cuda()
    #
    # Calculate average total training time over all processes.
    torch.distributed.all_reduce(Elapsed)
    Elapsed /= world_size

    #
    if rank == 0:
        print('Total Training Time:', Elapsed.item(), 'min')
        # torch.save(loss_history, 'ddp_loss.pt')
        # torch.save(train_acc_history, 'ddp_train_acc.pt')
        # torch.save(valid_acc_history, 'ddp_valid_acc.pt')
        save_plots(loss_history, train_acc_history, valid_acc_history)
    return
