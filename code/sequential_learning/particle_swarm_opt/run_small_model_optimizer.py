




def closure():
    optimizer.zero_grad()

    # Fetch a batch of data points
    inputs, labels = next(iter(dataloader))

    # Forward pass
    outputs = model(inputs)

    # Compute the loss
    loss = loss_function(outputs, labels)

    # Backward pass
    loss.backward()

    return loss