
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for epoch in range(8):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 400 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(
                    f"epoch: [{epoch}/3] loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
