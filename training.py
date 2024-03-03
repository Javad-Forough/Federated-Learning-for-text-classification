import torch

def train_model(model, data, optimizer, epochs):
    # Train the model federatedly
    for epoch in range(epochs):
        for i, (X, y) in enumerate(data):
            # Send the model to the client
            model = model.send(X.location)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X)
            
            # Calculate the loss
            loss = torch.nn.functional.binary_cross_entropy(pred.view(-1), y)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Get back the updated model
            model = model.get()

    return model
