import torch

def evaluate_model(model, data):
    # Evaluate the model
    model.eval()
    
    # Iterate through each client's data
    for i, (X, y) in enumerate(data):
        with torch.no_grad():
            # Make predictions
            pred = model(X)
            
            # Calculate accuracy
            acc = ((pred > 0.5).float() == y).float().mean().get().item()
            
            # Print accuracy
            print(f"Accuracy on client {i+1}'s data:", acc)
