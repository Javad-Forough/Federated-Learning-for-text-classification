import torch

def distribute_data(X_train, y_train, clients):
    # Send data to respective clients
    data = [(torch.tensor(X_train).float().send(client), torch.tensor(y_train).float().send(client)) for client in clients]
    
    return data