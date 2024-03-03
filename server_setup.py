import torch
import syft as sy

def setup_server(num_clients):
    # Set up PySyft's hook to intercept PyTorch functions
    hook = sy.TorchHook(torch)
    
    # Create virtual workers for each client
    clients = [sy.VirtualWorker(hook, id=f"client{i}") for i in range(1, num_clients + 1)]
    
    # Create a virtual worker for cryptographic operations
    crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")
    
    return clients, crypto_provider