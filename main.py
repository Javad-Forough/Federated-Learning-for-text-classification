import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from server_setup import setup_server
from client_setup import distribute_data
from training import train_model
from evaluation import evaluate_model
import pandas as pd

# Load the preprocessed dataset
def load_dataset(dataset_path):
    # Load the dataset from the specified path 
    dataset = pd.read_csv(dataset_path)
    
    # Assuming the last column contains the labels and the rest are features
    X = dataset.iloc[:, :-1].values  # Features
    y = dataset.iloc[:, -1].values   # Labels
    
    return X, y

X, y = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up server and clients
num_clients = 2 
clients, crypto_provider = setup_server(num_clients)

# Distribute data to clients
data_train = distribute_data(X_train, y_train, clients)

# Define model, loss function, and optimizer
input_size = len(X[0])  # X[0] contains the number of features
output_size = 1  # binary classification
hidden_size = 64 
model = SentimentClassifier(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model federatedly
for client_data in data_train:
    model = train_model(model, client_data, optimizer, epochs=5)  # Adjust epochs as needed

# Evaluate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = DataLoader(X,y, batch_size=32, shuffle=False)
accuracy = evaluate_model(model, data_loader, device)

print("Accuracy on test set:", accuracy)
