import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.to_frame().values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return train_losses

if __name__ == "__main__":
    
    np.random.seed(42)
    n_samples = 100000
    X1 = np.random.choice([1, -1], size=n_samples, p=[0.7, 0.3])
    X2 = np.random.choice([1, -1], size=n_samples, p=[0.4, 0.6])
    X3 = np.random.choice([1, -1], size=n_samples, p=[0.9, 0.1])

    probabilities = np.random.uniform(0.3, 0.7, size=3)
    print(probabilities)
    X4_to_X6 = []
    for p in probabilities:
        X = np.random.choice([1, -1], size=n_samples, p=[p, 1-p])
        X4_to_X6.append(X)

    X4_to_X6 = np.column_stack(X4_to_X6)

    X = np.column_stack((X1, X2, X3, X4_to_X6))
    eps = np.random.normal(0, 1, size=n_samples)

    Y = X1 * np.sin(X1) + 2*X2 * X3 + eps

    X = pd.DataFrame(X, columns = [f'X{i}' for i in range(1, 7)])
    Y = pd.Series(Y, name = 'Y')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    input_size = X.shape[1]
    hidden_size = 100
    output_size = 1
    learning_rate = 0.01
    num_epochs = 200
    batch_size = 1024

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = train_model(model, criterion, optimizer, train_loader, num_epochs)

    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSELoss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('b_e_MLP_results/train_curve.png')

    torch.save(model.state_dict(), 'b_e_MLP_results/mlp_model.pth')
