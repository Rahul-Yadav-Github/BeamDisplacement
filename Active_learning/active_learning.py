import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def active_learning_loop(model, train_data, train_labels, val_data, val_labels, num_iterations=10):
    for i in range(num_iterations):
        val_loader = DataLoader(TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32)), batch_size=32, shuffle=False)
        uncertainties = evaluate_uncertainty(model, val_loader)
        high_uncertainty_indices = np.argsort(uncertainties)[-10:]
        selected_data = val_data[high_uncertainty_indices]
        selected_labels = val_labels[high_uncertainty_indices]
        train_data = np.append(train_data, selected_data, axis=0)
        train_labels = np.append(train_labels, selected_labels, axis=0)
        train_loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32)), batch_size=32, shuffle=True)
        retrain_model(model, train_loader)

def evaluate_uncertainty(model, loader):
    uncertainties = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            uncertainty = torch.std(outputs).item()
            uncertainties.append(uncertainty)
    return np.array(uncertainties)

def retrain_model(model, train_loader, num_epochs=10):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
