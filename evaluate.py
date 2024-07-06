import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from Models.cnn_model import CNN
from Utils.data_preprocessing import load_data

# Load data
_, _, val_data, val_labels = load_data()

# Create DataLoader
val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('cnn_model.pth'))  # Adjust the path as needed
model.eval()

# Evaluate model
criterion = torch.nn.MSELoss()
val_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        predictions.extend(outputs.numpy())
        actuals.extend(labels.numpy())

val_loss /= len(val_loader)
print(f"Validation Loss: {val_loss:.4f}")

# Save predictions and actual values for further analysis
predictions = np.array(predictions)
actuals = np.array(actuals)
np.save('predictions.npy', predictions)
np.save('actuals.npy', actuals)
