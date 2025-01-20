import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
def parse_prop_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return np.array([list(map(float, line.split())) for line in data])
def parse_displacement_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    displacements = [float(line.strip()) for line in data]
    return max(displacements)

# Prepare dataset
def prepare_dataset(prop_files, disp_files):
    inputs = []
    outputs = []
    for prop_file, disp_file in zip(prop_files, disp_files):
        prop_matrix = parse_prop_file(prop_file)
        max_disp = parse_displacement_file(disp_file)
        inputs.append(prop_matrix)
        outputs.append(max_disp)
    inputs = np.expand_dims(np.array(inputs), axis=-1)  
    outputs = np.array(outputs)
    return inputs, outputs

# Load data
prop_files = ["path_to/prop1.dat", "path_to/prop2.dat"]  
disp_files = ["path_to/disp1.data", "path_to/disp2.data"] 
X, y = prepare_dataset(prop_files, disp_files)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Predict maximum displacement for new data
new_prop_file = "path_to/new_prop.dat"
new_disp_file = "path_to/new_disp.data"
new_X = parse_prop_file(new_prop_file)
new_X = np.expand_dims(new_X, axis=(0, -1))  # Add batch and channel dimensions
predicted_disp = model.predict(new_X)
print(f"Predicted Maximum Displacement: {predicted_disp[0][0]}")
