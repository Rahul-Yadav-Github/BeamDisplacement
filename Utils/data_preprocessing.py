import numpy as np
import glob

def load_data():
    topology_files = sorted(glob.glob('data/beam_topologies/*.npy'))
    displacement_files = sorted(glob.glob('data/displacements/*.npy'))
    
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    # Split data into training and validation sets
    split_ratio = 0.8
    split_index = int(len(topology_files) * split_ratio)

    for i, (topology_file, displacement_file) in enumerate(zip(topology_files, displacement_files)):
        topology = np.load(topology_file)
        displacement = np.load(displacement_file)
        
        if i < split_index:
            train_data.append(topology)
            train_labels.append(displacement)
        else:
            val_data.append(topology)
            val_labels.append(displacement)
    
    train_data = np.array(train_data).reshape(-1, 1, topology.shape[0], topology.shape[1])
    train_labels = np.array(train_labels).reshape(-1, 1)
    val_data = np.array(val_data).reshape(-1, 1, topology.shape[0], topology.shape[1])
    val_labels = np.array(val_labels).reshape(-1, 1)
    
    return train_data, train_labels, val_data, val_labels
