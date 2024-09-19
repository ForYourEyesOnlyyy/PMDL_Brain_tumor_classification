import os
import torch
import json
from collections import Counter

model_version_dir = "models"
data_version_dir = "datasets"

if not os.path.exists(data_version_dir):
    os.makedirs(data_version_dir)

if not os.path.exists(model_version_dir):
    os.makedirs(model_version_dir)

def save_model(model, optimizer, epoch, loss, accuracy, version):
    """
    Save the model, optimizer, and metadata for versioning.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - epoch (int): Current epoch.
    - loss (float): The loss value.
    - accuracy (float): The accuracy value.
    - version (str): Version identifier (e.g., 'v1.0.0').
    """
    model_dir = os.path.join(model_version_dir, version)
    
    # Ensure the version directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model state
    model_path = os.path.join(model_dir, 'model.pt')  # Changed to `.pt`
    torch.save(model.state_dict(), model_path)
    
    # Save the optimizer state
    optimizer_path = os.path.join(model_dir, 'optimizer.pt')  # Changed to `.pt`
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Model version {version} saved as .pt!")

def load_model(model, optimizer, version):
    """
    Load a specific version of the model and optimizer.
    
    Args:
    - model (torch.nn.Module): The model instance to load into.
    - optimizer (torch.optim.Optimizer): The optimizer to load into.
    - version (str): Version identifier (e.g., 'v1.0.0').
    """
    model_dir = os.path.join(model_version_dir, version)
    
    # Load model state
    model_path = os.path.join(model_dir, 'model.pt')  # Changed to `.pt`
    model.load_state_dict(torch.load(model_path))
    
    # Load optimizer state
    optimizer_path = os.path.join(model_dir, 'optimizer.pt')  # Changed to `.pt`
    optimizer.load_state_dict(torch.load(optimizer_path))
    
    # Load metadata (optional)
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Model version {version} loaded! Epoch: {metadata['epoch']}, Loss: {metadata['loss']}, Accuracy: {metadata['accuracy']}")
    
    return metadata

def get_next_version(type):
    if type == "data":
        version_dir = data_version_dir
    else:
        version_dir = model_version_dir
    versions = [d for d in os.listdir(version_dir) if os.path.isdir(os.path.join(version_dir, d))]
    if not versions:
        return 'v1.0.0'
    
    latest_version = sorted(versions)[-1]
    major, minor, patch = map(int, latest_version[1:].split('.'))
    patch += 1
    
    return f'v{major}.{minor}.{patch}'

def save_data(train_dataset, val_dataset, version):
    """
    Save the train and validation datasets as PyTorch datasets with versioning and metadata.
    
    Args:
    - train_dataset (torch.utils.data.Dataset): Training dataset (combined X and y).
    - val_dataset (torch.utils.data.Dataset): Validation dataset (combined X and y).
    - version (str): Version identifier (e.g., 'v1.0.0').
    """
    data_dir = os.path.join(data_version_dir, version)
    
    # Ensure the version directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the datasets
    torch.save(train_dataset, os.path.join(data_dir, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(data_dir, 'val_dataset.pt'))
    
    # Gather metadata for the train dataset
    train_size = len(train_dataset)
    # Assuming the dataset has labels at the second position in each item, count class occurrences
    class_counts = {}
    for _, label in train_dataset:
        class_counts[label.item()] = class_counts.get(label.item(), 0) + 1
    
    # Save metadata for datasets
    metadata = {
        'train_size': train_size,
        'class_counts': class_counts
    }
    metadata_path = os.path.join(data_dir, 'metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Dataset version {version} saved!")

def load_data(version):
    """
    Load a specific version of the train and validation datasets.
    
    Args:
    - version (str): Version identifier (e.g., 'v1.0.0').
    
    Returns:
    - datasets (dict): Loaded datasets including train_dataset and val_dataset.
    - metadata (dict): Metadata containing training information like class counts.
    """
    data_dir = os.path.join(data_version_dir, version)
    
    # Load datasets
    train_dataset = torch.load(os.path.join(data_dir, 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(data_dir, 'val_dataset.pt'))
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Dataset version {version} loaded! Train size: {metadata['train_size']}, Class counts: {metadata['class_counts']}")
    
    datasets = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset
    }
    
    return datasets, metadata
