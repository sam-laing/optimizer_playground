from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

def make_loaders(
    dataset: Dataset,
    batch_size: int = 128,
    seed: int = 0,
    val_size: int = None,
    shuffle: bool = True,
):
    """
    Create data loaders for training and testing.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
    
    Returns:
        train and val loaders (randomly select a val set with seed ) if val_size is not None else return train loader alone
    """
    assert isinstance(dataset, Dataset), "dataset must be a torch.utils.data.Dataset"
    assert isinstance(batch_size, int), "batch_size must be an integer"
    if val_size is not None:
        # Randomly select a validation set
        indices = np.arange(len(dataset))
        np.random.seed(seed)
        np.random.shuffle(indices)
        val_size = int(len(dataset) * val_size)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        
        return train_loader, val_loader
    else:   
        # Use the entire dataset for training
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader