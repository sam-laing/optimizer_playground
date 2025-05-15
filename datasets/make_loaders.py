from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

def make_loaders(
    dataset: Dataset,
    batch_size: int = 128,
    seed: int = 0,
    val_size: float = None,
    shuffle: bool = True,
):
    """
    Create data loaders for training and testing with isolated random state.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        seed: controlling both train-val and data loading generator 
        val_size: Size of validation set (if None, use the entire dataset for training)
    
    Returns:
        train and val loaders (randomly select a val set with seed ) if val_size is not None else return train loader alone
    """
    assert isinstance(dataset, Dataset), "dataset must be a torch.utils.data.Dataset"
    assert isinstance(batch_size, int), "batch_size must be an integer"

    rng = np.random.RandomState(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    if val_size is not None:
        indices = np.arange(len(dataset))
        rng.shuffle(indices)  
        val_size = int(len(dataset) * val_size)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        g = torch.Generator().manual_seed(seed)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, 
            generator = g if shuffle else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    else:   
        #if no val_size, just make everything train
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            generator=g if shuffle else None
        )
        return train_loader