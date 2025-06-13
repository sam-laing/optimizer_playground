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
    



from typing import Iterator, List, Tuple
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
def make_train_val_test_loaders(
    dataset_name: str,
    batch_size: int = 64,
    val_size: float = 0.2,
    test_size: float = 0.1,
    shuffle: bool = True,
    seed: int = 0
):
        if dataset_name == "mnist":
            dataset = MNIST(
                root="./data",
                train=True,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.1307,), (0.3081,)),
                    ]
                ),
                download=True,
            )

            test_dataset = MNIST(
                root="./data",
                train=False,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.1307,), (0.3081,)),
                    ]
                ),
                download=True,
            )

        elif dataset_name == "fashion_mnist":
            dataset = FashionMNIST(
                root="./data",
                train=True,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.2860,), (0.3530,)),
                    ]
                ),
                download=True,
            )

            test_dataset = FashionMNIST(
                root="./data",
                train=False,
                transform=Compose(
                    [
                        ToTensor(),
                        Normalize((0.2860,), (0.3530,)),
                    ]
                ),
                download=True,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        generator = torch.Generator()
        generator.manual_seed(seed)

        # make a subset for validation 
        indices = np.arange(len(dataset))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        val_size = int(len(dataset) * val_size)

        train_subset, val_subset = indices[val_size:], indices[:val_size]
        train_dataset = torch.utils.data.Subset(dataset, train_subset)
        val_dataset = torch.utils.data.Subset(dataset, val_subset)
        test_dataset = torch.utils.data.Subset(test_dataset, indices[:int(len(test_dataset) * test_size)])


        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader




if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = make_train_val_test_loaders(
        dataset_name="mnist",
        batch_size=64,
        val_size=0.2,
        test_size=0.1,
        shuffle=True,
        seed=42
    )
    
    for X, Y in train_loader:
        print(f"Batch X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"Batch X type: {type(X)}, Y type: {type(Y)}")
        break

    for X, Y in val_loader:
        print(f"Validation Batch X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"Validation Batch X type: {type(X)}, Y type: {type(Y)}")
        break

    for X, Y in test_loader:
        print(f"Test Batch X shape: {X.shape}, Y shape: {Y.shape}")
        print(f"Test Batch X type: {type(X)}, Y type: {type(Y)}")
        break



