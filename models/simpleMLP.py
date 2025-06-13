import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Module, Sequential
from torch import Tensor
from typing import Tuple, Dict, Any, Optional


class SimpleMLP(nn.Module):
    """    
    simple MLP with hidden layers. Optional dropout and layer normalization/batch normalization/RMS normalization

    Inputs:   
        dim_input: Number of input features (required)
        dim_output: Number of output features (required)
        hidden_layers: List of hidden layer sizes (required)
        activation: Activation function to use (relu, sigmoid, tanh, softmax) ... default is relu
        dropout: Dropout rate (0.0 to 1.0) ... default is 0.0
        batch_norm: Use batch normalization (True/False) ... default is True
        layer_norm: Use layer normalization (True/False) ... default is False
        rms_norm: Use RMS normalization (True/False) ... default is False
    """

    def __init__(
        self, dim_input: int, dim_output: int, hidden_layers: list,
        model_seed: int = 0, activation: str = "relu", dropout: float = 0.0,
        batch_norm: bool = True, layer_norm: bool = False, rms_norm: bool = False
    ):
        super(SimpleMLP, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.rms_norm = rms_norm
        self.model_seed = model_seed

        self.model_generator = torch.Generator()
        self.model_generator.manual_seed(model_seed)
        self.np_rng = np.random.RandomState(model_seed)

        self.model = self._build_model()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=self.model_generator)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.RMSNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
       


            
            

    def _build_model(self) -> Sequential:
        layers = []
        input_size = self.dim_input

        # Add hidden layers
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            if self.rms_norm:
                layers.append(nn.RMSNorm(hidden_size))

            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif self.activation == "tanh":
                layers.append(nn.Tanh())
            
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(input_size, self.dim_output))

        return Sequential(*layers)



if __name__ == "__main__":
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    from datasets import MnistDataset, make_loaders
    

    dataset = MnistDataset(root="./data")
    print("making loaders")
    train_loader, val_loader = make_loaders(
        dataset, batch_size=128, seed=0, val_size=0.2, shuffle=True
    )

    print("train_loader created")
    for X, Y in train_loader:
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        print(type(X), type(Y))
        break
    """ 
    model = SimpleMLP(
        dim_input=28*28, 
        dim_output=10, 
        hidden_layers=[128, 64], 
        activation="relu", 
        dropout=0.2, 
        batch_norm=True, 
        layer_norm=False, 
        rms_norm=False,
        model_seed=42
    )
    model.train()

    for X, Y in train_loader:
        print(type(X))
        X = X.view(X.size(0), -1)
        out = model(X)
        loss = F.cross_entropy(out, Y)
        print(f"Loss: {loss.item()}")
        loss.backward()
        break
        """