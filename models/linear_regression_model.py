import torch.nn as nn
import torch
import numpy as np

from torch.nn import Parameter
from torch import Tensor


class LinearRegressionModel(nn.Module):
    def __init__(self, dim_input: int, dim_output: int, seed: int=99):
        super(LinearRegressionModel, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.seed = seed
        self._set_seed(seed)
        self.W = Parameter(torch.randn(dim_input + 1, dim_output))
        self.W.data = torch.nn.init.xavier_uniform_(self.W.data)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass for linear regression model
        """
        return X @ self.W
    
    def get_weights(self) -> Tensor:
        """
        Get weights of the model
        """
        return self.W
