import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torch.nn import Parameter
from torch import Tensor

class LogisticRegressionModel(nn.Module):
    def __init__(self, dim_input: int, num_classes: int, seed: int=0):
        super(LogisticRegressionModel, self).__init__()
        self.dim_input = dim_input
        self.num_classes = num_classes
        self.seed = seed
        self._set_seed(seed)
        self.W = Parameter(torch.randn(dim_input + 1, num_classes))
        self.W.data = torch.nn.init.xavier_uniform_(self.W.data)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass for logistic regression model
        """
        return X @ self.W
    
    def get_weights(self) -> Tensor:
        """
        Get weights of the model
        """
        return self.W
