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

        self.model_generator = torch.Generator()
        self.model_generator.manual_seed(seed)
        self.np_rng = np.random.RandomState(seed)

        self.W = Parameter(torch.randn(dim_input + 1, num_classes))
        self.W.data = torch.nn.init.xavier_uniform_(self.W.data)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=self.model_generator)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
