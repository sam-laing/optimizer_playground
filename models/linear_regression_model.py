import torch.nn as nn
import torch
import numpy as np

from torch.nn import Parameter
from torch import Tensor

class LinearRegressionModel(nn.Module):
    def __init__(
            self, dim_input: int, dim_output: int, 
            model_seed: int=99,
            has_bias = False,  
            separate_bias: bool = True,
        ):
        super(LinearRegressionModel, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.model_seed = model_seed
        self.separate_bias = separate_bias
        self.has_bias = has_bias


        self.model_generator = torch.Generator()
        self.model_generator.manual_seed(model_seed)
        self.np_rng = np.random.RandomState(model_seed)


        if self.has_bias:
            if separate_bias:
                self.linear = nn.Linear(dim_input, dim_output, bias=True)
                # initialise weights with generator seed
                self.linear.weight = Parameter(
                    torch.randn(dim_output, dim_input, generator=self.model_generator))
                self.linear.bias = Parameter(
                    torch.randn(dim_output, generator=self.model_generator))
            else:
                self.W = Parameter(torch.randn(dim_input + 1, dim_output))
                self.W.data = torch.nn.init.xavier_uniform_(self.W.data)
        else:        
            self.W = Parameter(torch.randn(dim_input, dim_output))
            self.W.data = torch.nn.init.xavier_uniform_(self.W.data)

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass for linear regression model
        """
        if self.separate_bias:
            return self.linear(X)
        return X @ self.W
    
    def get_weights(self) -> Tensor:
        """
        Get weights of the model
        """
        return self.W

if __name__ == "__main__":
    # Example usage
    model = LinearRegressionModel(dim_input=200, dim_output=100, model_seed=42, separate_bias=False)
    X = torch.randn(300, 200)
    Y = model(X)


    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
