import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Iterable


class LinComb(nn.Module):
    r"""
    Applies the LinComb activation function:
    
    :math:`\text{LinComb}(x) = \sum_{i=1}^{n} w_i \cdot F_i(x)`
    
    LinComb is an analog of linear combination of a set of activation functions.
    Thus, is more flexible and adaptable to the data in exchange for speed.
    
    Args:
        activations: List of activation functions.
        
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> activations = [nn.ReLU(), nn.Sigmoid()]
        >>> m = LinComb(activation_functions)
        >>> input = torch.randn(10)
        >>> output = m(input)
    """
    
    def __init__(self, activations: Iterable[str]):
        super(LinComb, self).__init__()
        self.activation_functions = nn.ModuleList(activations)
        self.weights = nn.Parameter(torch.Tensor(len(activations)))

        self.weights.data.uniform_(-1, 1)

    def forward(self, input) -> Tensor:
        activations = [self.weights[i] * self.activations[i]
                       (input) for i in range(len(self.activation_functions))]
        return torch.sum(torch.stack(activations), dim=0)


class NormLinComb(nn.Module):
    r"""
    Applies the LinComb activation function:
    
    :math:`\text{NormLinComb}(x) = \frac{\sum_{i=1}^{n} w_i \cdot F_i(x)}{\|\|W\|\|}`
    
    A version of `LinComb` with normalized linear combination using weight's norm.
    
    Args:
        activations (): List of activation functions.
        
    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> activations = [nn.ReLU(), nn.Sigmoid()]
        >>> m = NormLinComb(activation_functions)
        >>> input = torch.randn(10)
        >>> output = m(input)
    """
    
    def __init__(self, activations: Iterable[str]):
        super(LinComb, self).__init__()
        self.activation_functions = nn.ModuleList(activations)
        self.weights = nn.Parameter(torch.Tensor(len(activations)))

        self.weights.data.uniform_(-1, 1)

    def forward(self, input) -> Tensor:
        activations = [self.weights[i] * self.activations[i]
                       (input) for i in range(len(self.activation_functions))]
        output = torch.sum(torch.stack(activations), dim=0)
        return output / torch.norm(self.norm)