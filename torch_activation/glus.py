import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ReGLU(nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{ReLU} (xW + b) \odot (xV + c)`

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1 

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = ReGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)

    """

    def __init__(self, dim: int = -1):
        super(ReGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=self.dim)
        return a * F.relu(b)


class GeGLU(nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{GELU} (xW + b) \odot (xV + c)`

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1 

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = GeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(GeGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)


class SwiGLU(nn.Module):
    r"""
    Applies the SwiGLU activation function, defined as:

    :math:`\sigma(x) =  \text{Swish} (xW + b) \odot (xV + c)`

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1 

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = SwiGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(SwiGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.silu(b)
    
class SeGLU(nn.Module):
    r"""
    Applies the SeGLU activation function, defined as:

    :math:`\text{SeGLU}(x) =  \text{SELU} (xW + b) \odot (xV + c)`

    Args:
        dim (int, optional): the dimension on which to split the input. Default: -1 

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::

        >>> m = SeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, dim: int = -1):
        super(SeGLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.selu(b)

if __name__ == "__main__":
    pass    

