import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ReGLU(torch.nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{ReLU} (xW + b) \odot (xV + c)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        size: The size of the last dimension of the input tensor.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> m = ReGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)

    """

    def __init__(self, size:int):
        super(ReGLU, self).__init__()
        self.linear = nn.Linear(size, size*2)

    def forward(self, x) -> Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * F.relu(b)


class GeGLU(torch.nn.Module):
    r"""
    Applies the GeGLU activation function, defined as:

    :math:`\text{GeGLU}(x) = \text{GELU} (xW + b) \odot (xV + c)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        size: The size of the last dimension of the input tensor.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> m = GeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, size:int):
        super(GeGLU, self).__init__()
        self.linear = nn.Linear(size, size*2)

    def forward(self, x) -> Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class SwiGLU(torch.nn.Module):
    r"""
    Applies the SwiGLU activation function, defined as:

    :math:`\sigma(x) =  \text{Swish} (xW + b) \odot (xV + c)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        size: The size of the last dimension of the input tensor.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> m = SwiGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, size:int):
        super(SwiGLU, self).__init__()
        self.linear = nn.Linear(size, size*2)

    def forward(self, x) -> Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * F.silu(b)
    
class SeGLU(torch.nn.Module):
    r"""
    Applies the SeGLU activation function, defined as:

    :math:`\text{SeGLU}(x) =  \text{SELU} (xW + b) \odot (xV + c)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        size: The size of the last dimension of the input tensor.

    Shape:
        - Input: :math:`(*)` where :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples::

        >>> m = SeGLU(20)
        >>> input = torch.randn(3, 20, 20)
        >>> output = m(input)
    """

    def __init__(self, size:int):
        super(SeGLU, self).__init__()
        self.linear = nn.Linear(size, size*2)

    def forward(self, x) -> Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return a * F.selu(b)

if __name__ == "__main__":
    pass    

