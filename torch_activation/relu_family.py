import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ShiLU(nn.Module):
    r"""
    Applies the ShiLU activation function:

    :math:`\text{ShiLU}(x) = \alpha * \max(0,x) + \beta`

    Args:
        alpha : Scaling factor for the positive part of the input. Default: 1.0.
        beta : Bias term added to the output. Default: 0.0.
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Attributes:
        alpha : Scaling factor for the positive part of the input. Default: 1.0.
        beta : Bias term added to the output. Default: 0.0.

    .. image:: ../images/activation_images/ShiLU.png

    Examples::
    
        >>> m = ShiLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = ShiLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, alpha=1.0, beta=0.0, inplace=False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            F.relu_(x)
            x.mul_(self.alpha)
            x.add_(self.beta)
            return x
        else:    
            return self.alpha * F.relu(x) + self.beta
        

class CReLU(nn.Module):
    r"""
    Applies the Concatenated Rectified Linear Unit activation function.

    :math:`\text{CReLU}(x) = \max(0,x) \oplus \max(0,-x)`

    Args:
        dim: Dimension along which to concatenate in the output tensor. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`

    .. image:: ../images/activation_images/CReLU.png

    Examples::

        >>> m = nn.CReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
        
        >>> m = CReLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """
    
    
    def __init__(self, dim=0):
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        return F.relu(torch.cat((x, -x), dim=self.dim))
        
        
class ReLUN(nn.Module):
    r"""Applies the element-wise function:

    :math:`\text{ReLUN}(x) = \min(\max(0,x), n)`

    Args:
        n: Upper bound for the function's output. Default is 1.0.
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Attributes:
        n: Upper bound for the function's output. Default is 1.0.
        
    .. image:: ../images/activation_images/ReLUN.png

    Examples::

        >>> m = nn.ReLUN(n=6.0) # ReLU6
        >>> x = torch.randn(2)
        >>> output = m(x)
        
        >>> m = nn.ReLUN(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)

    """
    def __init__(self, n=1.0, inplace=False):
        super(ReLUN, self).__init__()
        self.n = nn.Parameter(torch.tensor(n))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            x.clamp_max_(self.n.data)
            x.relu_()
            return x
        else:
            return torch.clamp(x, 0, self.n.data)
        
        
class SquaredReLU(nn.Module):
    r"""
    Applies the element-wise function:

    :math:`\text{SquaredReLU}(x) = \text{ReLU}(x)^2`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    .. image:: ../images/activation_images/SquaredReLU.png

    Examples::

        >>> m = nn.SquaredReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.SquaredReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """
    
    
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2)
        else:
            return F.relu(x).pow(2)
    
    
if __name__ == '__main__':
    pass