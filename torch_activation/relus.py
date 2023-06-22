import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class ShiLU(nn.Module):
    r"""
    Applies the ShiLU activation function:

    :math:`\text{ShiLU}(x) = \alpha \cdot \text{ReLU}(x) + \beta`

    Args:
        alpha (float, optional): Scaling factor for the positive part of the input. Default: 1.0.
        beta (float, optional): Bias term added to the output. Default: 0.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/ShiLU.png

    Examples::

        >>> m = torch_activation.ShiLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ShiLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, 
                 inplace: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
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

    :math:`\text{CReLU}(x) = \text{ReLU}(x) \oplus \text{ReLU}(-x)`

    Args:
        dim (int, optional): Dimension along which to concatenate in the output tensor. Default: 1
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*, C, *)` where :math:`*` means any number of additional dimensions
        - Output: :math:`(*, 2C, *)`

    Examples::

        >>> m = torch_activation.CReLU()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = torch_activation.CReLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x)
    """

    def __init__(self, dim: int = 0):
        super(CReLU, self).__init__()
        self.dim = dim

    def forward(self, x) -> Tensor:
        return F.relu(torch.cat((x, -x), dim=self.dim))


class ReLUN(nn.Module):
    r"""Applies the element-wise function:

    :math:`\text{ReLUN}(x) = \min(\text{ReLU}(x), n)`

    Args:
        n (float, optional): Upper bound for the function's output. Default is 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.ReLUN(n=6.0) # ReLU6
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.ReLUN(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)

    """

    def __init__(self, n: float = 1.0, inplace: bool = False):
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
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = torch_activation.SquaredReLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = torch_activation.SquaredReLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2)
        else:
            return F.relu(x).pow(2)


class StarReLU(nn.Module):
    r"""
    Applies the element-wise function:

    :math:`\text{StarReLU}(x) = s \cdot \text{ReLU}(x)^2 + b`

    Args:
        s (float, optional): Scaled factor for StarReLU, shared across channel. Default: 0.8944
        b (float, optional): Bias term for StarReLU, shared across channel. Default: -0.4472
        learnable (bool, optional): optionally make ``s`` and ``b`` trainable. Default: ``True``
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    .. image:: ../images/activation_images/SquaredReLU.png

    Examples::

        >>> m = torch_activation.StarReLU(s=1.0, b=0.0)
        >>> x = torch.randn(3, 384, 384)
        >>> output = m(x)

        >>> m = torch_activation.StarReLU(learnable=True, inplace=True)
        >>> x = torch.randn(3, 384, 384)
        >>> m(x)
    """

    def __init__(self, s: float = 0.8944, b: float = -0.4472, 
                 learnable: bool = True, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        if learnable:
            self.s = nn.Parameter(torch.tensor(s))
            self.b = nn.Parameter(torch.tensor(b))
        else:
            self.s = torch.tensor(s)
            self.b = torch.tensor(b)

    def forward(self, x) -> Tensor:
        if self.inplace:
            return F.relu_(x).pow_(2) \
                             .mul_(self.s) \
                             .add_(self.b)
        else:
            return self.s * F.relu(x).pow(2) + self.b


if __name__ == '__main__':
    pass
