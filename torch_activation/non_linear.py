import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CoLU(nn.Module):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(x) = \frac{x}{1-x \mul e^{-(x+e^x)}}`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/CoLU.png

    Examples::

        >>> m = nn.CoLU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.CoLU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False):
        super(CoLU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:
            return x.div_(1 - x * torch.exp(-1 * (x + torch.exp(x))))
        else:
            return x / (1 - x * torch.exp(-1 * (x + torch.exp(x))))


class DELU(nn.Module):
    r"""
    Applies the DELU activation function:

    :math:`\text{DELU}(x) = \begin{cases} \text{SiLU}(x), x \leqslant 0 \\x(n-1), \text{otherwise} \end{cases}`

    Args:
        n: Scaling factor for the positive part of the input. Default: 1.0.
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Attributes:
        n: Scaling factor for the positive part of the input. Default: 1.0.

    Examples:
        >>> m = nn.DELU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.DELU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, n=1.0, inplace=False):
        super(DELU, self).__init__()
        self.n = torch.nn.Parameter(torch.tensor(n))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        return torch.where(x <= 0, 
            F.silu(x), 
            (self.n + 0.5) * x + torch.abs(torch.exp(-x) - 1))
        
    def _forward_inplace(self, x):
        x[x <= 0] = F.silu(x[x <= 0])
        x[x > 0] = (self.n + 0.5) * x[x > 0] + torch.abs(torch.exp(-x[x > 0]) - 1)
        return x


class GCU(nn.Module):
    r"""
    Applies the Growing Cosine Unit activation function:

    :math:`\text{GCU}(x) = x \cos (x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../images/activation_images/GCU.png

    Examples::

        >>> m = nn.GCU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.GCU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace=False):
        super(GCU, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        if self.inplace:            
            return x.mul_(torch.cos(x))
        else:
            return x * torch.cos(x)


class CosLU(nn.Module):
    r"""
    Applies the Cosine Linear Unit function:
    
    :math:`\text{CosLU}(x) = (x + \alpha \cdot \cos(\beta x)) \cdot \sigma(x)`

    Args:
        alpha: Scaling factor for the cosine term. Default is 1.0.
        beta: Frequency factor for the cosine term. Default is 1.0.
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Attributes:
        alpha: Scaling factor for the cosine term. Default is 1.0.
        beta: Frequency factor for the cosine term. Default is 1.0.
        
    .. image:: ../images/activation_images/CosLU.png

    Examples::

        >>> m = CosLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CosLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x) 
    """

    def __init__(self, alpha=1.0, beta=1.0, inplace=False):
        super(CosLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta  = nn.Parameter(torch.tensor(beta))
        self.inplace = inplace

    def forward(self, x) -> Tensor:
        return self._forward_inplace(x) if self.inplace else self._forward(x)

    def _forward(self, x):
        result = x + self.alpha * torch.cos(self.beta * x)
        result *= torch.sigmoid(x)
        return result
    
    def _forward_inplace(self, x):
        s_x = torch.sigmoid(x)
        x.add_(self.alpha * torch.cos(self.beta * x))
        x.mul_(s_x)
        del s_x
        return x


class ScaledSoftSign(torch.nn.Module):
    r"""
    Applies the ScaledSoftSign activation function:

    :math:`\text{ScaledSoftSign}(x) = \frac{\alpha \mul x}{\beta + |x|}`

    Args:
        alpha: The initial value of the alpha parameter.
        beta: The initial value of the beta parameter.
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    
    Attributes:
        alpha (nn.Parameter): The alpha scaling parameter.
        beta (nn.Parameter): The beta scaling parameter.
        
    .. image:: ../images/activation_images/ScaledSoftSign.png

    Examples:
        >>> m = ScaledSoftSign(alpha=0.5, beta=1.0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
        
        >>> m = ScaledSoftSign(inplace=True)
        >>> x = torch.randn(2, 3)
        >>> m(x)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super(ScaledSoftSign, self).__init__()
        
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = torch.nn.Parameter(torch.tensor(beta))

    def forward(self, x) -> Tensor:
        abs_x = x.abs()
        alpha_x = self.alpha * x
        denom = self.beta + abs_x
        result = alpha_x / denom
        return result


if __name__ == "__main__":
    pass