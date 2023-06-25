import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from torch_activation.utils import plot_activation


class GCU(nn.Module):
    r"""
    Applies the Growing Cosine Unit activation function:

    :math:`\text{GCU}(x) = x \cos (x)`
    
    The GCU activation function try solves the XOR problem without feature-engineering, 
    and reportedly outperforms popular activations (e.g. Swish and Mish) in terms of computational efficiency, reduced training time, 
    and enabling smaller networks for classification tasks.

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Examples::

        >>> m = nn.GCU()
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = nn.GCU(inplace=True)
        >>> x = torch.randn(2)
        >>> m(x)
    """

    def __init__(self, inplace: bool = False):
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

    :math:`\text{CosLU}(x) = (x + a \cdot \cos(b x)) \cdot \sigma(x)`

    CosLU function is similar to `SinLU` but with a cosine function.
    
    Args:
        a (float, optional): Scaling factor for the cosine term. Default is 1.0.
        b (float, optional): Frequency factor for the cosine term. Default is 1.0.
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = CosLU(alpha=2.0, beta=1.0)
        >>> x = torch.randn(2)
        >>> output = m(x)

        >>> m = CosLU(inplace=True)
        >>> x = torch.randn(2, 3, 4)
        >>> m(x) 
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, 
                 inplace: bool = False):
        super(CosLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(a))
        self.beta = nn.Parameter(torch.tensor(b))
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


class SinLU(nn.Module):
    r"""
    Applies the Sinu-sigmoidal Linear Unit activation function:

    :math:`\text{SinLU}(x) = (x + \alpha \sin (\beta x)) \sigma (x)`
    
    SinLU is an activation function that combines the properties of trainable parameters, sinusoids, and ReLU-like functions. 
    By introducing the cumulative distribution function (CDF) of the logistic distribution and adding a sinusoidal term, SinLU smooths the output near x = 0 and modifies the loss landscape. 
    The parameters a and b control the amplitude and frequency of the sine function, allowing customization of the SinLU curve during training.

    Args:
        a (float, optional): Initial value for sine function magnitude. Default: 1.0.
        b (float, optional): Initial value for sine function period. Default: 1.0.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.SinLU(a=5.0, b=6.0)
        >>> x = torch.randn(2)
        >>> output = m(x)
    """
    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(SinLU, self).__init__()
        self.a = nn.Parameter(torch.Tensor([a]))
        self.b = nn.Parameter(torch.Tensor([b]))

    def forward(self, x) -> Tensor:
        return (x + self.a * torch.sin(self.b * x)) * torch.sigmoid(x)
