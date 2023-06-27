import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CoLU(nn.Module):
    r"""
    Applies the Collapsing Linear Unit activation function:

    :math:`\text{CoLU}(x) = \frac{x}{1-x \cdot e^{-(x + e^x)}}`

     See: https://doi.org/10.48550/arXiv.2112.12078

    Args:
        inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Here is a plot of the function and its derivative:
        
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


class ScaledSoftSign(torch.nn.Module):
    r"""
    Applies the ScaledSoftSign activation function:

    :math:`\text{ScaledSoftSign}(x) = \frac{\alpha \cdot x}{\beta + \|x\|}`
    
     See: https://doi.org/10.20944/preprints202301.0463.v1
   
    Args:
        alpha (float, optional): The initial value of the alpha parameter. Default: 1.0
        beta (float, optional): The initial value of the beta parameter. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Here is a plot of the function and its derivative:
        
    .. image:: ../images/activation_images/ScaledSoftSign.png

    Examples:
        >>> m = ScaledSoftSign(alpha=0.5, beta=1.0)
        >>> x = torch.randn(2, 3)
        >>> output = m(x)

        >>> m = ScaledSoftSign(inplace=True)
        >>> x = torch.randn(2, 3)
        >>> m(x)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        
        super(ScaledSoftSign, self).__init__()

        self.alpha = torch.nn.Parameter(Tensor([alpha]))
        self.beta = torch.nn.Parameter(Tensor([beta]))

    def forward(self, x) -> Tensor:
        abs_x = x.abs()
        alpha_x = self.alpha * x
        denom = self.beta + abs_x
        result = alpha_x / denom
        return result
    

class Phish(torch.nn.Module):
    r"""
    Applies the Phish activation function:

    :math:`\text{Phish}(x) = x \cdot \tanh (\text{GELU} (x))`

     See: `Phish: A Novel Hyper-Optimizable Activation Function`_.
     
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
        
    Here is a plot of the function and its derivative:
        
    .. image:: ../images/activation_images/Phish.png

    Examples:
        >>> m = Phish()
        >>> x = torch.randn(2, 3)
        >>> output = m(x)
        
    .. _`Phish: A Novel Hyper-Optimizable Activation Function`:
        https://www.semanticscholar.org/paper/Phish%3A-A-Novel-Hyper-Optimizable-Activation-Naveen/43eb5e22da6092d28f0e842fec53ec1a76e1ba6b
    """

    def __init__(self):
        super(Phish, self).__init__()

    def forward(self, x) -> Tensor:
        output = F.gelu(x)
        output = F.tanh(output)
        output = x * output
        return output


if __name__ == "__main__":
    pass
