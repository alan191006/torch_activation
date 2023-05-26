from .composite import DELU, DReLUs
from .trig import CosLU, GCU, SinLU
from .lincomb import LinComb, NormLinComb
from .non_linear import CoLU, ScaledSoftSign
from .glu_family import ReGLU, GeGLU, SeGLU, SwiGLU
from .relu_family import ShiLU, CReLU, ReLUN, SquaredReLU


__all__ = ["ShiLU", "DELU", "CReLU", "GCU",
           "CosLU", "CoLU", "ReLUN",
           "SquaredReLU", "ScaledSoftSign",
           "ReGLU", "GeGLU", "SeGLU", "SwiGLU"
           "LinComb", "NormLinComb", "SinLU"
           "DReLUs"]

__version__ = "0.1.0"