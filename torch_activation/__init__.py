from .composite import DELU, DReLUs
from .trig import CosLU, GCU, SinLU
from .lincomb import LinComb, NormLinComb
from .non_linear import CoLU, ScaledSoftSign
from .glus import ReGLU, GeGLU, SeGLU, SwiGLU
from .relus import ShiLU, CReLU, ReLUN, SquaredReLU, StarReLU


__all__ = ["ShiLU", "DELU", "CReLU", "GCU",
           "CosLU", "CoLU", "ReLUN",
           "SquaredReLU", "ScaledSoftSign",
           "ReGLU", "GeGLU", "SeGLU", "SwiGLU"
           "LinComb", "NormLinComb", "SinLU"
           "DReLUs", "StarReLU"]

__version__ = "0.1.2"