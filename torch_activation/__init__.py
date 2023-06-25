from .piece_wise import DELU
from .trig import CosLU, GCU, SinLU
from .lincomb import LinComb, NormLinComb
from .curves import CoLU, ScaledSoftSign, Phish
from .glus import ReGLU, GeGLU, SeGLU, SwiGLU
from .relus import ShiLU, CReLU, ReLUN, SquaredReLU, StarReLU


__all__ = ["ShiLU", "DELU", "CReLU", "GCU",
           "CosLU", "CoLU", "ReLUN",
           "SquaredReLU", "ScaledSoftSign",
           "ReGLU", "GeGLU", "SeGLU", "SwiGLU"
           "LinComb", "NormLinComb", "SinLU"
           "DReLUs", "StarReLU"]

__version__ = "0.1.3"