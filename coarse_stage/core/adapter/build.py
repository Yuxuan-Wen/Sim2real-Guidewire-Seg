from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .TRIBE import TRIBE_class
from .lame import LAME
from .tent import TENT
from .pl import PL
from .bn import BN
from .note import NOTE
from .ttac import TTAC
from .eata import EATA
from .cotta import CoTTA
from .petal import PETALFim


def build_adapter(name) -> type(BaseAdapter):
    if name == "rotta":
        return RoTTA
    elif name == "tribe":
        return TRIBE_class
    elif name == "lame":
        return LAME
    elif name == "tent":
        return TENT
    elif name == "pl":
        return PL
    elif name == "bn":
        return BN
    elif name == "note":
        return NOTE
    elif name == "ttac":
        return TTAC
    elif name == "eata":
        return EATA
    elif name == "cotta":
        return CoTTA
    elif name == "petal":
        return PETALFim
    else:
        raise NotImplementedError("Implement your own adapter")

