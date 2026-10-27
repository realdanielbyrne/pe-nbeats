from . import blocks
from .blocks import *


def __getattr__(name):
    return getattr(blocks, name)
