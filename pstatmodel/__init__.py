from . import utils
from ._version import get_versions
from .variable import ModelVariables, PredictorVariable

__version__ = get_versions()["version"]
del get_versions

__all__ = ["utils", "ModelVariables", "PredictorVariable"]
