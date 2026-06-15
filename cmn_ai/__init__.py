"""Common machine learning utilities."""

from importlib import metadata

from .learner import Learner

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["Learner", "__version__"]
