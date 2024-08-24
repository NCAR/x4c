# get the version
from importlib.metadata import version
__version__ = version('x4c')

from .core import load_dataset, open_dataset, open_mfdataset, XDataset, XDataArray
from .case import Timeseries

from . import utils