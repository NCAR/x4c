''' x4c: Xarray for CESM

An xarray extension for postprocessing, analysis and visualization of CESM output.
Registers a `.x` accessor on `xarray.Dataset` and `xarray.DataArray`.

.. warning::

    **Importing x4c sets ``xarray.set_options(keep_attrs=True)`` process-wide.**

    This is required, not incidental: x4c carries grid metadata (``gw``, ``lat``,
    ``lon``, ``dz``) in ``.attrs``, and xarray's default of dropping attrs through
    arithmetic would silently break ``.x.gm`` and friends after an operation as
    ordinary as ``da - 273.15``.

    The consequence is that any other library in the same process also sees
    ``keep_attrs=True``. If that matters, capture and restore it yourself::

        import xarray as xr
        prev = xr.get_options()['keep_attrs']
        import x4c
        ...
        xr.set_options(keep_attrs=prev)

Downloaded regrid weight files and the tutorial sample case are cached under
``$X4C_CACHE_DIR``, else ``$XDG_CACHE_HOME/x4c``, else ``~/.cache/x4c`` (see
:func:`x4c.utils.cache_dir`). :func:`x4c.fetch_sample_data` downloads the sample CESM
case the tutorial notebooks use.
'''
# get the version
from importlib.metadata import version
__version__ = version('x4c')

from .core import load_dataset, open_dataset, open_mfdataset, XDataset, XDataArray
from .case import History, Timeseries, Logs
from .spell import Spell

from . import utils
from .utils import fetch_sample_data
from .visual import (
    set_style,
    showfig,
    closefig,
    savefig,
    subplots,
    # `add_annotation` is what `subplots(annotation=True)` calls; exported so it can
    # also be applied to axes built directly with matplotlib. `infer_cmap` is the
    # colormap-from-long_name rule `.x.plot()` uses when no cmap is given.
    add_annotation,
    infer_cmap,
)
