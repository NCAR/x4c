import os
import glob
import re
import itertools
import hashlib
import tarfile
import numpy as np
import xarray as xr
import colorama as ca
import requests
from tqdm import tqdm
import datetime
import collections.abc
import cartopy.util
import shutil
import subprocess
import warnings

def p_header(text):
    print(ca.Fore.CYAN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_hint(text):
    print(ca.Fore.LIGHTBLACK_EX + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_success(text):
    print(ca.Fore.GREEN + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_fail(text):
    print(ca.Fore.RED + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)

def p_warning(text):
    print(ca.Fore.YELLOW + ca.Style.BRIGHT + text + ca.Style.RESET_ALL)


def import_xesmf():
    ''' Import xesmf on demand

    Lazy because xesmf (and its ESMF/esmpy backend) is conda-only and awkward to
    build from PyPI, so requiring it at import time made the whole package
    uninstallable from PyPI. Only the regridding paths need it.
    '''
    try:
        import xesmf as xe
    except ImportError as e:
        raise ImportError(
            'xesmf is required for regridding. Install it with '
            '`conda install -c conda-forge xesmf esmpy` (or `pip install "x4c[regrid]"` '
            'if you have a working ESMF).'
        ) from e
    return xe


def import_geocat_comp():
    ''' Import geocat.comp on demand (see :func:`import_xesmf`) '''
    try:
        import geocat.comp as gc
    except ImportError as e:
        raise ImportError(
            'geocat-comp is required for hybrid-to-pressure-level interpolation. '
            'Install it with `conda install -c conda-forge geocat-comp`.'
        ) from e
    return gc

def regrid_cam_se(ds, weight_file):
    """
    Regrid CAM-SE output using an existing ESMF weights file.

    Parameters
    ----------
    ds: xarray.Dataset
        Input dataset to be regridded. Must have the `ncol` dimension.
    weight_file: str or Path
        Path to existing ESMF weights file

    Returns
    -------
    regridded
        xarray.Dataset after regridding.

    Reference
    ---------
    ESDS post: https://ncar.github.io/esds/posts/2023/cam-se-analysis/#define-regridding-function-that-constructs-an-xesmf-regridder 
    
    """
    dataset = ds.copy()
    assert isinstance(dataset, xr.Dataset)
    weights = xr.open_dataset(weight_file)

    # input variable shape
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1: in_shape = [1, in_shape.item()]

    # output variable shapew
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    # print(f"Regridding from {in_shape} to {out_shape}")

    # Insert dummy dimension
    vars_with_ncol = [name for name in dataset.variables if 'ncol' in dataset[name].dims]
    updated = dataset[vars_with_ncol].transpose(..., 'ncol').expand_dims('dummy', axis=-2)

    # construct a regridder
    # use empty variables to tell xesmf the right shape
    # https://github.com/pangeo-data/xESMF/issues/202
    dummy_in = xr.Dataset(
        {
            'lat': ('lat', np.empty((in_shape[0],))),
            'lon': ('lon', np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            'lat': ('lat', weights.yc_b.data.reshape(out_shape)[:, 0]),
            'lon': ('lon', weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    xe = import_xesmf()
    regridder = xe.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method='bilinear',
        reuse_weights=True,
        periodic=True,
    )

    # Actually regrid, after renaming
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        regridded = regridder(updated.rename({'dummy': 'lat', 'ncol': 'lon'}), keep_attrs=True)
    # merge back any variables that didn't have the ncol dimension
    # And so were not regridded
    ds_out = xr.merge([dataset.drop_vars(regridded.variables, errors='ignore'), regridded])

    return ds_out

def annualize(ds, months=None, days_weighted=False):
    months = list(range(1, 13)) if months is None else np.abs(months)
    sds = ds.sel(time=ds['time.month'].isin(months))
    anchor = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    idx = months[-1]-1

    anchor_str = f'YE-{anchor[idx]}'

    if days_weighted:
        # Day-length weighted mean, normalized over each resample bin rather than
        # over the calendar year. Calendar-year normalization is wrong for any
        # wraparound season (e.g. DJF, where a bin spans two calendar years) and
        # for incomplete leading/trailing bins, whose weights then sum to well
        # under 1 and pull the value toward zero.
        # Dividing by the weights that actually contributed also keeps bins with
        # missing months (or NaNs in the data) unbiased -- matching the `.mean()`
        # behavior of the unweighted branch -- and leaves NaN, not 0, wherever a
        # bin has no valid input at all (`sum` counts NaNs as zeros, so the
        # denominator has to carry the masking).
        days_in_month = sds.time.dt.days_in_month
        num = (sds * days_in_month).resample(time=anchor_str).sum()
        den = (sds.notnull() * days_in_month).resample(time=anchor_str).sum()
        ds_ann = num / den.where(den > 0)
    else:
        ds_ann = sds.resample(time=anchor_str).mean()  # unweighted version

    # a Dataset has no `.name`; only propagate it for the DataArray case
    if isinstance(sds, xr.DataArray):
        ds_ann.name = sds.name

    return ds_ann

def monthly2annual(ds):
    month_length = ds.time.dt.days_in_month
    wgts_mon = month_length.groupby('time.year') / month_length.groupby('time.year').mean()
    ds_ann = (ds * wgts_mon).groupby('time.year').mean('time')
    return ds_ann.rename({'year':'time'})

def monthly2season(ds):
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby('time.season') / month_length.groupby('time.season').mean()
    ds_season = (ds * wgts).groupby('time.season').mean('time')
    return ds_season

def geo_mean(da, lat_min=-90, lat_max=90, lon_min=0, lon_max=360, lat_name='lat', lon_name='lon', **kws):
    ''' Calculate the geographical mean value of the climate field.

    Args:
        lat_min (float): the lower bound of latitude for the calculation.
        lat_max (float): the upper bound of latitude for the calculation.
        lon_min (float): the lower bound of longitude for the calculation.
        lon_max (float): the upper bound of longitude for the calculation.
        gw (optional): weight of each gridcell
        lat (optional): lat of each gridcell
        lon (optional): lon of each gridcell
    '''
    if 'gw' not in da.attrs and 'gw' not in kws:
        # calculation
        mask_lat = (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max)
        mask_lon = (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max)
        dac = da.sel({
                lat_name: da[lat_name][mask_lat],
                lon_name: da[lon_name][mask_lon],
            })
        wgts = np.cos(np.deg2rad(dac[lat_name]))
        m = dac.weighted(wgts).mean((lon_name, lat_name))
    elif 'gw' in da.attrs and 'lat' in da.attrs and 'lon' in da.attrs:
        gw = da.attrs['gw']
        lat = da.attrs['lat']
        lon = da.attrs['lon']
        m = da.where((lat>lat_min) & (lat<lat_max) & (lon>lon_min) & (lon<lon_max)).weighted(gw).mean(list(gw.dims))
    elif 'gw' in kws and 'lat' in kws and 'lon' in kws:
        gw = kws['gw']
        lat = kws['lat']
        lon = kws['lon']
        m = da.where((lat>lat_min) & (lat<lat_max) & (lon>lon_min) & (lon<lon_max)).weighted(gw).mean(list(gw.dims))
    else:
        # e.g. `gw` present but `lat`/`lon` missing -- the state a regrid used to
        # leave behind. Without this the function fell through to `return m` with
        # `m` unbound and raised `UnboundLocalError`.
        raise ValueError(
            'Cannot compute a geographical mean: need either no `gw` at all (to use '
            'the cos(lat) fallback on the lat/lon coordinates), or `gw` together with '
            '`lat` and `lon` -- in `da.attrs` or as keyword arguments. '
            f'Found attrs: {sorted(k for k in da.attrs if k in GRID_ATTRS)}; '
            f'kwargs: {sorted(k for k in kws if k in GRID_ATTRS)}.'
        )

    return m

#: the `xarray.DataArray`-valued attrs that x4c parks in `.attrs` to drive the
#: accessors; netCDF attributes must be scalars or strings, so these have to be
#: stripped before writing (see :func:`drop_grid_attrs`)
GRID_ATTRS = ('gw', 'lat', 'lon', 'dz')

def drop_grid_attrs(obj):
    ''' Return a copy of `obj` with the x4c grid attrs stripped, ready to serialize

    `gw`/`lat`/`lon`/`dz` are `xarray.DataArray`s carried in `.attrs` so that the
    accessors (`.x.gm`, `.x.nhm`, `.x.zavg`, ...) can find the grid metadata. netCDF
    attributes may only be scalars or strings, so they must come off before writing.

    Two things matter here:

    - The strip happens on a **copy**. Doing it in place would leave the object the
      caller still holds without its weights, so every subsequent `.x.gm`/`.x.nhm`/
      `.x.zavg` on it would raise `KeyError` far away from the write that caused it.
    - For a `Dataset`, the attrs are cleaned off the **variables as well as the
      dataset**. A variable that came through `XDataset.__getitem__` (or
      `XDataArray.ds`) carries its own copy of them, and those alone are enough to
      make `to_netcdf` fail.

    Args:
        obj (`xarray.Dataset` or `xarray.DataArray`): the object to clean

    Returns:
        the same type as `obj`, with :data:`GRID_ATTRS` removed at every level
    '''
    out = obj.copy()
    out.attrs = {k: v for k, v in obj.attrs.items() if k not in GRID_ATTRS}

    if isinstance(out, xr.Dataset):
        for vn in out.variables:
            var = out.variables[vn]
            var.attrs = {k: v for k, v in var.attrs.items() if k not in GRID_ATTRS}

    return out

def copy_grid_attrs(dst, src):
    ''' Copy the x4c grid attrs from `src` onto `dst`, in place

    Needed after arithmetic between two DataArrays. Even with
    ``xr.set_options(keep_attrs=True)``, a binary operation keeps only the attrs
    that are *identical* in both operands: plain strings like ``comp`` and ``units``
    survive, but ``gw``/``lat``/``lon``/``dz`` are `DataArray`s, compare as
    conflicting, and get dropped. A derived variable built as ``a + b`` therefore
    loses its area weight, and the next ``.x.gm`` fails with ``KeyError: 'gw'``.

    Args:
        dst (`xarray.DataArray`): the result of the arithmetic
        src (`xarray.DataArray` or `xarray.Dataset`): an operand to take the grid
            metadata from

    Returns:
        `xarray.DataArray`: `dst`, for chaining
    '''
    for k in GRID_ATTRS:
        if k in src.attrs and k not in dst.attrs:
            dst.attrs[k] = src.attrs[k]
    return dst


def coslat_weight(ds, lat_name='lat', lon_name='lon'):
    ''' The cos(lat) area weight for a regular lat/lon grid

    Args:
        ds (`xarray.Dataset`): a dataset with a regular lat (and optionally lon) coordinate
        lat_name (str): the name of the latitude coordinate
        lon_name (str): the name of the longitude coordinate

    Returns:
        `xarray.DataArray`: cos(lat) broadcast over (lat, lon) when both are
        present, otherwise the 1-D cos(lat).

    Used both when attaching `gw` at load time (:func:`update_ds`) and after
    regridding onto a regular grid (:func:`x4c.core.XDataset.regrid`), so that
    the two agree on the weighting convention.
    '''
    coslat = np.cos(np.deg2rad(ds[lat_name]))
    if lon_name in ds.variables:
        # 2-D cos(lat) area weight spanning (lat, lon)
        return coslat.broadcast_like(ds[lat_name] * ds[lon_name])
    else:
        return coslat

def update_attrs(da, da_src):
    da.attrs = dict(da_src.attrs)
    if 'comp' in da.attrs and 'time' in da.coords:
        da.time.attrs['long_name'] = 'Model Year'

    return da

def update_ds(ds, path, vn=None, comp=None, hstr=None, grid=None, shift_time=False,
              gw_name=None, lat_name=None, lon_name=None):
    if shift_time:
        freq = ds['time'].to_index().freq
        m = re.match(r'(?:(\d+))?([A-Za-z]+)', freq)
        if not m: raise ValueError(f'Cannot parse time frequency: {freq}')
        n = int(m.group(1)) if m.group(1) else 1
        unit = m.group(2)

        times = ds['time'].values
        if 'M' in unit.upper():
            ds['time'] = [(t - datetime.timedelta(days=15)).replace(day=1) for t in times]
        elif 'D' in unit.upper():
            ds['time'] = times - datetime.timedelta(days=n)
        elif 'H' in unit.upper():
            ds['time'] = times - datetime.timedelta(hours=n)
        else:
            raise ValueError(f'Unsupported time unit for shifting: {unit}')

    if type(path) in (list, tuple):
        ds.attrs['path'] = [os.path.abspath(p) for p in path]
    else:
        ds.attrs['path'] = os.path.abspath(path)

    if vn is not None: ds.attrs['vn'] = vn
    if comp is not None: ds.attrs['comp'] = comp
    if hstr is not None: ds.attrs['hstr'] = hstr
    if grid is not None: ds.attrs['grid'] = grid

    if 'comp' in ds.attrs:
        # `rof` (RTM/MOSART) is on a regular lat/lon grid, like `atm`/`lnd`.
        # Unlisted components fall back to the regular lat/lon convention rather than
        # raising: the lookups below are all guarded by an `in ds` check, so a name that
        # turns out to be absent simply degrades to the cos(lat) weight.
        gw_dict = {
            'atm': 'area',
            'ocn': 'TAREA',
            'ice': 'tarea',
            'lnd': 'area',
            'rof': 'area',
        }

        lon_dict = {
            'atm': 'lon',
            'ocn': 'TLONG',
            'ice': 'TLON',
            'lnd': 'lon',
            'rof': 'lon',
        }

        lat_dict = {
            'atm': 'lat',
            'ocn': 'TLAT',
            'ice': 'TLAT',
            'lnd': 'lat',
            'rof': 'lat',
        }

        comp = ds.attrs['comp']
        gw_name = gw_dict.get(comp, 'area') if gw_name is None else gw_name
        lat_name = lat_dict.get(comp, 'lat') if lat_name is None else lat_name
        lon_name = lon_dict.get(comp, 'lon') if lon_name is None else lon_name

    if gw_name is not None and gw_name in ds:
        ds.attrs['gw'] = ds[gw_name]
    elif 'gw' in ds.variables:
        ds.attrs['gw'] = ds['gw']
    elif 'lat' in ds.variables:
        ds.attrs['gw'] = coslat_weight(ds)

    if lat_name is not None and lat_name in ds: ds.attrs['lat'] = ds[lat_name]
    if lon_name is not None and lon_name in ds: ds.attrs['lon'] = ds[lon_name]

    return ds

def infer_months_char(months):
    char_list = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    out_str = ''
    for i in months:
        out_str += char_list[np.abs(i)-1]
    return out_str


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def add_cyclic_point(da):
    data_wrap, lon_wrap = cartopy.util.add_cyclic_point(da.values, coord=da.lon)
    da_new_coords = {k: v.copy(deep=True) for k, v in da.coords.items()}
    da_new_coords['lon'] = lon_wrap
    da_wrap = xr.DataArray(data_wrap, dims=da.dims, coords=da_new_coords)
    da_wrap.attrs = da.attrs.copy()
    return da_wrap

def ds_lon360(ds, lon_name='lon'):
    ''' Convert the longitude of an xarray.Dataset from (-180, 180) to (0, 360)
    '''
    ds_out = ds.assign_coords({lon_name: ((ds[lon_name] + 360) % 360)})
    ds_out = ds_out.sortby(lon_name)
    return ds_out

def ann_modifier(da, ann_method, long_name=None):
    if long_name is None:
        if 'long_name' in da.attrs:
            long_name = da.attrs['long_name']
        else:
            long_name = da.name

    if ann_method == 'ann':
        da_out = da.x.annualize()
        da_out.attrs['long_name'] = f'{long_name} (Annual)'
    elif ann_method == 'climo':
        da_out = da.x.climo
        da_out.attrs['long_name'] = f'{long_name} (Climatology)'
    else:
        months = [int(s) for s in ann_method.split(',')]
        months_char = infer_months_char(months)
        da_out = da.x.annualize(months=months)
        da_out.attrs['long_name'] = f'{long_name} ({months_char})'

    return da_out

def convert_units(da, units=None):
    if units is not None:
        if 'units' in da.attrs:
            if da.attrs['units'] == 'K' and units == 'degC':
                da -= 273.15
                da.attrs['units'] = '°C'
            elif da.attrs['units'] == 'degC' and units == 'K':
                da += 273.15
                da.attrs['units'] = 'K'
            elif da.attrs['units'] == 'degC' and units == 'degC' or units is None:
                da.attrs['units'] = '°C'
        else:
            p_warning("The input `xarray.DataArray` doesn't have units.")

    return da

def expand_braces(pattern):
    '''
    Expands a string with brace-enclosed options like:
    'atm/*/*.cam.{h0a,h0i}.*.nc' --> [
        'atm/*/*.cam.h0a.*.nc',
        'atm/*/*.cam.h0i.*.nc'
    ]
    Supports multiple sets of {}.
    '''
    # Find all brace-enclosed segments
    matches = list(re.finditer(r'\{([^}]+)\}', pattern))
    if not matches:
        return [pattern]

    # Extract options for each set of braces
    segments = []
    last_end = 0
    static_parts = []

    for match in matches:
        static_parts.append(pattern[last_end:match.start()])
        segments.append(match.group(1).split(','))
        last_end = match.end()

    static_parts.append(pattern[last_end:])  # tail

    # Generate combinations
    expanded = []
    for combo in itertools.product(*segments):
        s = ''.join([sp + c for sp, c in zip(static_parts, combo)] + [static_parts[-1]])
        expanded.append(s)

    return expanded


def find_paths(root_dir, path_pattern='comp/proc/tseries/*/casename.hstr.vn.timespan.nc', delimiters=['/', '.'],
               avoid_list=None, verbose=False, **kws):
    s = path_pattern
    for d in delimiters:
        s = ' '.join(s.split(d))
    path_elements = s.split()

    for e in path_elements:
        if e in kws:
            value = kws[e]
            if isinstance(value, list):
                pattern_str = '{' + ','.join(value) + '}'
                path_pattern = path_pattern.replace(e, pattern_str)
            else:
                path_pattern = path_pattern.replace(e, value)
        elif e in ['proc', 'tseries', 'nc']:
            pass
        elif e in ['timespan', 'date']:
            path_pattern = path_pattern.replace(e, '[0-9]*[0-9]')
        else:
            path_pattern = path_pattern.replace(e, '*')

    path_patterns = expand_braces(path_pattern)
    if verbose: p_header(f'path_patterns: {path_patterns}')
    paths = []
    for path in path_patterns:
        paths_tmp = glob.glob(os.path.join(root_dir, path))
        paths.extend(paths_tmp)

    # sort based on timespan
    paths = sorted(paths, key=lambda x: x.split('.')[-2])
    if avoid_list is not None:
        paths_new = []
        for path in paths:
            add_path = True
            # match the basename, not the full path: the tokens describe filename
            # segments (`.once.`, `.h0.`), so matching the whole path let an avoid
            # token in any parent directory silently discard every file
            basename = os.path.basename(path)
            for avoid_str in avoid_list:
                if avoid_str in basename:
                    add_path = False
                    break
            if add_path: paths_new.append(path)
        paths = paths_new
    return paths

def get_hstr(paths, casename):
    hstr_set = set()

    # Pattern to extract what's after mdl.
    pattern = re.compile(rf'{re.escape(casename)}\.((?:[^0-9][^.]*\.?)+)')

    # Pattern to remove trailing date strings like .0001-01 or .0001-01-0001-12
    date_like_pattern = re.compile(r'(\.?\d{4}-\d{2}(?:-\d{4}-\d{2})?)$')

    for path in paths:
        filename = os.path.basename(path)
        match = pattern.search(filename)
        if match:
            hstr = match.group(1)
            # Remove date-like suffix
            hstr = date_like_pattern.sub('', hstr)
            hstr = hstr.rstrip('.')
            if 'h' in hstr:  # Only keep if 'h' is present
                hstr_set.add(hstr)

    return sorted(hstr_set)

def add_months(dt: datetime.datetime, months: int) -> datetime.datetime:
    """Add months to a datetime without relativedelta."""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31,
                       29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)

def minus_months(dt: datetime.datetime, months: int) -> datetime.datetime:
    """Minus months to a datetime without relativedelta."""
    month = dt.month - 1 - months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31,
                       29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                       31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)

def parse_timespan(timespan: tuple[str, str]):
    start, end = timespan
    date_elements, nparts = {}, {}
    date = {}
    date['year'], date['month'], date['day'], date['hour'] = {}, {}, {}, {}
    date_elements['start']= start.split('-')
    date_elements['end']= end.split('-')

    for tag in ['start', 'end']:
        nparts[tag] = len(date_elements[tag])
        if nparts[tag] == 1:
            date['year'][tag] = int(date_elements[tag][0])
            date['month'][tag] = 1
            date['day'][tag] = 1
            date['hour'][tag] = 0
            timespan_precision = 'year'
        elif nparts[tag] == 2:
            date['year'][tag] = int(date_elements[tag][0])
            date['month'][tag] = int(date_elements[tag][1])
            date['day'][tag] = 1
            date['hour'][tag] = 0
            timespan_precision = 'month'
        elif nparts[tag] == 3:
            date['year'][tag] = int(date_elements[tag][0])
            date['month'][tag] = int(date_elements[tag][1])
            date['day'][tag] = int(date_elements[tag][2])
            date['hour'][tag] = 0
            timespan_precision = 'day'
        elif nparts[tag] == 4:
            date['year'][tag] = int(date_elements[tag][0])
            date['month'][tag] = int(date_elements[tag][1])
            date['day'][tag] = int(date_elements[tag][2])
            date['hour'][tag] = int(date_elements[tag][3])
            timespan_precision = 'hour'
        else:
            raise ValueError(f'Invalid timespan element format. Expected format: YYYY-MM-DD-HH.')

    start_dt = datetime.datetime(date['year']['start'], date['month']['start'], date['day']['start'], date['hour']['start'])
    end_dt = datetime.datetime(date['year']['end'], date['month']['end'], date['day']['end'], date['hour']['end'])
    return start_dt, end_dt, timespan_precision

def parse_timestamps(timespan: tuple[str, str], timestep:int, timestep_unit:str='year'):
    start_dt, end_dt, timespan_precision = parse_timespan(timespan)

    timestamp_list = []
    current = start_dt
    while current <= end_dt:
        if timestep_unit == 'year':
            nxt = add_months(current, timestep * 12)
            current_end = minus_months(nxt, 1)
        elif timestep_unit == 'month':
            nxt = add_months(current, timestep)
            current_end = minus_months(nxt, 1)
        elif timestep_unit == 'day':
            nxt = current + datetime.timedelta(days=timestep)
            current_end = nxt - datetime.timedelta(days=1)
        elif timestep_unit == 'hour':
            nxt = current + datetime.timedelta(hours=timestep)
            current_end = nxt - datetime.timedelta(hours=1)
        else:
            raise ValueError('Unsupported timestep_unit. Choose from year, month, day, hour.')

        if timespan_precision == 'year':
            current_str = f'{current.year:04d}'
            current_end_str = f'{current_end.year:04d}'
        elif timespan_precision == 'month':
            current_str = f'{current.year:04d}-{current.month:02d}'
            current_end_str = f'{current_end.year:04d}-{current_end.month:02d}'
        elif timespan_precision == 'day':
            current_str = f'{current.year:04d}-{current.month:02d}-{current.day:02d}'
            current_end_str = f'{current_end.year:04d}-{current_end.month:02d}-{current_end.day:02d}'
        elif timespan_precision == 'hour':
            current_str = f'{current.year:04d}-{current.month:02d}-{current.day:02d}-{current.hour*3600:05d}'
            current_end_str = f'{current_end.year:04d}-{current_end.month:02d}-{current_end.day:02d}-{current_end.hour*3600:05d}'

        timestamp_list.append((current_str, current_end_str))
        current = nxt

    return timestamp_list

def cesm_str2datetime(s: str) -> datetime.datetime:
    """Convert CESM timestamp 'YYYY-MM-DD-SSSSS' or 'YYYYMMDDSSSSSS' to a datetime."""

    if "-" in s:  # dash-separated formats
        nparts = len(s.split('-'))
        if nparts == 4:
            year, month, day, sec_str = s.split('-')
            seconds = int(sec_str)
            base = datetime.datetime(int(year), int(month), int(day))
            res = base + datetime.timedelta(seconds=seconds)
        elif nparts == 3:
            year, month, day = s.split('-')
            res = datetime.datetime(int(year), int(month), int(day))
        elif nparts == 2:
            year, month = s.split('-')
            res = datetime.datetime(int(year), int(month), 1)
        elif nparts == 1:
            # unreachable while guarded by `'-' in s`, but it used to do
            # `int(s.split('-'))` -- int() of a list -- and would raise TypeError the
            # moment the guard changed
            res = datetime.datetime(int(s), 1, 1)
        else:
            raise ValueError(
                f'Cannot parse CESM timestamp {s!r}: expected 1-4 dash-separated '
                'fields (YYYY[-MM[-DD[-SSSSS]]]).'
            )
    else:  # compact format, e.g. "YYYYMMDDSSSSSS"
        year   = int(s[0:4])
        month  = int(s[4:6])
        day    = int(s[6:8])
        seconds = int(s[8:]) if len(s) > 8 else 0
        base = datetime.datetime(year, month, day)
        res = base + datetime.timedelta(seconds=seconds)

    return res

def add_dash_to_timestamp(timestamp:str):
    if len(timestamp) == 4:
        # year
        res = timestamp
    elif len(timestamp) == 6:
        # month
        res = f'{timestamp[0:4]}-{timestamp[4:6]}'
    elif len(timestamp) == 8:
        # day
        res = f'{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}'
    elif len(timestamp) == 14:
        res = f'{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}-{timestamp[8:14]}'
    else:
        raise ValueError(f'Invalid timestamp format: {timestamp}. Supported formats: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDSSSSSS')
    return res

def int_to_timestamp(t: int) -> str:
    t_str = str(t)
    if len(t_str) <= 4:
        # year
        t_str = t_str.zfill(4)
    elif len(t_str) <= 6:
        # month
        t_str = t_str.zfill(6)
    elif len(t_str) <= 8:
        # day
        t_str = t_str.zfill(8)
    elif len(t_str) <= 14:
        # second
        t_str = t_str.zfill(14)
    else:
        raise ValueError('Invalid integer timestamp format. Supported formats: YYYY, YYYYMM, YYYYMMDD, YYYYMMDDSSSSSS')

    return add_dash_to_timestamp(t_str)

def timespan_int2str(timespan: tuple[int, int]) -> tuple[str, str]:
    start, end = timespan
    start_str = int_to_timestamp(start)
    end_str = int_to_timestamp(end)
    return (start_str, end_str)

def normalize_timespan(timespan):
    ''' Coerce a timespan to the ``('YYYY-MM', 'YYYY-MM')`` string form

    Accepts ints (``(1, 20)``), strings (``('0001-01', '0020-12')``), or a mix.
    Element-wise on purpose: the all-or-nothing checks this replaces either skipped
    conversion for a mixed tuple, or -- if the check were simply loosened -- fed an
    already-formatted string back through `int_to_timestamp`, which zero-pads it into
    nonsense (``'0001-01'`` -> ``'00001-01'`` -> ``'0000-1-01'``).

    Args:
        timespan (tuple): (start, end), each an int or a timestamp string

    Returns:
        tuple[str, str]
    '''
    if timespan is None: return None
    start, end = timespan[0], timespan[-1]
    return (
        start if isinstance(start, str) else int_to_timestamp(start),
        end if isinstance(end, str) else int_to_timestamp(end),
    )

def datetime_truncate(dt: datetime.datetime, precision: str = 'day') -> datetime.datetime:
    if precision == 'year':
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif precision == 'month':
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif precision == 'day':
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif precision == 'hour':
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from 'year', 'month', 'day', 'hour'.")

#: repository holding x4c's downloadable data: the regrid weights live in its tree,
#: the sample datasets are attached to its Releases
DATA_REPO = 'fzhu2e/x4c-data'

#: base URL for the on-demand regrid weight / SCRIP files
WGTS_URL = f'https://github.com/{DATA_REPO}/raw/main/regrid_wgts'


def cache_dir():
    ''' The directory x4c downloads regrid weight files into

    Resolution order:

    1. ``$X4C_CACHE_DIR``, if set
    2. ``$XDG_CACHE_HOME/x4c``, if set
    3. ``~/.cache/x4c``

    Deliberately *not* the installed package directory: that fails outright on a
    read-only or shared ``site-packages``, and in an editable install it drops
    multi-MB binaries into the source tree.
    '''
    base = os.environ.get('X4C_CACHE_DIR')
    if not base:
        xdg = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
        base = os.path.join(xdg, 'x4c')
    return base


def fetch_wgt_file(fname, verbose=True):
    ''' Return a local path to the weight file `fname`, downloading it if needed

    Looks in the user cache first, then alongside the package (so files shipped in
    the wheel, or already sitting in a source checkout, are still found), and only
    then downloads into the cache.

    Args:
        fname (str): the file's basename, e.g. ``map_ne30pg3_TO_1x1d_aave.nc.gz``

    Returns:
        str: an existing local path
    '''
    cached = os.path.join(cache_dir(), fname)
    if os.path.exists(cached):
        return cached

    # files bundled with the package, or left in a source checkout by older versions
    packaged = os.path.join(os.path.dirname(__file__), 'regrid_wgts', fname)
    if os.path.exists(packaged):
        return packaged

    url = f'{WGTS_URL}/{fname}'
    if verbose: p_header(f'Downloading the weight file from: {url}')
    download(url, cached)
    _require_gzip(cached, url)
    if verbose: p_success(f'>>> cached at: {cached}')
    return cached


def _require_gzip(path, url):
    ''' Delete and complain if `path` is not gzip, as everything x4c downloads is

    `download` raises on an HTTP error status, which is not enough: GitHub answers a
    *missing* path under ``/raw/`` with **200 and an HTML page**, so a wrong or
    not-yet-pushed URL would otherwise leave that page in the cache under a
    ``.nc.gz`` name, and the failure would surface much later as an unintelligible
    error from inside `xr.open_dataset`.
    '''
    with open(path, 'rb') as f:
        magic = f.read(2)
    if magic != b'\x1f\x8b':
        head = open(path, 'rb').read(200)
        os.remove(path)
        raise RuntimeError(
            f'{url} did not return a gzip file (it starts with {head[:40]!r}). '
            'A GitHub HTML page usually means the URL is wrong or the file has not '
            'been pushed yet. The download was removed rather than cached.'
        )


#: the sample CESM data the tutorial notebooks run against, keyed by the short case
#: name passed to :func:`fetch_sample_data`. Each is published as a Release asset on
#: :data:`DATA_REPO` rather than committed -- half a gigabyte of netCDF has no business
#: in a git history, and a Release asset does not weigh on `git clone`. One entry per
#: case, versioned by its own tag, so a second one can be added without touching the
#: fetching code:
#:
#: - ``dataset``: its directory in the data repository, reused for the cache
#: - ``case``: the case directory the archive extracts to
#: - ``tag``: the Release tag carrying the asset
#: - ``archive``: the asset filename
#: - ``sha256``: expected checksum of the asset, or `None` to skip verification
SAMPLE_DATA = {
    'cesm1': {
        'dataset': 'cesm1_sample_data',
        'case': 'b.e13.B1850C5.ne16_g16.icesm131_d18O_fixer.Miocene.3xCO2.005',
        'tag': 'cesm1_sample_data-v1',
        'archive': 'cesm1_sample_data-v1.tar.gz',
        'sha256': '93a087a7bce6ac121d1fd79408fbcbfe13bdba551b904b11e134db0fa76a455f',
    },
}

#: the case :func:`fetch_sample_data` returns when none is given
DEFAULT_SAMPLE_CASE = 'cesm1'


def sample_data_info(case=DEFAULT_SAMPLE_CASE):
    ''' The :data:`SAMPLE_DATA` entry for `case`, with a listing on a bad key '''
    try:
        return SAMPLE_DATA[case]
    except KeyError:
        raise KeyError(
            f'Unknown sample case {case!r}. Available: '
            f'{", ".join(sorted(SAMPLE_DATA))}.'
        ) from None


def sample_data_url(case=DEFAULT_SAMPLE_CASE):
    ''' The Release-asset download URL of the sample data for `case` '''
    info = sample_data_info(case)
    return (
        f'https://github.com/{DATA_REPO}/releases/download/'
        f'{info["tag"]}/{info["archive"]}'
    )


def fetch_sample_data(case=DEFAULT_SAMPLE_CASE, url=None, sha256=None,
                      verbose=True, force=False):
    ''' Return a local path to a tutorial sample case, downloading it if needed

    The tutorial notebooks run against a reduced copy of a real CESM case. It lives in
    the cache directory (see :func:`cache_dir`), not in the repository.

    Resolution order:

    1. ``$X4C_SAMPLE_DIR``, if set and it contains the case -- use this to point at a
       copy you already have, e.g. on a shared filesystem.
    2. the cache directory, if the case is already extracted there
    3. otherwise download the archive and extract it

    Args:
        case (str): which sample case to fetch; a key of :data:`SAMPLE_DATA`
        url (str): override the download location. Defaults to
            :func:`sample_data_url`, or ``$X4C_SAMPLE_URL`` if that is set.
        sha256 (str): expected checksum of the archive. Defaults to the case's
            registered checksum; `None` skips verification.
        verbose (bool): report what is being used or fetched
        force (bool): re-download even if the case is already present

    Returns:
        str: path to the case directory, ready to hand to :class:`x4c.Timeseries`

    Examples:
        >>> import x4c
        >>> case_dir = x4c.fetch_sample_data(case='cesm1')
    '''
    info = sample_data_info(case)
    casename = info['case']
    archive_name = info['archive']

    # 1. an explicit local copy
    env_dir = os.environ.get('X4C_SAMPLE_DIR')
    if env_dir and not force:
        candidate = env_dir if os.path.basename(env_dir.rstrip('/')) == casename \
            else os.path.join(env_dir, casename)
        if os.path.isdir(candidate):
            if verbose: p_hint(f'>>> using the sample case from $X4C_SAMPLE_DIR: {candidate}')
            return candidate
        raise FileNotFoundError(
            f'$X4C_SAMPLE_DIR is set to {env_dir!r} but no `{casename}` was found '
            'there. Unset it to download the sample instead.'
        )

    # 2. already extracted in the cache, laid out as in the data repository
    dest = os.path.join(cache_dir(), 'sample_data', info['dataset'])
    case_dir = os.path.join(dest, casename)
    if os.path.isdir(case_dir) and not force:
        if verbose: p_hint(f'>>> using the cached sample case: {case_dir}')
        return case_dir

    # 3. download and extract
    url = url or os.environ.get('X4C_SAMPLE_URL') or sample_data_url(case)
    sha256 = sha256 if sha256 is not None else info['sha256']

    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, archive_name)

    if not os.path.exists(archive) or force:
        if verbose: p_header(f'>>> Downloading the sample case ({url})')
        try:
            download(url, archive)
        except requests.HTTPError as e:
            raise RuntimeError(
                f'Could not download the sample case from {url} ({e}).\n'
                'If the Release asset has not been published yet, point x4c at a local '
                f'copy instead:\n    export X4C_SAMPLE_DIR=/path/containing/{casename}\n'
                'or pass an explicit `url=`.'
            ) from e
        _require_gzip(archive, url)

    if sha256:
        actual = _sha256(archive)
        if actual != sha256:
            os.remove(archive)
            raise ValueError(
                f'Checksum mismatch for {archive_name}: expected {sha256}, got '
                f'{actual}. The download was removed; retry, or pass sha256=None to '
                'skip verification.'
            )
        if verbose: p_success(f'>>> checksum verified ({sha256[:16]}...)')

    if verbose: p_header(f'>>> Extracting into {dest}')
    with tarfile.open(archive, 'r:gz') as tf:
        # refuse absolute paths and `..` escapes rather than trusting the archive
        for m in tf.getmembers():
            if m.name.startswith('/') or '..' in m.name.split('/'):
                raise ValueError(f'Refusing unsafe path in archive: {m.name!r}')
        tf.extractall(dest)

    if not os.path.isdir(case_dir):
        raise RuntimeError(
            f'{archive_name} did not contain a `{casename}` directory.'
        )

    os.remove(archive)   # the extracted tree is what matters; drop the tarball
    if verbose: p_success(f'>>> sample case ready: {case_dir}')
    return case_dir


def _sha256(path, chunk_size=1 << 20):
    ''' SHA-256 of a file, read in chunks so a multi-hundred-MB archive is fine '''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def download(url: str, fname: str, chunk_size=1024, show_bar=True, timeout=60):
    ''' Download `url` to `fname`, atomically

    Two failure modes this guards against, both of which used to poison the cache:

    - **An HTTP error body written as if it were data.** Without
      `raise_for_status()`, a 404 wrote GitHub's HTML error page into the target,
      and since callers only check `os.path.exists` the corrupt file was never
      re-fetched -- every later run failed inside `xr.open_dataset` with an
      unrelated-looking error.
    - **A partial file from an interrupted transfer.** The download goes to a
      temporary path in the same directory and is renamed into place only after it
      completes, so `fname` either does not exist or is whole.
    '''
    dirname = os.path.dirname(fname) or '.'
    os.makedirs(dirname, exist_ok=True)

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))

    tmp_fname = f'{fname}.part'
    try:
        if show_bar:
            with open(tmp_fname, 'wb') as file, tqdm(
                desc='Fetching data',
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
        else:
            with open(tmp_fname, 'wb') as file:
                for data in resp.iter_content(chunk_size=chunk_size):
                    file.write(data)

        os.replace(tmp_fname, fname)
    except BaseException:
        # never leave a truncated file where a valid cache entry is expected
        if os.path.exists(tmp_fname):
            os.remove(tmp_fname)
        raise


def move_with_overwrite(src, dst_dir):
    # Construct the full destination path
    dst = os.path.join(dst_dir, os.path.basename(src))
    
    if os.path.exists(dst):
        os.remove(dst)

    shutil.move(src, dst)

def rsync_move(src_paths, dst_dir):
    """
    Move a file or directory from src to dst using rsync.
    Equivalent to shutil.move, but more robust for large files and preserves metadata.
    """
    cmd = ['rsync', '-a']
    for path in src_paths:
        cmd += [str(path)]
    cmd += [str(dst_dir)]
    print(f'>>> {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def gcd(lat1, lon1, lat2, lon2, radius=6371.0):
    ''' 2D Great Circle Distance [km]

    Args:
        radius (float): Earth radius
    '''
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = radius * c
    return dist


def find_nearest2d(da:xr.DataArray, lat, lon, lat_name='lat', lon_name='lon', new_dim='sites', r=1):
    da_res = da.sel({lat_name: lat, lon_name:lon}, method='nearest')
    if da_res.isnull().any():
        if isinstance(lat, (int, float)): lat = [lat]
        if isinstance(lon, (int, float)): lon = [lon]
        da_res_list = []
        for la, lo in zip(lat, lon):
            # da_sub = da.sel({lat_name: slice(la-r, la+r), lon_name: slice(lo-r, lo+r)})  # won't work for some cases
            # mask_lat = (da.__dict__[lat_name] > la-r)&(da.__dict__[lat_name] < la+r)
            # mask_lon = (da.__dict__[lon_name] > lo-r)&(da.__dict__[lon_name] < lo+r)
            mask_lat = (da[lat_name] > la-r)&(da[lat_name] < la+r)
            mask_lon = (da[lon_name] > lo-r)&(da[lon_name] < lo+r)
            da_sub = da.sel({lat_name: mask_lat, lon_name: mask_lon})

            dist = gcd(da_sub[lat_name], da_sub[lon_name], la, lo)
            da_sub_valid = da_sub.where(~np.isnan(da_sub), drop=True)
            valid_mask = ~np.isnan(da_sub_valid)
            if valid_mask.sum() == 0:
                raise ValueError('No valid values found. Please try larger `r` values.')

            dist_min = dist.where(dist == dist.where(~np.isnan(da_sub_valid)).min(), drop=True)
            nearest_lat = dist_min[lat_name].values.item()
            nearest_lon = dist_min[lon_name].values.item()
            da_res = da_sub_valid.sel({lat_name: nearest_lat, lon_name: nearest_lon}, method='nearest')
            da_res_list.append(da_res)
        da_res = xr.concat(da_res_list, dim=new_dim).squeeze()

    return da_res

def move_and_overwrite(src_path, dst_dir):
    fname = os.path.basename(src_path)
    dst_path = os.path.join(dst_dir, fname)
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.move(src_path, dst_dir)