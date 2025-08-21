import os
import glob
import re
import itertools
import numpy as np
import xarray as xr
import xesmf as xe
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
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shapew
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    # print(f"Regridding from {in_shape} to {out_shape}")

    # Insert dummy dimension
    vars_with_ncol = [name for name in dataset.variables if "ncol" in dataset[name].dims]
    updated = dataset.copy().update(
        dataset[vars_with_ncol].transpose(..., "ncol").expand_dims("dummy", axis=-2)
    )

    # construct a regridder
    # use empty variables to tell xesmf the right shape
    # https://github.com/pangeo-data/xESMF/issues/202
    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    regridder = xe.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method="bilinear",
        reuse_weights=True,
        periodic=True,
    )

    # Actually regrid, after renaming
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        regridded = regridder(updated.rename({"dummy": "lat", "ncol": "lon"}), keep_attrs=True)
    # merge back any variables that didn't have the ncol dimension
    # And so were not regridded
    ds_out = xr.merge([dataset.drop_vars(regridded.variables, errors='ignore'), regridded])

    return ds_out

def annualize(ds, months=None, days_weighted=False):
    months = list(range(1, 13)) if months is None else np.abs(months)
    sds = ds.sel(time=ds['time.month'].isin(months))
    anchor = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    idx = months[-1]-1

    if days_weighted:
        # weighted version
        days_in_month = sds.time.dt.days_in_month
        weights = days_in_month.groupby('time.year') / days_in_month.groupby('time.year').sum()
        ds_weighted = sds * weights
        ds_ann = ds_weighted.resample(time=f'YE-{anchor[idx]}').sum()
        ds_ann = ds_ann.where(sds.notnull())
    else:
        ds_ann = sds.resample(time=f'YE-{anchor[idx]}').mean()  # unweighted version

    try:
        ds_ann.name = sds.name
    except:
        pass

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
    return m

def update_attrs(da, da_src):
    da.attrs = dict(da_src.attrs)
    if 'comp' in da.attrs and 'time' in da.coords:
        da.time.attrs['long_name'] = 'Model Year'

    return da

def update_ds(ds, path, vn=None, comp=None, grid=None, adjust_month=False,
              gw_name=None, lat_name=None, lon_name=None):
    if adjust_month:
        ds['time'] = ds['time'].get_index('time') - datetime.timedelta(days=1)

    if type(path) in (list, tuple):
        ds.attrs['path'] = [os.path.abspath(p) for p in path]
    else:
        ds.attrs['path'] = os.path.abspath(path)

    if vn is not None: ds.attrs['vn'] = vn
    if comp is not None: ds.attrs['comp'] = comp
    if grid is not None: ds.attrs['grid'] = grid

    if 'comp' in ds.attrs:
        gw_dict = {
            'atm': 'area',
            'ocn': 'TAREA',
            'ice': 'tarea',
            'lnd': 'area',
        }

        lon_dict = {
            'atm': 'lon',
            'ocn': 'TLONG',
            'ice': 'TLON',
            'lnd': 'lon',
        }

        lat_dict = {
            'atm': 'lat',
            'ocn': 'TLAT',
            'ice': 'TLAT',
            'lnd': 'lat',
        }

        gw_name = gw_dict[ds.attrs['comp']] if gw_name is None else gw_name
        lat_name = lat_dict[ds.attrs['comp']] if lat_name is None else lat_name
        lon_name = lon_dict[ds.attrs['comp']] if lon_name is None else lon_name

    if gw_name is not None and gw_name in ds:
        ds.attrs['gw'] = ds[gw_name]
    elif 'gw' in ds.variables:
        ds.attrs['gw'] = ds['gw']
    elif 'lat' in ds.variables:
        ds.attrs['gw'] = ds['lat']

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

# def find_paths(root_dir, path_pattern='comp/proc/tseries/month_1/casename.mdl.hstr.vn.timespan.nc', delimiters=['/', '.'],
#                avoid_list=None, verbose=False, **kws):
#     s = path_pattern
#     for d in delimiters:
#         s = ' '.join(s.split(d))
#     path_elements = s.split()

#     for e in path_elements:
#         if e in kws:
#             value = kws[e]
#             if isinstance(value, list):
#                 pattern_str = '{' + ','.join(value) + '}'
#                 path_pattern = path_pattern.replace(e, pattern_str)
#             else:
#                 path_pattern = path_pattern.replace(e, value)
#         elif e in ['proc', 'tseries', 'month_1', 'nc']:
#             pass
#         elif e in ['timespan', 'date']:
#             path_pattern = path_pattern.replace(e, '*[0-9]')
#         else:
#             path_pattern = path_pattern.replace(e, '*')

#     path_patterns = expand_braces(path_pattern)
#     if verbose: p_header(f'path_patterns: {path_patterns}')
#     paths = []
#     for pat in path_patterns:
#         paths_tmp = glob.glob(os.path.join(root_dir, pat))
#         paths.extend(paths_tmp)

#     # sort based on timespak h
#     paths = sorted(paths, key=lambda x: x.split('.')[-2])
#     if avoid_list is not None:
#         paths_new = [] 
#         for path in paths:
#             add_path = True
#             for avoid_str in avoid_list:
#                 if avoid_str in path:
#                     add_path = False
#                     break
#             if add_path: paths_new.append(path)
#         paths = paths_new
#     return paths

# def get_hstr(paths, mdl):
#     hstr_set = set()

#     # Pattern to extract what's after mdl.
#     pattern = re.compile(rf'{re.escape(mdl)}\.((?:[^0-9][^.]*\.?)+)')

#     # Pattern to remove trailing date strings like .0001-01 or .0001-01-0001-12
#     date_like_pattern = re.compile(r'(\.?\d{4}-\d{2}(?:-\d{4}-\d{2})?)$')

#     for path in paths:
#         filename = os.path.basename(path)
#         match = pattern.search(filename)
#         if match:
#             hstr = match.group(1)
#             # Remove date-like suffix
#             hstr = date_like_pattern.sub('', hstr)
#             hstr = hstr.rstrip('.')
#             if 'h' in hstr:  # Only keep if 'h' is present
#                 hstr_set.add(hstr)

#     return sorted(hstr_set)

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
            path_pattern = path_pattern.replace(e, '*[0-9]')
        else:
            path_pattern = path_pattern.replace(e, '*')

    path_patterns = expand_braces(path_pattern)
    if verbose: p_header(f'path_patterns: {path_patterns}')
    paths = []
    for pat in path_patterns:
        paths_tmp = glob.glob(os.path.join(root_dir, pat))
        paths.extend(paths_tmp)

    # sort based on timespak h
    paths = sorted(paths, key=lambda x: x.split('.')[-2])
    if avoid_list is not None:
        paths_new = [] 
        for path in paths:
            add_path = True
            for avoid_str in avoid_list:
                if avoid_str in path:
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
            next = add_months(current, timestep * 12)
            current_end = minus_months(next, 1)
        elif timestep_unit == 'month':
            next = add_months(current, timestep)
            current_end = minus_months(next, 1)
        elif timestep_unit == 'day':
            next = current + datetime.timedelta(days=timestep)
            current_end = next - datetime.timedelta(days=1)
        elif timestep_unit == 'hour':
            next = current + datetime.timedelta(hours=timestep)
            current_end = next - datetime.timedelta(hours=1)
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
        current = next

    return timestamp_list

def cesm_str2datetime(s: str) -> datetime.datetime:
    """Convert CESM timestamp 'YYYY-MM-DD-SSSSS' to a datetime."""
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
        year = s.split('-')
        res = datetime.datetime(int(year), 1, 1)

    return res

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

def download(url: str, fname: str, chunk_size=1024, show_bar=True):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    if show_bar:
        with open(fname, 'wb') as file, tqdm(
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
        with open(fname, 'wb') as file:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)


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
    print('>>> {cmd}')
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