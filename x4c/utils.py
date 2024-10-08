import os
import glob
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
    regridded = regridder(updated.rename({"dummy": "lat", "ncol": "lon"}), keep_attrs=True)
    # merge back any variables that didn't have the ncol dimension
    # And so were not regridded
    ds_out = xr.merge([dataset.drop_vars(regridded.variables, errors='ignore'), regridded])

    return ds_out

def annualize(ds, months=None):
    months = list(range(1, 13)) if months is None else np.abs(months)
    sds = ds.sel(time=ds['time.month'].isin(months))
    anchor = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    idx = months[-1]-1
    try:
        ds_ann = sds.resample(time=f'YE-{anchor[idx]}').mean()  # new version
    except:
        ds_ann = sds.resample(time=f'A-{anchor[idx]}').mean()   # old version
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
        grid_weight_dict = {
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

        gw_name = grid_weight_dict[ds.attrs['comp']] if gw_name is None else gw_name
        lat_name = lat_dict[ds.attrs['comp']] if lat_name is None else lat_name
        lon_name = lon_dict[ds.attrs['comp']] if lon_name is None else lon_name

    if gw_name is not None and gw_name in ds: ds['gw'] = ds[gw_name]
    if lat_name is not None and lat_name in ds: ds['lat'] = ds[lat_name]
    if lon_name is not None and lon_name in ds: ds['lon'] = ds[lon_name]

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
            p_warning("The inpu `xarray.DataArray` doesn't have a unit.")

    return da

def find_paths(root_dir, path_pattern='comp/proc/tseries/month_1/casename.mdl.h_str.vn.timespan.nc', delimiters=['/', '.'],
               avoid_list=None, verbose=False, **kws):
    s = path_pattern
    for d in delimiters:
        s = ' '.join(s.split(d))
    path_elements = s.split()

    for e in path_elements:
        if e in kws:
            path_pattern = path_pattern.replace(e, kws[e])
        elif e in ['proc', 'tseries', 'month_1', 'nc']:
            pass
        elif e in ['timespan', 'date']:
            path_pattern = path_pattern.replace(e, '*[0-9]')
        else:
            path_pattern = path_pattern.replace(e, '*')

    if verbose: p_header(f'path_pattern: {path_pattern}')
    # sort based on timespan
    paths = sorted(glob.glob(os.path.join(root_dir, path_pattern)), key=lambda x: x.split('.')[-2])
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