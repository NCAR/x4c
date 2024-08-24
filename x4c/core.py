import xarray as xr
xr.set_options(keep_attrs=True)

import xesmf as xe
from . import utils

import os
dirpath = os.path.dirname(__file__)

def load_dataset(path, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Load a netCDF file and form a `xarray.Dataset`

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds = xr.load_dataset(path, **kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_dataset(path, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Open a netCDF file and form a `xarray.Dataset` with a lazy load mode

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds = xr.open_dataset(path, **kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_mfdataset(paths, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Open multiple netCDF files and form a `xarray.Dataset` in a lazy load mode

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds0 = xr.open_dataset(paths[0], decode_cf=False)
    dims_other_than_time = list(ds0.dims)
    try:
        dims_other_than_time.remove('time')
    except:
        pass

    chunk_dict = {k: -1 for k in dims_other_than_time}

    _kws = {
        'data_vars': 'minimal',
        'coords': 'minimal',
        'compat': 'override',
        'chunks': chunk_dict,
        'parallel': True,
    }
    _kws.update(kws)
    ds = xr.open_mfdataset(paths, **_kws)
    ds = utils.update_ds(ds, vn=vn, path=paths, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

@xr.register_dataset_accessor('x')
class XDataset:
    def __init__(self, ds=None):
        self.ds = ds

    def regrid(self, dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True):
        ''' Regrid the CESM output to a normal lat/lon grid

        Supported atmosphere regridding: ne16np4, ne16pg3, ne30np4, ne30pg3, ne120np4, ne120pg4 TO 1x1d / 2x2d.
        Supported ocean regridding: any grid similar to g16 TO 1x1d / 2x2d.
        For any other regridding, `weight_file` must be provided by the user.

        For the atmosphere grid regridding, the default method is area-weighted;
        while for the ocean grid, the default is bilinear.

        Args:
            dlon (float): longitude spacing
            dlat (float): latitude spacing
            weight_file (str): the path to an ESMF-generated weighting file for regridding
            gs (str): grid style in 'T' or 'U' for the ocean grid
            method (str): regridding method for the ocean grid
            periodic (bool): the assumption of the periodicity of the data when perform the regrid method

        '''
        comp = self.ds.attrs['comp']
        grid = self.ds.attrs['grid']

        if weight_file is not None:
            # using a user-provided weight file for any unsupported regridding
            ds_rgd = utils.regrid_cam_se(ds, weight_file=weight_file)
        else:
            if grid[:2] == 'ne':
                # SE grid
                if grid in ['ne16np4', 'ne16pg3', 'ne30np4', 'ne30pg3', 'ne120np4', 'ne120pg3']:
                    ds = self.ds.copy()
                    if comp == 'lnd':
                        ds = ds.rename_dims({'lndgrid': 'ncol'})

                    wgt_fpath = os.path.join(dirpath, f'./regrid_wgts/map_{grid}_TO_{dlon}x{dlat}d_aave.nc.gz')
                    if not os.path.exists(wgt_fpath):
                        url = f'https://github.com/fzhu2e/x4c-regrid-wgts/raw/main/data/map_{grid}_TO_{dlon}x{dlat}d_aave.nc.gz'
                        utils.p_header(f'Downloading the weight file from: {url}')
                        utils.download(url, wgt_fpath)

                    ds_rgd = utils.regrid_cam_se(ds, weight_file=wgt_fpath)
                else:
                    raise ValueError('The specified `grid` is not supported. Please specify a `weight_file`.')

            elif grid[:2] == 'fv':
                # FV grid
                ds = xr.Dataset()
                ds['lat'] = self.ds.lat
                ds['lon'] = self.ds.lon

                regridder = xe.Regridder(
                    ds, xe.util.grid_global(dlon, dlat, cf=True, lon1=360),
                    method=method, periodic=periodic,
                )
                ds_rgd = regridder(self.ds, keep_attrs=True)

            elif comp in ['ocn', 'ice']:
                # ocn grid
                ds = xr.Dataset()
                if gs == 'T':
                    ds['lat'] = self.ds.TLAT
                    if comp == 'ice':
                        ds['lon'] = self.ds.TLON
                    else:
                        ds['lon'] = self.ds.TLONG
                elif gs == 'U':
                    ds['lat'] = self.ds.ULAT
                    if comp == 'ice':
                        ds['lon'] = self.ds.ULON
                    else:
                        ds['lon'] = self.ds.ULONG
                else:
                    raise ValueError('`gs` options: {"T", "U"}.')

                regridder = xe.Regridder(
                    ds, xe.util.grid_global(dlon, dlat, cf=True, lon1=360),
                    method=method, periodic=periodic,
                )

                ds_rgd = regridder(self.ds, keep_attrs=True)

            else:
                raise ValueError(f'grid [{grid}] is not supported; please provide a corresponding `weight_file`.')

        try:
            ds_rgd = ds_rgd.drop_vars('latitude_longitude')
        except:
            pass

        ds_rgd.attrs = dict(self.ds.attrs)
        # utils.p_success(f'Dataset regridded to regular grid: [dlon: {dlon} x dlat: {dlat}]')
        if 'lat' in ds_rgd.attrs: del(ds_rgd.attrs['lat'])
        if 'lon' in ds_rgd.attrs: del(ds_rgd.attrs['lon'])
        return ds_rgd


    def __getitem__(self, key):
        da = self.ds[key]

        if 'path' in self.ds.attrs:
            da.attrs['path'] = self.ds.attrs['path']

        if 'gw' in self.ds:
            da.attrs['gw'] = self.ds['gw'].fillna(0)

        if 'lat' in self.ds:
            da.attrs['lat'] = self.ds['lat']

        if 'lon' in self.ds:
            da.attrs['lon'] = self.ds['lon']

        if 'dz' in self.ds:
            da.attrs['dz'] = self.ds['dz']

        if 'comp' in self.ds.attrs:
            da.attrs['comp'] = self.ds.attrs['comp']
            if 'time' in da.coords:
                da.time.attrs['long_name'] = 'Model Year'

        if 'grid' in self.ds.attrs:
            da.attrs['grid'] = self.ds.attrs['grid']


        return da

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon', 'dz']:
            if v in self.ds.attrs: del(self.ds.attrs[v])

        return self.ds.to_netcdf(path, **kws)

    @property
    def climo(self):
        ds = self.ds.groupby('time.month').mean(dim='time')
        ds.attrs['climo_period'] = (self.ds['time.year'].values[0], self.ds['time.year'].values[-1])
        if 'comp' in self.ds.attrs: ds.attrs['comp'] = self.ds.attrs['comp']
        if 'grid' in self.ds.attrs: ds.attrs['grid'] = self.ds.attrs['grid']
        if 'month' in ds.coords:
            ds = ds.rename({'month': 'time'})
        return ds
        

@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, da=None):
        self.da = da

    def regrid(self, **kws):
        ds_rgd = self.ds.x.regrid(**kws)
        da = ds_rgd.x.da
        da.name = self.da.name
        if 'lat' in da.attrs: del(da.attrs['lat'])
        if 'lon' in da.attrs: del(da.attrs['lon'])
        return da

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon', 'dz']:
            if v in self.da.attrs: del(self.da.attrs[v])

        return self.da.to_netcdf(path, **kws)

    @property
    def climo(self):
        da = self.da.groupby('time.month').mean(dim='time')
        da.attrs['climo_period'] = (self.da['time.year'].values[0], self.da['time.year'].values[-1])
        if 'comp' in self.da.attrs: da.attrs['comp'] = self.da.attrs['comp']
        if 'grid' in self.da.attrs: da.attrs['grid'] = self.da.attrs['grid']
        if 'month' in da.coords:
            da = da.rename({'month': 'time'})
        return da
