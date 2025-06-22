import xarray as xr
xr.set_options(keep_attrs=True)

import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geocat.comp as gc

from . import utils, visual
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
        comp (str): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    _kws = {'use_cftime': True, 'decode_timedelta': True}
    _kws.update(kws)
    ds = xr.open_dataset(path, **_kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, grid=grid, adjust_month=adjust_month)
    return ds

def open_mfdataset(paths, adjust_month=False, comp=None, grid=None, vn=None, **kws):
    ''' Open multiple netCDF files and form a `xarray.Dataset` in a lazy load mode

    Args:
        path (str): path to the netCDF file
        adjust_month (bool): adjust the month of the `xarray.Dataset` (the default CESM output has a month shift)
        comp (str): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"
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
        'use_cftime': True,
        'decode_timedelta': True,
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
            ds_rgd = utils.regrid_cam_se(self.ds, weight_file=weight_file)
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

    def get_plev(self, ps, vn=None, lev_mode='hybrid', **kws):
        _kws = {'lev_dim': 'lev'}
        if 'hyam' in self.ds: _kws['hyam'] = self.ds['hyam']
        if 'hybm' in self.ds: _kws['hybm'] = self.ds['hybm']

        _kws.update(kws)
        if vn is None:
            da = self.da
            vn = self.ds.attrs['vn']
        else:
            da = self.ds[vn]

        if isinstance(ps, xr.Dataset):
            ps_da = ps['PS']
        elif isinstance(ps, xr.DataArray):
            ps_da = ps

        if lev_mode == 'hybrid':
            da_plev = gc.interpolation.interp_hybrid_to_pressure(da, ps_da, **_kws)
        else:
            raise ValueError('`lev_mode` unknown')

        ds_plev = self.ds.copy()
        del(ds_plev[vn])
        ds_plev[vn] = da_plev
        return ds_plev

    def zavg(self, depth_top, depth_bot, vn=None):
        if vn is None:
            da = self.da
            vn = self.ds.attrs['vn']
        else:
            da = self.ds[vn]

        da_zavg = da.sel(z_t=slice(depth_top, depth_bot)).weighted(self.ds['dz']).mean('z_t')

        ds_zavg = self.ds.copy()
        ds_zavg[vn] = da_zavg
        return ds_zavg
        
    def annualize(self, months=None, days_weighted=False, time2year=False):
        ''' Annualize/seasonalize a `xarray.Dataset`

        Args:
            months (list of int): a list of integers to represent month combinations,
                e.g., `None` means calendar year annualization, [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

        '''
        ds_ann = utils.annualize(self.ds, months=months, days_weighted=days_weighted)
        ds_ann.attrs = dict(self.ds.attrs)
        if time2year:
            years = [t.year for t in ds_ann.time.values]
            ds_ann = ds_ann.assign_coords({'time': years})
        return ds_ann


    def __getitem__(self, key):
        da = self.ds[key]

        if 'path' in self.ds.attrs:
            da.attrs['path'] = self.ds.attrs['path']

        if 'gw' in self.ds.attrs:
            da.attrs['gw'] = self.ds.attrs['gw'].fillna(0)

        if 'lat' in self.ds.data_vars:
            da.coords['lat'] = self.ds['lat']

        if 'lon' in self.ds.data_vars:
            da.coords['lon'] = self.ds['lon']

        if 'lat' in self.ds.attrs:
            da.attrs['lat'] = self.ds.attrs['lat']

        if 'lon' in self.ds.attrs:
            da.attrs['lon'] = self.ds.attrs['lon']

        if 'dz' in self.ds:
            da.attrs['dz'] = self.ds['dz']

        if 'comp' in self.ds.attrs:
            da.attrs['comp'] = self.ds.attrs['comp']
            if 'time' in da.coords:
                da.time.attrs['long_name'] = 'Model Year'

        if 'grid' in self.ds.attrs:
            da.attrs['grid'] = self.ds.attrs['grid']


        return da

    @property
    def da(self):
        ''' get its `xarray.DataArray` version '''
        if 'vn' in self.ds.attrs:
            vn = self.ds.attrs['vn']
            return self.ds.x[vn]
        else:
            raise ValueError('`vn` not existed in `Dataset.attrs`')

    @property
    def climo(self):
        ds = self.ds.groupby('time.month').mean(dim='time')
        ds.attrs['climo_period'] = (self.ds['time.year'].values[0], self.ds['time.year'].values[-1])
        if 'comp' in self.ds.attrs: ds.attrs['comp'] = self.ds.attrs['comp']
        if 'grid' in self.ds.attrs: ds.attrs['grid'] = self.ds.attrs['grid']
        if 'month' in ds.coords:
            ds = ds.rename({'month': 'time'})
        return ds

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon', 'dz']:
            if v in self.ds.attrs: del(self.ds.attrs[v])

        return self.ds.to_netcdf(path, **kws)
        

@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, da=None):
        self.da = da

    # def nearest2d(self, lat, lon, lat_name='lat', lon_name='lon', new_dim='sites', extra_dim=None, r=1):
    #     if extra_dim is None:
    #         da_res = utils.find_nearest2d(self.da, lat, lon, lat_name=lat_name, lon_name=lon_name, new_dim=new_dim, r=r)
    #     else:
    #         da_res_extra_list = []
    #         for i in range(self.da.sizes[extra_dim]):
    #             da_sub = self.da.isel({extra_dim: i})
    #             da_sub_res = utils.find_nearest2d(da_sub, lat, lon, lat_name=lat_name, lon_name=lon_name, new_dim=new_dim, r=r)
    #             da_res_extra_list.append(da_sub_res)
    #         da_res = xr.concat(da_res_extra_list, dim=extra_dim).squeeze()

    #     return da_res

    def annualize(self, months=None, days_weighted=False):
        ''' Annualize/seasonalize a `xarray.DataArray`

        Args:
            months (list of int): a list of integers to represent month combinations,
                e.g., [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

        '''
        da = utils.annualize(self.da, months=months, days_weighted=days_weighted)
        da = utils.update_attrs(da, self.da)
        return da

    def regrid(self, **kws):
        ds_rgd = self.ds.x.regrid(**kws)
        da = ds_rgd.x.da
        da.name = self.da.name
        if 'lat' in da.attrs: del(da.attrs['lat'])
        if 'lon' in da.attrs: del(da.attrs['lon'])
        return da

    def get_plev(self, **kws):
        '''
        See: https://geocat-comp.readthedocs.io/en/v2024.04.0/user_api/generated/geocat.comp.interpolation.interp_hybrid_to_pressure.html
        '''
        _kws = {'lev_dim': 'lev'}
        _kws.update(kws)
        da = gc.interpolation.interp_hybrid_to_pressure(self.da, **_kws)
        da.name = self.da.name
        return da

    def zavg(self, depth_top, depth_bot):
        da_zavg = self.da.sel(z_t=slice(depth_top, depth_bot)).weighted(self.da.attrs['dz']).mean('z_t')
        return da_zavg

    def to_netcdf(self, path, **kws):
        for v in ['gw', 'lat', 'lon', 'dz']:
            if v in self.da.attrs: del(self.da.attrs[v])

        return self.da.to_netcdf(path, **kws)

    def nearest2d(self, lat=None, lon=None, lat_coord='lat', lon_coord='lon', lat_dim='lat', lon_dim='lon'):
        '''
        Select the nearest non-NaN grid point.
    
        Parameters:
            da: xarray.DataArray or Dataset
            lat_name, lon_name: names of coordinate variables in da
            target_lat, target_lon: float or 1D arrays of lat/lon values to match
    
        Returns:
            xarray.DataArray or Dataset sliced at nearest grid points
        '''
        lats = self.da.coords[lat_coord].values
        lons = self.da.coords[lon_coord].values
        if lats.ndim == 2 and lons.ndim == 2:
            lats2d = self.da.coords[lat_coord].values
            lons2d = self.da.coords[lon_coord].values
        elif lats.ndim == 1 and lons.ndim == 1:
            lats1d = self.da.coords[lat_coord].values
            lons1d = self.da.coords[lon_coord].values
            lons2d, lats2d = np.meshgrid(lons1d, lats1d)

        other_dims = set(self.da.dims) - set([lat_dim, lon_dim])
        isel_indexer = {dim: 0 for dim in other_dims}
        da_latlon = self.da.isel(**isel_indexer)
        mask = ~np.isnan(da_latlon.values)

        valid_lats = lats2d[mask]
        valid_lons = lons2d[mask]
        valid_indices = np.array(np.nonzero(mask)).T

        target_lat = np.atleast_1d(lat)
        target_lon = np.atleast_1d(lon)

        sel_list = []
        for lat0, lon0 in zip(target_lat, target_lon):
            dists = utils.gcd(lat0, lon0, valid_lats, valid_lons)
            best_idx = np.argmin(dists)
            iy, ix = valid_indices[best_idx]
            sel_list.append(self.da.isel({lat_dim: iy, lon_dim: ix}))

        return xr.concat(sel_list, dim='site').assign_coords(site=np.arange(len(sel_list)))


    @property
    def ds(self):
        ''' get its `xarray.Dataset` version '''
        ds_tmp = self.da.to_dataset()

        for v in ['gw', 'lat', 'lon']:
            if v in self.da.attrs: ds_tmp[v] = self.da.attrs[v]

        for v in ['comp', 'grid']:
            if v in self.da.attrs: ds_tmp.attrs[v] = self.da.attrs[v]
        
        ds_tmp[self.da.name] = self.da
        ds_tmp.attrs['vn'] = self.da.name
        return ds_tmp

    @property
    def gm(self):
        ''' the global area-weighted mean '''
        gw = self.da.attrs['gw']
        da = self.da.weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Global Mean {da.attrs["long_name"]}'
        return da

    @property
    def nhm(self):
        ''' the NH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat>0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'NH Mean {da.attrs["long_name"]}'
        return da

    @property
    def shm(self):
        ''' the SH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat<0).weighted(gw).mean(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'SH Mean {da.attrs["long_name"]}'
        return da

    @property
    def gs(self):
        ''' the global area-weighted sum '''
        gw = self.da.attrs['gw']
        da = self.da.weighted(gw).sum(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Global Sum {da.attrs["long_name"]}'
        return da

    @property
    def nhs(self):
        ''' the NH area-weighted sum '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat>0).weighted(gw).sum(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'NH Sum {da.attrs["long_name"]}'
        return da

    @property
    def shs(self):
        ''' the SH area-weighted sum '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        da = self.da.where(lat<0).weighted(gw).sum(list(gw.dims))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'SH Sum {da.attrs["long_name"]}'
        return da

    @property
    def somin(self):
        ''' the Southern Ocean min'''
        da = self.da.sel(lat=slice(-90, -28)).min(('z_t', 'lat'))
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Southern Ocean (90°S-28°S) {da.attrs["long_name"]}'
        return da

    @property
    def zm(self):
        ''' the zonal mean
        '''
        if 'lon' not in self.da.dims:
            da = self.da.x.regrid().mean('lon')
        else:
            da = self.da.mean('lon')

        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Zonal Mean {da.attrs["long_name"]}'
        return da

    @property
    def climo(self):
        da = self.da.groupby('time.month').mean(dim='time')
        da.attrs['climo_period'] = (self.da['time.year'].values[0], self.da['time.year'].values[-1])
        if 'comp' in self.da.attrs: da.attrs['comp'] = self.da.attrs['comp']
        if 'grid' in self.da.attrs: da.attrs['grid'] = self.da.attrs['grid']
        if 'month' in da.coords:
            da = da.rename({'month': 'time'})
        return da

    def geo_mean(self, ind=None, latlon_range=(-90, 90, 0, 360), **kws):
        ''' The lat-weighted mean given a lat/lon range or a climate index name

        Args:
            latlon_range (tuple or list): the lat/lon range for lat-weighted average 
                in format of (lat_min, lat_max, lon_min, lon_max)

            ind (str): a climate index name; supported names include:
            
                * 'nino3.4'
                * 'nino1+2'
                * 'nino3'
                * 'nino4'
                * 'tpi'
                * 'wp'
                * 'dmi'
                * 'iobw'
        '''

        if ind is None:
            lat_min, lat_max, lon_min, lon_max = latlon_range
            da = utils.geo_mean(self.da, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, **kws)
        elif ind == 'nino3.4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-170, 360), lon_max=np.mod(-120, 360), **kws)
        elif ind == 'nino1+2':
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(-90, 360), lon_max=np.mod(-80, 360), **kws)
        elif ind == 'nino3':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(-150, 360), lon_max=np.mod(-90, 360), **kws)
        elif ind == 'nino4':
            da = utils.geo_mean(self.da, lat_min=-5, lat_max=5, lon_min=np.mod(160, 360), lon_max=np.mod(-150, 360), **kws)
        elif ind == 'wpi':
            # Western Pacific Index
            da = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(120, 360), lon_max=np.mod(150, 360), **kws)
        elif ind == 'tpi':
            # Tri-Pole Index
            v1 = utils.geo_mean(self.da, lat_min=25, lat_max=45, lon_min=np.mod(140, 360), lon_max=np.mod(-145, 360), **kws)
            v2 = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=np.mod(170, 360), lon_max=np.mod(-90, 360), **kws)
            v3 = utils.geo_mean(self.da, lat_min=-50, lat_max=-15, lon_min=np.mod(150, 360), lon_max=np.mod(-160, 360), **kws)
            da = v2 - (v1 + v3)/2
        elif ind == 'dmi':
            # Indian Ocean Dipole Mode
            dmiw = utils.geo_mean(self.da, lat_min=-10, lat_max=10, lon_min=50 ,lon_max=70, **kws)
            dmie = utils.geo_mean(self.da,lat_min=-10,lat_max=0,lon_min=90,lon_max=110, **kws)
            da = dmiw - dmie
        elif ind == 'iobw':
            # Indian Ocean Basin Wide
            da =  utils.geo_mean(self.da, lat_min=-20, lat_max=20, lon_min=40 ,lon_max=100, **kws)
        else:
            raise ValueError('`ind` options: {"nino3.4", "nino1+2", "nino3", "nino4", "wpi", "tpi", "dmi", "iobw"}')

        da.attrs = dict(self.da.attrs)
        if 'comp' in da.attrs and 'time' in da.coords:
            da.time.attrs['long_name'] = 'Model Year'
        return da

    def is_latlon(self):
        da = self.da.squeeze()
        return ('lat' in da.coords and 'lon' in da.coords)

    def is_cam_se(self):
        da = self.da.squeeze()
        return ('ncol' in da.dims)

    def is_pop(self):
        da = self.da.squeeze()
        return ('nlat' in da.dims and 'nlon' in da.dims)

    def is_map(self):
        return self.is_latlon() or self.is_cam_se() or self.is_pop()

    def plot(self, title=None, figsize=None, ax=None, latlon_range=None, add_clabels=False, clevels=None, clabel_kwargs=None,
             projection='Robinson', transform='PlateCarree', central_longitude=180, proj_args=None, bad_color='dimgray',
             add_gridlines=False, gridline_labels=True, gridline_style='--', ssv=None, log=False, vmin=None, vmax=None,
             coastline_zorder=99, coastline_width=1, site_markersizes=100, df_sites=None, colname_dict=None, gs='T', ux=False,
             site_marker_dict=None, site_color_dict=None, count_site_num=False, lgd_kws=None, legend=True, return_im=False, **kws):
        ''' The plotting functionality

        Args:
            title (str): figure title
            figsize (tuple or list): figure size in format of (w, h)
            ax (`matplotlib.axes`): a `matplotlib.axes`
            latlon_range (tuple or list): lat/lon range in format of (lat_min, lat_max, lon_min, lon_max)
            projection (str): a projection name supported by `Cartopy`
            transform (str): a projection name supported by `Cartopy`
            central_longitude (float): the central longitude of the map to plot
            proj_args (dict): other keyword arguments for projection
            add_gridlines (bool): if True, the map will be added with gridlines
            gridline_labels (bool): if True, the lat/lon ticklabels will appear
            gridline_style (str): the gridline style, e.g., '-', '--'
            ssv (`xarray.DataArray`): a sea surface variable used for plotting the coastlines
            gs (str): grid style in 'T' or 'U' for the ocean grid
            coastline_zorder (int): the layer order for the coastlines
            coastline_width (float): the width of the coastlines
            df_sites (`pandas.DataFrame`): a `pandas.DataFrame` that stores the information of a collection of sites
            colname_dict (dict): a dictionary of column names for `df_sites` in the "key:value" format "assumed name:real name"

        '''
        da = self.da.squeeze()
        ndim = len(da.dims)
        if self.is_map():
            # map
            if ax is None:
                if figsize is None: figsize = (10, 3)
                fig = plt.figure(figsize=figsize)
                proj_args = {} if proj_args is None else proj_args
                proj_args_default = {'central_longitude': central_longitude}
                proj_args_default.update(proj_args)
                _projection = ccrs.__dict__[projection](**proj_args_default)
                ax = plt.subplot(projection=_projection)

            if 'units' in da.attrs:
                cbar_lb = f'{da.name} [{da.units}]'
            else:
                cbar_lb = da.name

            _transform = ccrs.__dict__[transform]()
            _plt_kws = {
                'transform': _transform,
                'extend': 'both',
                'cmap': visual.infer_cmap(da),
                'cbar_kwargs': {
                    'label': cbar_lb,
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)
            if 'add_colorbar' in kws and kws['add_colorbar'] is False:
                del(_plt_kws['cbar_kwargs'])

            if latlon_range is not None:
                lat_min, lat_max, lon_min, lon_max = latlon_range
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=_transform)

            if add_gridlines:
                gl = ax.gridlines(linestyle=gridline_style, draw_labels=gridline_labels)
                gl.top_labels = False
                gl.right_labels = False

            if 'cyclic' in kws:
                cyclic = kws['cyclic']
                cyclic = _plt_kws.pop('cyclic')
            else:
                cyclic = False

            if cyclic:
                da_original = da.copy()
                da = utils.add_cyclic_point(da_original)
                da.name = da_original.name
                da.attrs = da_original.attrs

            # add coastlines
            if ssv is not None:
                if cyclic: ssv = utils.add_cyclic_point(ssv)
                # use a sea surface variable with NaNs for coastline plotting
                if ('comp' in da.attrs) and (da.attrs['comp'] in ['ocn', 'ice']):
                    ax.contourf(ssv.lon, ssv.lat, np.isnan(ssv), levels=[0.5, 1.5], colors='white', transform=_transform, zorder=2)
                ax.contour(ssv.lon, ssv.lat, np.isnan(ssv), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)
            elif ('comp' in da.attrs) and (da.attrs['comp'] in ['ocn', 'ice']) and ('lat' in da.coords and 'lon' in da.coords):
                # using NaNs in the dataarray itself for coastline plotting
                ax.contour(da.lon, da.lat, np.isnan(da), levels=[0, 1], colors='k', transform=_transform, zorder=coastline_zorder, linewidths=coastline_width)
            else:
                # use the modern coastlines from Cartopy
                ax.coastlines(zorder=coastline_zorder, linewidth=coastline_width)

            if log: _plt_kws.update({'norm': LogNorm(vmin=vmin, vmax=vmax)})

            if self.is_cam_se():
               # CAM-SE grid without regridding
                levels = _plt_kws['levels'] if 'levels' in _plt_kws else None
                if levels is None:
                    cmap = _plt_kws['cmap']
                    norm = None
                else:
                    extend = _plt_kws['extend']
                    nbins = len(levels)-1
                    ncolors = nbins + {'neither': 0, 'min': 1, 'max': 1, 'both': 2}[extend]
                    cmap = plt.get_cmap(_plt_kws['cmap'], ncolors)
                    norm = BoundaryNorm(boundaries=levels, ncolors=ncolors, extend=extend)

                if ux is False:
                    # using tricontourf for CAM-SE grid
                    __plt_kws = _plt_kws.copy()
                    del(__plt_kws['cbar_kwargs'])
                    del(__plt_kws['cmap'])
                    im = ax.tricontourf(da.lon, da.lat, da, cmap=cmap, norm=norm, **__plt_kws)
                else:
                    # using UXarray for CAM-SE grid
                    try:
                        import uxarray as ux
                    except:
                        raise ImportError('UXarray is required for this method. Please install it via `conda install -c conda-forge uxarray`.')

                    grid = da.attrs['grid']
                    wgt_fpath = os.path.join(dirpath, f'./regrid_wgts/scrip_{grid}.nc.gz')
                    if not os.path.exists(wgt_fpath):
                        url = f'https://github.com/fzhu2e/x4c-regrid-wgts/raw/main/data/scrip_{grid}.nc.gz'
                        utils.p_header(f'Downloading the weight file from: {url}')
                        utils.download(url, wgt_fpath)

                    uxgrid = ux.open_grid(wgt_fpath)
                    uxda = ux.UxDataArray(da, uxgrid=uxgrid).rename({'ncol': 'n_face'})

                    pc = uxda.to_polycollection(projection=_projection)
                    pc.set_cmap(cmap)
                    pc.set_norm(norm)
                    pc.set_antialiased(False)
                    im = ax.add_collection(pc)

                if latlon_range is None: ax.set_global()
                if 'add_colorbar' in kws and kws['add_colorbar'] is False:
                    pass
                else:
                    cbar = plt.colorbar(im, ax=ax, extend=_plt_kws['extend'], **_plt_kws['cbar_kwargs'])
                    cbar.ax.minorticks_on()
                    cbar.ax.yaxis.set_minor_locator(MultipleLocator(2))

            elif self.is_pop():
                # POP grid without regridding
                __plt_kws = _plt_kws.copy()
                del(__plt_kws['cbar_kwargs'])
                if gs=='T':
                    lat_flat, lon_flat = da.TLAT.values.ravel(), da.TLONG.values.ravel()
                elif gs=='U':
                    lat_flat, lon_flat = da.ULAT.values.ravel(), da.ULONG.values.ravel()

                z_flat = da.values.ravel()
                valid_mask = ~np.isnan(z_flat)  # or just use mask_flat, since z_flat is from the same field
                lon_valid = lon_flat[valid_mask]
                lat_valid = lat_flat[valid_mask]
                z_valid = z_flat[valid_mask]
                im = ax.tricontourf(lon_valid, lat_valid, z_valid,  **__plt_kws)

                if latlon_range is None: ax.set_global()
                if 'add_colorbar' in kws and kws['add_colorbar'] is False:
                    pass
                else:
                    cbar = plt.colorbar(im, ax=ax, extend=_plt_kws['extend'], **_plt_kws['cbar_kwargs'])
                    cbar.ax.minorticks_on()
                    cbar.ax.yaxis.set_minor_locator(MultipleLocator(2))

            else:
                # regular lat-lon grid
                im = da.plot.contourf(ax=ax, **_plt_kws)

            if df_sites is not None:
                # plot scatter points for sites
                colname_dict = {} if colname_dict is None else colname_dict
                _colname_dict={'lat': 'lat', 'lon':'lon', 'value': 'value', 'type': 'type'}
                _colname_dict.update(colname_dict)
                site_lons = df_sites[_colname_dict['lon']].values if _colname_dict['lon'] in df_sites else None
                site_lats = df_sites[_colname_dict['lat']].values if _colname_dict['lat'] in df_sites else None
                site_vals = list(df_sites[_colname_dict['value']].values) if _colname_dict['value'] in df_sites else None
                site_types = list(df_sites[_colname_dict['type']].values) if _colname_dict['type'] in df_sites else ['default']*len(df_sites)

                _marker_dict = {
                    'default': 'o',
                }
                if site_marker_dict is not None:
                    _marker_dict.update(site_marker_dict)

                if site_vals is None:
                    site_colors = 'gray' if site_color_dict is None else [site_color_dict[t] for t in site_types]
                else:
                    site_colors = site_vals

                if site_marker_dict is None:
                    type_list = sorted(list(set(site_types)))
                else:
                    type_list = list(site_marker_dict)

                if isinstance(site_colors, str):
                    for site_type in type_list:
                        idx = [i for i, x in enumerate(site_types) if x == site_type]
                        ax.scatter(
                            site_lons[idx], site_lats[idx],
                            s=site_markersizes, marker=_marker_dict[site_type],
                            edgecolors='k', c=site_colors,
                            zorder=99, transform=_transform,
                        )
                elif isinstance(site_colors, list):
                    cmap_obj = plt.get_cmap(_plt_kws['cmap'])
                    norm = BoundaryNorm(im.levels, ncolors=cmap_obj.N, clip=True)
                    for site_type in type_list:
                        idx = [i for i, x in enumerate(site_types) if x == site_type]
                        ax.scatter(
                            site_lons[idx], site_lats[idx],
                            s=site_markersizes, marker=_marker_dict[site_type],
                            edgecolors='k', c=[site_colors[i] for i in idx], cmap=cmap_obj, norm=norm,
                            zorder=99, transform=_transform,
                        )

                if legend and len(list(set(site_types)))>=1:
                    for site_type in type_list:
                        if count_site_num:
                            n = site_types.count(site_type)
                            lb = f'{site_type} (n={n})'
                        else:
                            lb = f'{site_type}'

                        ax.scatter(
                            [], [], c='gray', marker=_marker_dict[site_type],
                            edgecolor='k', s=site_markersizes, label=lb,
                        )

                    lgd_kws = {} if lgd_kws is None else lgd_kws
                    _lgd_kws = {
                        'frameon': False,
                        'loc': 'lower left',
                        'bbox_to_anchor': (0.1, -0.2),
                        'columnspacing': 1.,
                        'handletextpad': 0.05,
                        'handleheight': 1.,
                        'ncol': 3,
                    }
                    _lgd_kws.update(lgd_kws)
                    ax.legend(**_lgd_kws)

        elif ndim == 2:
            # vertical
            if figsize is None: figsize = (6, 3)
            if ax is None: fig, ax = plt.subplots(figsize=figsize)
            _plt_kws = {
                'extend': 'both',
                'cmap': visual.infer_cmap(da),
                'cbar_kwargs': {
                    'label': f'{da.name} [{da.units}]',
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)
            # add color for missing data
            if bad_color is not None:
                ax.set_facecolor(bad_color)

            if 'add_colorbar' in kws and kws['add_colorbar'] is False:
                del(_plt_kws['cbar_kwargs'])

            im = da.plot.contourf(ax=ax, **_plt_kws)
            if add_clabels:
                # _contour_kws = {
                #     'levels': _plt_kws['levels'],
                #     'zorder': 99,
                #     'colors': 'k',
                # }
                # im = da.plot.contour(ax=ax, **_contour_kws)
                clabel_kwargs = {} if clabel_kwargs is None else clabel_kwargs
                _clabel_kwargs = {
                    'fontsize': 12,
                    'inline': True,
                    'colors': 'w',
                }
                _clabel_kwargs.update(clabel_kwargs)
                clbs = ax.clabel(im, clevels, zorder=99, **_clabel_kwargs)
                for txt in clbs:
                    txt.set_path_effects([plt.matplotlib.patheffects.Stroke(linewidth=2, foreground='black'), plt.matplotlib.patheffects.Normal()])

            if 'xlabel' not in kws:
                xlabel = ax.xaxis.get_label()
                if 'climo_period' in da.attrs:
                    ax.set_xlabel('Month')
                    ax.set_xticks(range(1, 13))
                    ax.set_xticklabels(range(1, 13))
                elif 'lat' in str(xlabel):
                    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
                    ax.set_xticklabels(['90°S', '60°S', '30°S', 'EQ', '30°N', '60°N', '90°N'])
                    ax.set_xlim([-90, 90])
                    ax.set_xlabel('Latitude')
            else:
                ax.set_xlabel(kws['xlabel'])

            if 'ylabel' not in kws:
                ylabel = ax.yaxis.get_label()
                if 'depth' in str(ylabel):
                    ax.invert_yaxis()
                    if 'z_t' in da.coords:
                        if da['z_t'].units == 'centimeters':
                            ax.set_yticks([0, 2e5, 4e5])
                        elif da['z_t'].units == 'km':
                            ax.set_yticks([0, 2, 4])
                    else:
                        ax.set_yticks([0, 2e5, 4e5])

                    ax.set_yticklabels([0, 2, 4])
                    ax.set_ylabel('Depth [km]')
            else:
                ax.set_ylabel(kws['ylabel'])

            ax.grid(False)

        else:
            # zonal mean, timeseries, others
            if figsize is None: figsize = (6, 3)
            if ax is None: fig, ax = plt.subplots(figsize=figsize)
            _plt_kws = {}
            _plt_kws = utils.update_dict(_plt_kws, kws)
            da.plot(ax=ax, **_plt_kws)

            if 'units' in da.attrs:
                ylabel = f'{da.name} [{da.units}]'
            else:
                ylabel = da.name

            ax.set_ylabel(ylabel)

        if title is None and 'long_name' in da.attrs:
            title = da.attrs['long_name']

        ax.set_title(title, weight='bold')

        if 'fig' in locals():
            return (fig, ax, im) if return_im else (fig, ax)
        else:
            return (ax, im) if return_im else ax