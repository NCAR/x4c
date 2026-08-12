import xarray as xr

# NOTE: this is a PROCESS-WIDE side effect of importing x4c -- it changes xarray's
# behavior for all of the caller's code, not just x4c's.
#
# It is load-bearing rather than incidental: the whole accessor design carries the
# grid metadata (`gw`, `lat`, `lon`, `dz`) in `.attrs`, and with xarray's default
# `keep_attrs=False` those are dropped by ordinary arithmetic, so `da.x.gm` would
# stop working after something as simple as `da - 273.15`.
#
# Scoping it to individual operations (`with xr.set_options(keep_attrs=True):`)
# would be the polite fix, but the propagation happens across essentially every
# arithmetic and reduction path in the package plus user code in between, so a
# partial conversion would silently drop weights rather than fail loudly. Left
# global and documented deliberately; see review finding 3.7.
xr.set_options(keep_attrs=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator

import cartopy.crs as ccrs
from eofs.xarray import Eof

from . import utils, visual
import os

def load_dataset(path, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws):
    ''' Load a netCDF file and form a `xarray.Dataset`

    Args:
        path (str): path to the netCDF file
        shift_time (bool): shift the time of the `xarray.Dataset` (the CESM1 output has a time shift)
        comp (str): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    _kws = {'decode_times': xr.coders.CFDatetimeCoder(use_cftime=True), 'decode_timedelta': True}
    _kws.update(kws)
    ds = xr.load_dataset(path, **_kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, hstr=hstr, grid=grid, shift_time=shift_time)
    return ds

def open_dataset(path, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws):
    ''' Open a netCDF file and form a `xarray.Dataset` with a lazy load mode

    Args:
        path (str): path to the netCDF file
        shift_time (bool): shift the time of the `xarray.Dataset` (the CESM1 output has a time shift)
        comp (str): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    _kws = {'decode_times': xr.coders.CFDatetimeCoder(use_cftime=True), 'decode_timedelta': True}
    _kws.update(kws)
    ds = xr.open_dataset(path, **_kws)
    ds = utils.update_ds(ds, vn=vn, path=path, comp=comp, hstr=hstr, grid=grid, shift_time=shift_time)
    return ds

def open_mfdataset(paths, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws):
    ''' Open multiple netCDF files and form a `xarray.Dataset` in a lazy load mode

    Args:
        path (str): path to the netCDF file
        shift_time (bool): shift the time of the `xarray.Dataset` (the default CESM output has a time shift)
        comp (str): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"
        grid (str): the grid tag for the CESM output (e.g., ne16, g16)
        vn (str): variable name

    '''
    ds0 = xr.open_dataset(paths[0], decode_cf=False)
    dims_other_than_time = list(ds0.dims)
    if 'time' in dims_other_than_time:
        dims_other_than_time.remove('time')

    chunk_dict = {k: -1 for k in dims_other_than_time}

    _kws = {
        'data_vars': 'minimal',
        'coords': 'minimal',
        'compat': 'override',
        'chunks': chunk_dict,
        'parallel': True,
        'decode_times': xr.coders.CFDatetimeCoder(use_cftime=True),
        'decode_timedelta': True,
    }
    _kws.update(kws)
    ds = xr.open_mfdataset(paths, **_kws)
    ds = utils.update_ds(ds, vn=vn, path=paths, comp=comp, hstr=hstr, grid=grid, shift_time=shift_time)
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
        xe = utils.import_xesmf()

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

                    wgt_fpath = utils.fetch_wgt_file(f'map_{grid}_TO_{dlon}x{dlat}d_aave.nc.gz')
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
                # See utils.ensure_contiguous: xESMF's own internal reshaping over the
                # horizontal dims of a multi-dim Dataset commonly leaves the input
                # non-C-contiguous, which apply_weights then has to fix up on every
                # call ("Input array is not C_CONTIGUOUS. Will affect performance.").
                ds_rgd = regridder(self.ds.map(utils.ensure_contiguous), keep_attrs=True)

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

                # See utils.ensure_contiguous / the FV branch above.
                ds_rgd = regridder(self.ds.map(utils.ensure_contiguous), keep_attrs=True)

            else:
                raise ValueError(f'grid [{grid}] is not supported; please provide a corresponding `weight_file`.')

        # xESMF adds this CF grid-mapping stub; harmless but noisy
        ds_rgd = ds_rgd.drop_vars('latitude_longitude', errors='ignore')

        ds_rgd.attrs = dict(self.ds.attrs)
        # utils.p_success(f'Dataset regridded to regular grid: [dlon: {dlon} x dlat: {dlat}]')

        # The output lives on a regular lat/lon grid, so the source grid's `gw`/`lat`/`lon`
        # no longer describe it and must be re-derived. Carrying the source `gw` over would
        # leave a weight whose dims (`ncol`, `nlat`/`nlon`) are absent from the output, and
        # a subsequent weighted mean would then *broadcast* instead of reducing -- silently
        # turning a global mean into a full outer product.
        # Note the remapped `area`/`TAREA` variable is not a valid target-grid weight
        # either: interpolating a source area field does not give the destination cell
        # areas. Hence cos(lat), matching the convention in `utils.update_ds`.
        if 'lat' in ds_rgd.variables:
            ds_rgd.attrs['gw'] = utils.coslat_weight(ds_rgd)
        elif 'gw' in ds_rgd.attrs:
            # no target lat to weight by: drop it, since an absent weight fails loudly
            # while a stale one fails silently
            del(ds_rgd.attrs['gw'])

        for v in ['lat', 'lon']:
            if v in ds_rgd.variables:
                ds_rgd.attrs[v] = ds_rgd[v]
            elif v in ds_rgd.attrs:
                del(ds_rgd.attrs[v])

        return ds_rgd

    def get_plev(self, ps, vn=None, lev_mode='hybrid', **kws):
        """
        Interpolate a hybrid-level field to pressure levels and return a Dataset.

        This method converts a 3D atmospheric variable that is on hybrid model
        levels (a/k/a k-levels) into pressure levels using the provided surface
        pressure `ps` (either an `xarray.DataArray` or an `xarray.Dataset` that
        contains a variable named "PS"). It wraps
        `geocat.comp.interpolation.interp_hybrid_to_pressure` and returns a
        copy of the original `Dataset` with the requested variable replaced by
        its pressure-level version.

        Args:
            ps (xarray.DataArray or xarray.Dataset): surface pressure. If a
                `Dataset` is passed the method will look for the variable
                named "PS". Dimensions must align with the variable being
                interpolated.
            vn (str, optional): variable name in `self.ds` to interpolate. If
                not provided the method will use the dataset attribute
                `ds.attrs['vn']` and `self.da`.
            lev_mode (str, optional): currently only supports "hybrid".
                (Reserved for future expansion.)
            **kws: additional keyword arguments forwarded to
                `geocat.comp.interpolation.interp_hybrid_to_pressure`.
                By default `lev_dim` is set to `'lev'`. If the dataset
                contains `hyam`/`hybm` arrays they will be passed automatically.

        Returns:
            xarray.Dataset: a copy of `self.ds` with `vn` replaced by the
            pressure-level `DataArray` produced by the interpolation.

        Notes:
            - Requires `geocat.comp` to be available and the dataset to include
              the hybrid coefficients (`hyam`, `hybm`) when using hybrid
              vertical coordinates.
            - The returned dataset preserves the original dataset attributes
              and coordinate structure except that the specified variable is
              now on pressure levels.
        """

        # prepare keyword args for geocat function, default lev dim is 'lev'
        _kws = {'lev_dim': 'lev'}
        # if the dataset contains hybrid coefficients, pass them through
        if 'hyam' in self.ds: _kws['hyam'] = self.ds['hyam']
        if 'hybm' in self.ds: _kws['hybm'] = self.ds['hybm']

        _kws.update(kws)

        # select the variable to interpolate
        if vn is None:
            da = self.da
            vn = self.ds.attrs['vn']
        else:
            da = self.ds[vn]

        # accept either a Dataset containing 'PS' or a DataArray
        if isinstance(ps, xr.Dataset):
            ps_da = ps['PS']
        elif isinstance(ps, xr.DataArray):
            ps_da = ps
        else:
            raise TypeError('`ps` must be an xarray.DataArray or xarray.Dataset containing "PS"')

        # perform interpolation for the supported vertical-mode
        if lev_mode == 'hybrid':
            gc = utils.import_geocat_comp()
            da_plev = gc.interpolation.interp_hybrid_to_pressure(da, ps_da, **_kws)
        else:
            raise ValueError('`lev_mode` unknown')

        # return a dataset copy with the variable replaced by the pressure-level field
        ds_plev = self.ds.copy()
        del(ds_plev[vn])

        # On an unstructured grid `lat`/`lon` are *data variables* of the Dataset, but
        # `XDataset.__getitem__` promotes them to *coords* on the extracted DataArray.
        # Assigning that DataArray straight back in leaves xarray unable to decide
        # which they are ("unable to determine if these variables should be
        # coordinates or not"), so demote them first.
        conflicting = [c for c in da_plev.coords if c in ds_plev.data_vars]
        if conflicting:
            da_plev = da_plev.reset_coords(conflicting, drop=True)

        ds_plev[vn] = da_plev
        return ds_plev

    def zavg(self, depth_top, depth_bot, vn=None):
        '''
        Vertically average an ocean/column field between two depths and return a Dataset.

        The method selects the vertical range along the `z_t` coordinate from
        `depth_top` to `depth_bot`, applies area/volume weights provided by the
        dataset variable `dz`, computes the weighted mean over the vertical
        dimension, and returns a copy of the original `Dataset` with the
        specified variable replaced by its vertically averaged version.

        Args:
            depth_top (float): upper bound of the vertical slice (same units as `z_t`).
            depth_bot (float): lower bound of the vertical slice (same units as `z_t`).
            vn (str, optional): variable name in `self.ds` to average. If not
                provided the method will use the dataset attribute
                `ds.attrs['vn']` and `self.da`.

        Returns:
            xarray.Dataset: a copy of `self.ds` with `vn` replaced by the
            vertically averaged `DataArray`.

        Notes:
            - This method expects a vertical coordinate named `z_t` and a
              thickness/weight variable named `dz` in the dataset. The
              weighting is `dz` (e.g., layer thickness) and the mean is taken
              over the `z_t` dimension.
        '''

        # choose variable to operate on
        if vn is None:
            da = self.da
            vn = self.ds.attrs['vn']
        else:
            da = self.ds[vn]

        # select vertical slice and compute dz-weighted mean over z_t
        da_sel = da.sel(z_t=slice(depth_top, depth_bot))
        dz = self.ds['dz'].sel(z_t=slice(depth_top, depth_bot))
        da_zavg = da_sel.weighted(dz.fillna(0)).mean('z_t')

        # return a dataset copy with the variable replaced by its vertical average
        ds_zavg = self.ds.copy()
        ds_zavg[vn] = da_zavg
        # Convert the horizontal area weight into a *volume* weight (area x wet column
        # thickness) so a subsequent area-mean yields a true volume-weighted average.
        if 'gw' in self.ds.attrs:
            mask = da_sel.notnull()
            if 'time' in mask.dims:
                mask = mask.isel(time=0)
            col_thick = dz.where(mask).sum('z_t')
            ds_zavg.attrs['gw'] = self.ds.attrs['gw'] * col_thick
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
        '''
        Compute the climatology (monthly mean) of the dataset.

        This property groups the dataset by calendar month and computes the
        mean over the `time` dimension for each month. It also records the
        `climo_period` as a tuple (start_year, end_year) in the returned
        dataset's attributes and preserves `comp`/`grid` attributes when
        present. If the grouping result uses a `month` coordinate it is
        renamed to `time` to keep downstream interfaces consistent.

        Returns:
            xarray.Dataset: monthly climatology where the `time` coordinate
            indexes months (1-12). `ds.attrs['climo_period']` documents the
            original temporal coverage used to compute the climatology.
        '''

        # group by calendar month and compute mean over time
        ds = self.ds.groupby('time.month').mean(dim='time')

        # store the period used to compute climatology (start_year, end_year)
        ds.attrs['climo_period'] = (self.ds['time.year'].values[0], self.ds['time.year'].values[-1])

        # preserve useful dataset-level attributes
        if 'comp' in self.ds.attrs: ds.attrs['comp'] = self.ds.attrs['comp']
        if 'grid' in self.ds.attrs: ds.attrs['grid'] = self.ds.attrs['grid']

        # rename the month coordinate to `time` for a consistent API
        if 'month' in ds.coords:
            ds = ds.rename({'month': 'time'})

        return ds

    @property
    def anom(self):
        '''
        Compute monthly anomalies relative to the climatology.

        This property subtracts the monthly climatology (from
        `XDataset.climo`) from the dataset to produce anomalies for each
        time step. The climatology is aligned by month before subtraction so
        that, e.g., all Januaries are compared against the January climatology.

        Returns:
            xarray.Dataset: dataset of anomalies with the same coordinates as
            the original dataset.
        '''

        # subtract the monthly climatology (aligning months) to obtain anomalies
        ds = self.ds.groupby('time.month') - self.climo.rename({'time': 'month'})
        return ds

    def to_netcdf(self, path, **kws):
        ''' Write to netCDF, dropping the non-serializable x4c grid attrs

        The grid attrs (`gw`/`lat`/`lon`/`dz`) are stripped from a copy, so this
        `Dataset` keeps them and stays usable by the accessors afterwards.
        '''
        return utils.drop_grid_attrs(self.ds).to_netcdf(path, **kws)


@xr.register_dataarray_accessor('x')
class XDataArray:
    def __init__(self, da=None):
        self.da = da


    def annualize(self, months=None, days_weighted=False):
        ''' Annualize/seasonalize a `xarray.DataArray`

        Args:
            months (list of int): a list of integers to represent month combinations,
                e.g., [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

        '''
        da = utils.annualize(self.da, months=months, days_weighted=days_weighted)
        da = utils.update_attrs(da, self.da)
        return da

    def regrid(self, *args, **kws):
        '''
        Regrid this DataArray by delegating to the parent Dataset regrid.

        Positional arguments are forwarded too, so `da.x.regrid(1, 1)` works the same
        as `ds.x.regrid(1, 1)`. Keyword-only used to be the signature here, which made
        the documented spell form `|regrid(1,1)` fail on the DataArray path -- the one
        `Timeseries.calc` actually uses.

        This wraps `XDataset.regrid` by converting the `DataArray` to a
        temporary `Dataset`, calling the dataset-level regrid helper, then
        extracting and returning the regridded `DataArray`. Any dataset-level
        `lat`/`lon` attributes added during the transformation are removed from
        the returned `DataArray` attributes for cleanliness.

        **Forwarded kwargs** are the same as `XDataset.regrid` (e.g., `dlon`,
        `dlat`, `weight_file`, `gs`, `method`, `periodic`).
        '''

        # delegate to the Dataset regrid and extract the regridded DataArray
        ds_rgd = self.ds.x.regrid(*args, **kws)
        da = ds_rgd.x.da
        da.name = self.da.name

        # `gw`/`lat`/`lon` are deliberately kept: they were re-derived on the target grid
        # by `XDataset.regrid`, and the hemispheric means (`nhm`/`shm`/`nhs`/`shs`) read
        # `attrs['lat']`. Dropping them here would make this path diverge from
        # `ds.x.regrid().x.da`, which is meant to be equivalent.
        return da

    def get_plev(self, **kws):
        '''
        See: https://geocat-comp.readthedocs.io/en/v2024.04.0/user_api/generated/geocat.comp.interpolation.interp_hybrid_to_pressure.html
        '''
        _kws = {'lev_dim': 'lev'}
        _kws.update(kws)
        gc = utils.import_geocat_comp()
        da = gc.interpolation.interp_hybrid_to_pressure(self.da, **_kws)
        da.name = self.da.name
        return da

    def zavg(self, depth_top, depth_bot):
        # dz-weighted vertical average over [depth_top, depth_bot], per water column
        da = self.da.sel(z_t=slice(depth_top, depth_bot))
        dz = self.da.attrs['dz'].sel(z_t=slice(depth_top, depth_bot))
        da_zavg = da.weighted(dz.fillna(0)).mean('z_t')
        da_zavg.attrs = dict(self.da.attrs)
        # Convert the horizontal area weight into a *volume* weight (area x wet column
        # thickness within the depth range) so that a subsequent area-mean (e.g. .x.gm)
        # returns a true volume-weighted average rather than a dz-weighted mean of
        # per-level area means (the latter mis-weights levels where the ocean area
        # changes with depth). Requires zavg to run before the horizontal mean.
        if 'gw' in self.da.attrs:
            mask = da.notnull()
            if 'time' in mask.dims:
                mask = mask.isel(time=0)
            col_thick = dz.where(mask).sum('z_t')
            da_zavg.attrs['gw'] = self.da.attrs['gw'] * col_thick
        return da_zavg

    def to_netcdf(self, path, **kws):
        ''' Write to netCDF, dropping the non-serializable x4c grid attrs

        The grid attrs (`gw`/`lat`/`lon`/`dz`) are stripped from a copy, so this
        `DataArray` keeps them and stays usable by the accessors afterwards.
        '''
        return utils.drop_grid_attrs(self.da).to_netcdf(path, **kws)

    def nearest2d(self, lat=None, lon=None, lat_coord='lat', lon_coord='lon', lat_dim='lat', lon_dim='lon'):
        '''
        Select the nearest non-NaN grid point(s) for the given lat/lon targets.

        Given one or more target `lat`/`lon` pairs, this method finds the
        nearest valid (non-NaN across non-spatial dims) grid cell in the
        DataArray and returns a concatenated `DataArray` with a new dimension
        `site` indexing the selected points.

        Parameters:
            lat (float or array-like): target latitude(s).
            lon (float or array-like): target longitude(s).
            lat_coord (str): name of latitude coordinate in the DataArray.
            lon_coord (str): name of longitude coordinate in the DataArray.
            lat_dim (str): latitude dimension name.
            lon_dim (str): longitude dimension name.

        Returns:
            xarray.DataArray: concatenated selections at nearest grid points
            with a new `site` coordinate.
        '''
        lats = self.da.coords[lat_coord].values
        lons = self.da.coords[lon_coord].values
        if lats.ndim == 2 and lons.ndim == 2:
            lats2d = lats
            lons2d = lons
        elif lats.ndim == 1 and lons.ndim == 1:
            lons2d, lats2d = np.meshgrid(lons, lats)
        else:
            # previously fell through leaving lats2d/lons2d unbound, for an
            # `UnboundLocalError` a few lines later
            raise ValueError(
                f'`{lat_coord}` and `{lon_coord}` must both be 1-D or both 2-D; '
                f'got {lats.ndim}-D and {lons.ndim}-D.'
            )

        # mask grid cells that contain NaNs along the non-spatial dimensions.
        # `.transpose` matters: `.values` follows the DataArray's own dim order, so
        # without it a (time, lon, lat) array yields a mask whose axes are swapped
        # relative to lats2d/lons2d, and the (iy, ix) pair below indexes the wrong cell.
        reduce_dims = [d for d in self.da.dims if d not in (lat_dim, lon_dim)]
        valid = ~self.da.isnull().any(dim=reduce_dims) if reduce_dims else ~self.da.isnull()
        mask = valid.transpose(lat_dim, lon_dim).values

        if mask.shape != lats2d.shape:
            raise ValueError(
                f'coordinate grid {lats2d.shape} does not match the data grid '
                f'{mask.shape} over ({lat_dim}, {lon_dim}).'
            )
        if not mask.any():
            raise ValueError('No valid (non-NaN) grid cells to select from.')

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

    def nearest3d(self, lat=None, lon=None, depth=None,
                  lat_coord='lat', lon_coord='lon',
                  lat_dim='lat', lon_dim='lon',
                  depth_coord='z_t', depth_dim='z_t',
                  depth_unit='cm'):
        '''
        Select the nearest non-NaN 3D grid cell(s) for the given lat/lon/depth targets.

        Given one or more target `lat`/`lon`/`depth` triples, this method finds the
        nearest valid (non-NaN across non-spatial dims) grid cell in the full 3D
        (depth, lat, lon) domain of the DataArray and returns a concatenated
        `DataArray` with a new `site` dimension indexing the selected points.

        Distance is Euclidean in km, combining horizontal great-circle distance
        and absolute vertical |dz|. The `depth_coord` is converted from
        `depth_unit` to km, while target `depth` values are assumed to be in
        meters.

        Parameters:
            lat (float or array-like): target latitude(s).
            lon (float or array-like): target longitude(s).
            depth (float or array-like): target depth(s) in meters.
            lat_coord (str): name of latitude coordinate.
            lon_coord (str): name of longitude coordinate.
            lat_dim (str): latitude dimension name.
            lon_dim (str): longitude dimension name.
            depth_coord (str): name of vertical coordinate.
            depth_dim (str): vertical dimension name.
            depth_unit (str): unit of `depth_coord` ('cm' for CESM POP, 'm', or 'km').

        Returns:
            xarray.DataArray: concatenated selections at nearest 3D grid points
            with a new `site` coordinate.
        '''
        da = self.da

        # 2D lat/lon grids, kept as DataArrays so distance ops broadcast cleanly
        lats = da.coords[lat_coord]
        lons = da.coords[lon_coord]
        if lats.ndim == 1 and lons.ndim == 1:
            lon2d_np, lat2d_np = np.meshgrid(lons.values, lats.values)
            lat2d = xr.DataArray(lat2d_np, dims=(lat_dim, lon_dim))
            lon2d = xr.DataArray(lon2d_np, dims=(lat_dim, lon_dim))
        else:
            lat2d = lats
            lon2d = lons

        # vertical coordinate in km
        unit_scale = {'cm': 1e-5, 'm': 1e-3, 'km': 1.0}
        if depth_unit not in unit_scale:
            raise ValueError(f"depth_unit must be one of {list(unit_scale)}, got {depth_unit!r}")
        z_km = da.coords[depth_coord] * unit_scale[depth_unit]

        # mask cells with NaN across non-spatial dims
        reduce_dims = [d for d in da.dims if d not in (depth_dim, lat_dim, lon_dim)]
        mask = ~da.isnull().any(dim=reduce_dims) if reduce_dims else ~da.isnull()

        target_lat = np.atleast_1d(lat)
        target_lon = np.atleast_1d(lon)
        target_depth = np.atleast_1d(depth)

        sel_list = []
        for lat0, lon0, depth0 in zip(target_lat, target_lon, target_depth):
            h = utils.gcd(lat0, lon0, lat2d, lon2d)        # (lat_dim, lon_dim), km
            v = np.abs(z_km - depth0 * 1e-3)               # (depth_dim,), km
            dist = np.sqrt(h ** 2 + v ** 2).where(mask)    # (depth, lat, lon)
            idx = dist.argmin(dim=[depth_dim, lat_dim, lon_dim])
            # cast to plain ints: `argmin` hands back 0-d DataArrays, and indexing a
            # float32 dimension coordinate (POP's `z_t`) with one of those trips an
            # assertion inside pandas' Index constructor. `nearest2d` already passes
            # ints for the same reason.
            idx = {k: int(v) for k, v in idx.items()}
            sel_list.append(da.isel(idx))

        return xr.concat(sel_list, dim='site').assign_coords(site=np.arange(len(sel_list)))

    def eof(self, n=4, weight=True):
        ''' Perform EOF analysis

        Args:
            n (int): number of modes to return
            weight (bool): weight the field by sqrt(cos(lat)) before solving, so
                that the modes are area-fair. Requires a `lat` coordinate or attr.
        '''
        if weight:
            if 'lat' in self.da.coords:
                coslat = np.cos(np.deg2rad(self.da.coords['lat']))
            elif 'lat' in self.da.attrs:
                coslat = np.cos(np.deg2rad(self.da.attrs['lat']))
            else:
                # previously left `coslat` unbound, for a `NameError` on the next line
                raise ValueError(
                    'EOF weighting needs a latitude: none found in `da.coords["lat"]` '
                    'or `da.attrs["lat"]`. Pass `weight=False` for an unweighted solve.'
                )

            if 'time' not in self.da.dims:
                raise ValueError(f'EOF analysis needs a `time` dimension; got {self.da.dims}.')

            wgts = np.sqrt(coslat).broadcast_like(self.da.isel(time=0))  # (lat, lon)
            solver = Eof(self.da, weights=wgts)
        else:
            solver = Eof(self.da)

        pcs  = solver.pcs(npcs=n, pcscaling=1)       # standardized PCs
        eofs = solver.eofs(neofs=n, eofscaling=2)     # scaled EOFs
        var  = solver.varianceFraction(neigs=n)
        return pcs, eofs, var

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

    def _spatial_dims(self, gw):
        ''' The dims to reduce over for an area-weighted reduction

        Reduce over exactly the dims the weight spans (horizontal); this keeps the
        vertical (e.g. `z_t`) intact.

        The weight lives in `.attrs`, where xarray cannot validate it against the
        data, so any operation that changes the horizontal grid can leave a stale
        weight attached. A stale weight whose dims are absent from the data would
        *broadcast* rather than reduce -- silently turning a global mean into a full
        outer product -- so refuse it explicitly instead.
        '''
        missing = [d for d in gw.dims if d not in self.da.dims]
        if len(gw.dims) == 0 or len(missing) > 0:
            raise ValueError(
                f'The area weight `gw` spans dims {tuple(gw.dims)}, which are not all '
                f'present in the data dims {tuple(self.da.dims)}'
                + (f' (missing: {missing})' if missing else '')
                + '. The weight is stale or mismatched, so a weighted reduction would '
                'broadcast instead of reduce. Re-attach a weight for the current grid '
                '(e.g. via `x4c.utils.update_ds`).'
            )

        return [d for d in self.da.dims if d in gw.dims]

    @property
    def gm(self):
        ''' the global area-weighted mean '''
        gw = self.da.attrs['gw']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.weighted(gw).mean(spatial_dims)
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Global Mean {da.attrs["long_name"]}'
        return da

    @property
    def nhm(self):
        ''' the NH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.where(lat>0).weighted(gw).mean(spatial_dims)
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'NH Mean {da.attrs["long_name"]}'
        return da

    @property
    def shm(self):
        ''' the SH area-weighted mean '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.where(lat<0).weighted(gw).mean(spatial_dims)
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'SH Mean {da.attrs["long_name"]}'
        return da

    @property
    def gs(self):
        ''' the global area-weighted sum '''
        gw = self.da.attrs['gw']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.weighted(gw).sum(spatial_dims)
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'Global Sum {da.attrs["long_name"]}'
        return da

    @property
    def nhs(self):
        ''' the NH area-weighted sum '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.where(lat>0).weighted(gw).sum(spatial_dims)
        da = utils.update_attrs(da, self.da)
        if 'long_name' in da.attrs: da.attrs['long_name'] = f'NH Sum {da.attrs["long_name"]}'
        return da

    @property
    def shs(self):
        ''' the SH area-weighted sum '''
        gw = self.da.attrs['gw']
        lat = self.da.attrs['lat']
        spatial_dims = self._spatial_dims(gw)
        da = self.da.where(lat<0).weighted(gw).sum(spatial_dims)
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
        if 'month' in da.coords: da = da.rename({'month': 'time'})
        return da

    @property
    def anom(self):
        da = self.da.groupby('time.month') - self.climo.rename({'time': 'month'})
        return da

    def geo_mean(self, ind=None, latlon_range=(-90, 90, 0, 360), **kws):
        '''
        Calculate the geospatial-weighted (latitude or area) mean over a specified region or climate index.
        Parameters
        ----------
        ind : str, optional
            Climate index name. Supported indices include:
            - 'nino3.4': Niño 3.4 region
            - 'nino1+2': Niño 1+2 region
            - 'nino3': Niño 3 region
            - 'nino4': Niño 4 region
            - 'wpi': Western Pacific Index
            - 'tpi': Tri-Pole Index
            - 'dmi': Dipole Mode Index (Indian Ocean)
            - 'iobw': Indian Ocean Basin-Wide Index
            If None, uses latlon_range instead. Default is None.
        latlon_range : tuple or list, optional
            Latitude and longitude range for computing the mean in the format
            (lat_min, lat_max, lon_min, lon_max). Default is (-90, 90, 0, 360).
        **kws : dict
            Additional keyword arguments passed to utils.geo_mean().
        Returns
        -------
        xarray.DataArray
            Latitude-weighted mean values over the specified region or index.
            Attributes from the original data are preserved. Time coordinate
            long_name is updated to 'Model Year' if applicable.
        Raises
        ------
        ValueError
            If ind is not one of the supported climate index names.
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
        return ('lat' in da.dims and 'lon' in da.dims)

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

        # x4c-only options must be POPPED, not merely read: `update_dict` below copies
        # everything left in `kws` into the matplotlib call, where an unknown kwarg
        # either warns ("kwargs were not used by contour") or raises.
        do_regrid = kws.pop('regrid', False)
        add_colorbar = kws.pop('add_colorbar', True)
        cyclic = kws.pop('cyclic', False)
        if do_regrid:
            da = da.x.regrid(gs=gs)

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
            if not add_colorbar:
                _plt_kws.pop('cbar_kwargs', None)

            if latlon_range is not None:
                lat_min, lat_max, lon_min, lon_max = latlon_range
                ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=_transform)

            if add_gridlines:
                gl = ax.gridlines(linestyle=gridline_style, draw_labels=gridline_labels)
                gl.top_labels = False
                gl.right_labels = False

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

            if log:
                if 'levels' in _plt_kws:
                    raise ValueError(
                        '`log=True` and `levels` cannot be combined: a LogNorm and '
                        'explicit contour levels fight over the same scale (matplotlib '
                        'then raises "upper_level must be larger than lower_level"). '
                        'Pass log-spaced `levels` instead, e.g. np.logspace(0, 3, 28).'
                    )
                _plt_kws.update({'norm': LogNorm(vmin=vmin, vmax=vmax)})

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
                    # pop, not del: `cbar_kwargs` is already gone when the caller
                    # passed add_colorbar=False, and this branch used to KeyError
                    __plt_kws.pop('cbar_kwargs', None)
                    del(__plt_kws['cmap'])
                    im = ax.tricontourf(da.lon, da.lat, da, cmap=cmap, norm=norm, **__plt_kws)
                else:
                    # using UXarray for CAM-SE grid
                    try:
                        import uxarray as ux
                    except ImportError as e:
                        # `except ImportError`, not a bare `except`: a bare one also
                        # swallowed failures from *inside* a successfully-found uxarray
                        # and reported them as "not installed"
                        raise ImportError(
                            'UXarray is required for this method. Please install it via '
                            '`conda install -c conda-forge uxarray`.'
                        ) from e

                    grid = da.attrs['grid']
                    wgt_fpath = utils.fetch_wgt_file(f'scrip_{grid}.nc.gz')
                    uxgrid = ux.open_grid(wgt_fpath)
                    uxda = ux.UxDataArray(da, uxgrid=uxgrid).rename({'ncol': 'n_face'})

                    # take the projection off the axes rather than the local built above:
                    # a caller-supplied `ax` (e.g. from `visual.subplots`) never binds that
                    # local, and this path would then raise `NameError`
                    if not hasattr(ax, 'projection'):
                        raise ValueError('`ux=True` requires a Cartopy GeoAxes; got a plain `matplotlib` axes.')
                    pc = uxda.to_polycollection(projection=ax.projection)
                    pc.set_cmap(cmap)
                    pc.set_norm(norm)
                    pc.set_antialiased(False)
                    im = ax.add_collection(pc)

                if latlon_range is None: ax.set_global()
                if add_colorbar:
                    cbar = plt.colorbar(im, ax=ax, extend=_plt_kws['extend'], **_plt_kws['cbar_kwargs'])
                    cbar.ax.minorticks_on()
                    cbar.ax.yaxis.set_minor_locator(MultipleLocator(2))

            elif self.is_pop():
                # POP grid without regridding
                __plt_kws = _plt_kws.copy()
                __plt_kws.pop('cbar_kwargs', None)
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
                if add_colorbar:
                    cbar = plt.colorbar(im, ax=ax, extend=_plt_kws['extend'], **_plt_kws['cbar_kwargs'])
                    cbar.ax.minorticks_on()
                    cbar.ax.yaxis.set_minor_locator(MultipleLocator(2))

            else:
                # regular lat-lon grid
                if cyclic:
                    da_original = da.copy()
                    da = utils.add_cyclic_point(da_original)
                    da.name = da_original.name
                    da.attrs = da_original.attrs

                im = da.plot.contourf(ax=ax, add_colorbar=add_colorbar, **_plt_kws)

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
                    'label': f'{da.name} [{da.units}]' if 'units' in da.attrs else f'{da.name}',
                    'aspect': 10,
                },
            }
            _plt_kws = utils.update_dict(_plt_kws, kws)
            # add color for missing data
            if bad_color is not None:
                ax.set_facecolor(bad_color)

            if not add_colorbar:
                _plt_kws.pop('cbar_kwargs', None)

            im = da.plot.contourf(ax=ax, add_colorbar=add_colorbar, **_plt_kws)
            if add_clabels:
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
                if 'climo_period' in da.attrs and ('time' in da.dims or 'month' in da.dims):
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