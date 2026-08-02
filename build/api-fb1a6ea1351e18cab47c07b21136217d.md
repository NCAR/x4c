# API Reference

## Core Features

### `load_dataset`

`load_dataset(path, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws)`

Load a netCDF file and form a `xarray.Dataset`

**Parameters**

- `path` (`str`): path to the netCDF file

- `shift_time` (`bool`): shift the time of the `xarray.Dataset` (the CESM1 output has a time shift)

- `comp` (`str`): the tag for CESM component, including "atm", "ocn", "lnd", "ice", and "rof"

- `grid` (`str`): the grid tag for the CESM output (e.g., ne16, g16)

- `vn` (`str`): variable name

---

### `open_dataset`

`open_dataset(path, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws)`

Open a netCDF file and form a `xarray.Dataset` with a lazy load mode

**Parameters**

- `path` (`str`): path to the netCDF file

- `shift_time` (`bool`): shift the time of the `xarray.Dataset` (the CESM1 output has a time shift)

- `comp` (`str`): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"

- `grid` (`str`): the grid tag for the CESM output (e.g., ne16, g16)

- `vn` (`str`): variable name

---

### `open_mfdataset`

`open_mfdataset(paths, shift_time=False, comp=None, hstr=None, grid=None, vn=None, **kws)`

Open multiple netCDF files and form a `xarray.Dataset` in a lazy load mode

**Parameters**

- `path` (`str`): path to the netCDF file

- `shift_time` (`bool`): shift the time of the `xarray.Dataset` (the default CESM output has a time shift)

- `comp` (`str`): the tag for general CESM components, including "atm", "ocn", "lnd", "ice", and "rof"

- `grid` (`str`): the grid tag for the CESM output (e.g., ne16, g16)

- `vn` (`str`): variable name

---

### `XDataset`

#### • `regrid`

`regrid(dlon=1, dlat=1, weight_file=None, gs='T', method='bilinear', periodic=True)`

Regrid the CESM output to a normal lat/lon grid

Supported atmosphere regridding: ne16np4, ne16pg3, ne30np4, ne30pg3, ne120np4, ne120pg4 TO 1x1d / 2x2d.
Supported ocean regridding: any grid similar to g16 TO 1x1d / 2x2d.
For any other regridding, `weight_file` must be provided by the user.

For the atmosphere grid regridding, the default method is area-weighted;
while for the ocean grid, the default is bilinear.

**Parameters**

- `dlon` (`float`): longitude spacing

- `dlat` (`float`): latitude spacing

- `weight_file` (`str`): the path to an ESMF-generated weighting file for regridding

- `gs` (`str`): grid style in 'T' or 'U' for the ocean grid

- `method` (`str`): regridding method for the ocean grid

- `periodic` (`bool`): the assumption of the periodicity of the data when perform the regrid method

---

#### • `get_plev`

`get_plev(ps, vn=None, lev_mode='hybrid', **kws)`

Interpolate a hybrid-level field to pressure levels and return a Dataset.

This method converts a 3D atmospheric variable that is on hybrid model
levels (a/k/a k-levels) into pressure levels using the provided surface
pressure `ps` (either an `xarray.DataArray` or an `xarray.Dataset` that
contains a variable named "PS"). It wraps
`geocat.comp.interpolation.interp_hybrid_to_pressure` and returns a
copy of the original `Dataset` with the requested variable replaced by
its pressure-level version.

**Parameters**

- `ps` (`xarray.DataArray or xarray.Dataset`): surface pressure. If a
  `Dataset` is passed the method will look for the variable
  named "PS". Dimensions must align with the variable being
  interpolated.

- `vn` (`str`): variable name in `self.ds` to interpolate. If
  not provided the method will use the dataset attribute
  `ds.attrs['vn']` and `self.da`.

- `lev_mode` (`str`): currently only supports "hybrid".
  (Reserved for future expansion.)

- `**kws`: additional keyword arguments forwarded to
  `geocat.comp.interpolation.interp_hybrid_to_pressure`.
  By default `lev_dim` is set to `'lev'`. If the dataset
  contains `hyam`/`hybm` arrays they will be passed automatically.

**Returns**

- xarray.Dataset: a copy of `self.ds` with `vn` replaced by the pressure-level `DataArray` produced by the interpolation.

**Notes**

- Requires `geocat.comp` to be available and the dataset to include
  the hybrid coefficients (`hyam`, `hybm`) when using hybrid
  vertical coordinates.
- The returned dataset preserves the original dataset attributes
  and coordinate structure except that the specified variable is
  now on pressure levels.

---

#### • `zavg`

`zavg(depth_top, depth_bot, vn=None)`

Vertically average an ocean/column field between two depths and return a Dataset.

The method selects the vertical range along the `z_t` coordinate from
`depth_top` to `depth_bot`, applies area/volume weights provided by the
dataset variable `dz`, computes the weighted mean over the vertical
dimension, and returns a copy of the original `Dataset` with the
specified variable replaced by its vertically averaged version.

**Parameters**

- `depth_top` (`float`): upper bound of the vertical slice (same units as `z_t`).

- `depth_bot` (`float`): lower bound of the vertical slice (same units as `z_t`).

- `vn` (`str`): variable name in `self.ds` to average. If not
  provided the method will use the dataset attribute
  `ds.attrs['vn']` and `self.da`.

**Returns**

- xarray.Dataset: a copy of `self.ds` with `vn` replaced by the vertically averaged `DataArray`.

**Notes**

- This method expects a vertical coordinate named `z_t` and a
  thickness/weight variable named `dz` in the dataset. The
  weighting is `dz` (e.g., layer thickness) and the mean is taken
  over the `z_t` dimension.

---

#### • `annualize`

`annualize(months=None, days_weighted=False, time2year=False)`

Annualize/seasonalize a `xarray.Dataset`

**Parameters**

- `months` (`list of int`): a list of integers to represent month combinations,
  e.g., `None` means calendar year annualization, [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

---

#### • `to_netcdf`

`to_netcdf(path, **kws)`

Write to netCDF, dropping the non-serializable x4c grid attrs

The grid attrs (`gw`/`lat`/`lon`/`dz`) are stripped from a copy, so this
`Dataset` keeps them and stays usable by the accessors afterwards.

---

### `XDataArray`

#### • `annualize`

`annualize(months=None, days_weighted=False)`

Annualize/seasonalize a `xarray.DataArray`

**Parameters**

- `months` (`list of int`): a list of integers to represent month combinations,
  e.g., [7,8,9] means JJA annualization, and [-12,1,2] means DJF annualization

---

#### • `regrid`

`regrid(*args, **kws)`

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

---

#### • `get_plev`

`get_plev(**kws)`

See: https://geocat-comp.readthedocs.io/en/v2024.04.0/user_api/generated/geocat.comp.interpolation.interp_hybrid_to_pressure.html

---

#### • `to_netcdf`

`to_netcdf(path, **kws)`

Write to netCDF, dropping the non-serializable x4c grid attrs

The grid attrs (`gw`/`lat`/`lon`/`dz`) are stripped from a copy, so this
`DataArray` keeps them and stays usable by the accessors afterwards.

---

#### • `nearest2d`

`nearest2d(lat=None, lon=None, lat_coord='lat', lon_coord='lon', lat_dim='lat', lon_dim='lon')`

Select the nearest non-NaN grid point(s) for the given lat/lon targets.

Given one or more target `lat`/`lon` pairs, this method finds the
nearest valid (non-NaN across non-spatial dims) grid cell in the
DataArray and returns a concatenated `DataArray` with a new dimension
`site` indexing the selected points.

**Parameters**

- `lat` (`float or array - like`): target latitude(s).

- `lon` (`float or array - like`): target longitude(s).

- `lat_coord` (`str`): name of latitude coordinate in the DataArray.

- `lon_coord` (`str`): name of longitude coordinate in the DataArray.

- `lat_dim` (`str`): latitude dimension name.

- `lon_dim` (`str`): longitude dimension name.

**Returns**

- xarray.DataArray: concatenated selections at nearest grid points with a new `site` coordinate.

---

#### • `nearest3d`

`nearest3d(lat=None, lon=None, depth=None, lat_coord='lat', lon_coord='lon', lat_dim='lat', lon_dim='lon', depth_coord='z_t', depth_dim='z_t', depth_unit='cm')`

Select the nearest non-NaN 3D grid cell(s) for the given lat/lon/depth targets.

Given one or more target `lat`/`lon`/`depth` triples, this method finds the
nearest valid (non-NaN across non-spatial dims) grid cell in the full 3D
(depth, lat, lon) domain of the DataArray and returns a concatenated
`DataArray` with a new `site` dimension indexing the selected points.

Distance is Euclidean in km, combining horizontal great-circle distance
and absolute vertical |dz|. The `depth_coord` is converted from
`depth_unit` to km, while target `depth` values are assumed to be in
meters.

**Parameters**

- `lat` (`float or array - like`): target latitude(s).

- `lon` (`float or array - like`): target longitude(s).

- `depth` (`float or array - like`): target depth(s) in meters.

- `lat_coord` (`str`): name of latitude coordinate.

- `lon_coord` (`str`): name of longitude coordinate.

- `lat_dim` (`str`): latitude dimension name.

- `lon_dim` (`str`): longitude dimension name.

- `depth_coord` (`str`): name of vertical coordinate.

- `depth_dim` (`str`): vertical dimension name.

- `depth_unit` (`str`): unit of `depth_coord` ('cm' for CESM POP, 'm', or 'km').

**Returns**

- xarray.DataArray: concatenated selections at nearest 3D grid points with a new `site` coordinate.

---

#### • `eof`

`eof(n=4, weight=True)`

Perform EOF analysis

**Parameters**

- `n` (`int`): number of modes to return

- `weight` (`bool`): weight the field by sqrt(cos(lat)) before solving, so
  that the modes are area-fair. Requires a `lat` coordinate or attr.

---

#### • `geo_mean`

`geo_mean(ind=None, latlon_range=(-90, 90, 0, 360), **kws)`

Calculate the geospatial-weighted (latitude or area) mean over a specified region or climate index.

**Parameters**

- `ind` (`str`): Climate index name. Supported indices include:
  - 'nino3.4': Niño 3.4 region
  - 'nino1+2': Niño 1+2 region
  - 'nino3': Niño 3 region
  - 'nino4': Niño 4 region
  - 'wpi': Western Pacific Index
  - 'tpi': Tri-Pole Index
  - 'dmi': Dipole Mode Index (Indian Ocean)
  - 'iobw': Indian Ocean Basin-Wide Index
  If None, uses latlon_range instead. Default is None.

- `latlon_range` (`tuple or list`): Latitude and longitude range for computing the mean in the format
  (lat_min, lat_max, lon_min, lon_max). Default is (-90, 90, 0, 360).

- `**kws` (`dict`): Additional keyword arguments passed to utils.geo_mean().

**Returns**

- `xarray.DataArray`: Latitude-weighted mean values over the specified region or index.
  Attributes from the original data are preserved. Time coordinate
  long_name is updated to 'Model Year' if applicable.

**Raises**

- `ValueError`: If ind is not one of the supported climate index names.

---

#### • `plot`

`plot(title=None, figsize=None, ax=None, latlon_range=None, add_clabels=False, clevels=None, clabel_kwargs=None, projection='Robinson', transform='PlateCarree', central_longitude=180, proj_args=None, bad_color='dimgray', add_gridlines=False, gridline_labels=True, gridline_style='--', ssv=None, log=False, vmin=None, vmax=None, coastline_zorder=99, coastline_width=1, site_markersizes=100, df_sites=None, colname_dict=None, gs='T', ux=False, site_marker_dict=None, site_color_dict=None, count_site_num=False, lgd_kws=None, legend=True, return_im=False, **kws)`

The plotting functionality

**Parameters**

- `title` (`str`): figure title

- `figsize` (`tuple or list`): figure size in format of (w, h)

- `ax` (``matplotlib.axes``): a `matplotlib.axes`

- `latlon_range` (`tuple or list`): lat/lon range in format of (lat_min, lat_max, lon_min, lon_max)

- `projection` (`str`): a projection name supported by `Cartopy`

- `transform` (`str`): a projection name supported by `Cartopy`

- `central_longitude` (`float`): the central longitude of the map to plot

- `proj_args` (`dict`): other keyword arguments for projection

- `add_gridlines` (`bool`): if True, the map will be added with gridlines

- `gridline_labels` (`bool`): if True, the lat/lon ticklabels will appear

- `gridline_style` (`str`): the gridline style, e.g., '-', '--'

- `ssv` (``xarray.DataArray``): a sea surface variable used for plotting the coastlines

- `gs` (`str`): grid style in 'T' or 'U' for the ocean grid

- `coastline_zorder` (`int`): the layer order for the coastlines

- `coastline_width` (`float`): the width of the coastlines

- `df_sites` (``pandas.DataFrame``): a `pandas.DataFrame` that stores the information of a collection of sites

- `colname_dict` (`dict`): a dictionary of column names for `df_sites` in the "key:value" format "assumed name:real name"

---

## CESM Postprocessing

### `History`

Handle CESM history files for a single case.

Provides utilities to discover history file paths, list time-series
variables, split (isolate) variables into separate files, and
re-merge them across time ranges. Designed to work with NCO tools
and MPI for parallel operations.

#### • `get_ts_vns`

`get_ts_vns(comp, hstr, exclude_vars=[...])`

Return list of time-varying variable names for a given component
and hstr by inspecting the first history file.

---

#### • `get_paths`

`get_paths(comp, hstr, timespan=None)`

Return history file paths for a component/hstr optionally
filtered by a timespan.

timespan may be provided in a variety of formats accepted by
utils.parse_timespan.

---

#### • `isolate_vn`

`isolate_vn(vn, comp, hstr, in_path, output_dirpath, overwrite=True)`

Create a new netCDF file containing only variable `vn` from
the input history file `in_path`.

Uses `ncks` to drop other variables and writes result to
`output_dirpath` with a standardized filename.

---

#### • `bigbang`

`bigbang(comp, hstr, output_dirpath, timespan=None, overwrite=True, nproc=1, vns=None)`

Split history files into per-variable files in parallel using MPI.

Each MPI rank handles a subset of (file,variable) tasks.

---

#### • `get_hstr_based_on_vn`

`get_hstr_based_on_vn(vn)`

Return the first hstr that contains variable `vn`.

This searches across all components and hstrs and returns the
matching hstr string or None if not found.

---

#### • `merge_vn`

`merge_vn(hstr, vn, input_dirpath, output_dirpath, timespan=None, overwrite=True, compression=1)`

Concatenate per-variable files across time into a single file.

Uses `ncrcat` with optional compression level to produce an
aggregated timeseries file for `vn` and `hstr`.

---

#### • `bigcrunch`

`bigcrunch(comp, hstr, input_dirpath, output_dirpath, timespan=None, overwrite=True, nproc=1, compression=1, vns=None)`

Merge per-variable files back into timeseries files in parallel.

Coordinates work across MPI ranks similar to `bigbang`.

---

#### • `gen_ts`

`gen_ts(output_dirpath, staging_dirpath=None, comps=['atm', 'ocn', 'lnd', 'ice', 'rof'], timespan=None, timestep=None, timestep_unit='year', dir_structure='comp/proc/tseries/hstr', overwrite=True, nproc=1, compression=1)`

Generate timeseries files for selected components and timespans.

This orchestrates splitting (`bigbang`) and merging
(`bigcrunch`) stages and moves results from staging to final
output directories.

---

#### • `find_timespan_files`

`find_timespan_files(timespan, comps=['atm', 'ice', 'ocn', 'rof', 'lnd'])`

List the history files within a timespan, without touching them

Resolves the glob in Python rather than through a shell, so no part of
`root_dir` is ever interpreted as shell syntax.

**Parameters**

- `timespan` (`tuple or list`): [start_year, end_year], inclusive, integers

- `comps` (`list`): components to search

**Returns**

- list of str: matching paths, sorted

---

#### • `rm_timespan`

`rm_timespan(timespan, comps=['atm', 'ice', 'ocn', 'rof', 'lnd'], nworkers=None, rehearsal=True)`

Delete the archived history files within a timespan

This is the one destructive operation in x4c, so it is deliberately
conservative:

- `rehearsal=True` (the default) only *reports* what would be deleted.
- The file list is resolved with `glob` and removed with `os.remove`, rather
  than interpolated into a `rm -f ... shell=True` command. A `root_dir`
  containing a space or a shell metacharacter used to change which files were
  deleted.
- The exact list is printed before anything is removed.

**Parameters**

- `timespan` (`tuple or list`): [start_year, end_year], inclusive, integers

- `comps` (`list`): components to clean

- `nworkers` (`int`): parallel workers for the deletion (default: 8)

- `rehearsal` (`bool`): if True, only list the files; nothing is deleted

**Returns**

- list of str: the paths that were (or would be) removed

---

## CESM Diagnostics

### `Timeseries`

CESM Timeseries case helper.

Manages discovery and loading of preprocessed CESM timeseries files
produced by CESM postprocessing. Provides convenience methods to
locate paths, load raw or derived diagnostics, compute spells, and
create plots and seasonal means.

#### • `get_paths`

`get_paths(comp, hstr, vn, timespan=None)`

Return list of timeseries file paths for `vn` under `comp/hstr`.

If `timespan` is provided it filters the returned paths to those
fully covering the requested interval.

---

#### • `get_comp_hstr`

`get_comp_hstr(vn)`

Find all (component, hstr) pairs where `vn` is present.

---

#### • `load`

`load(vn, vtype=None, comp=None, hstr=None, timespan=None, load_idx=-1, verbose=True, reload=False, **kws)`

Load a variable or derived diagnostic into `self.ds`.

Automatically detects whether `vn` is a raw timeseries or a
derived diagnostic and loads or computes it. Results are stored
in `self.ds[vn]`.

---

#### • `calc`

`calc(spell, comp=None, timespan=None, load_idx=-1, recalculate=False, verbose=True, **kws)`

Compute a diagnostic spell and cache the result.

The `spell` string controls regridding, slicing, spatial/vertical
averaging and other modifiers parsed by `Spell`. The final
xarray DataArray is stored in `self.diags[spell]`.

---

#### • `plot`

`plot(spell, t_idx=None, regrid=False, gs='T', ssv='SSH', recalculate_ssv=False, timespan=None, **kws)`

Plot a computed diagnostic `spell`.

Detects plot type (map, ts, zm, yz) from the DataArray and
dispatches to the plotting helpers in `diags`/`visual`.

---

#### • `quickview`

`quickview(timespan=None, nrow=None, ncol=None, wspace=0.3, hspace=0.5, ax_loc=None, figsize=None, stat_period=-50, roll_int=50, ylim_dict=None, spells=None, recalculate=False)`

Create a multi-panel overview figure for a selection of spells.

Returns `(fig, ax)` where `ax` is a dict of axes keyed by spell
keys.

---

#### • `get_ts`

`get_ts(vn, comp=None, hstr=None, timespan=None, slicing=False, regrid=False, dlat=1, dlon=1)`

Open and return a Dataset for `vn`, without caching it in `self.ds`.

Applies optional slicing and regridding before returning the Dataset.

**Parameters**

- `vn` (`str`): variable name

- `comp` (`str`): component; inferred from `vn` when it is unambiguous

- `hstr` (`str`): history-stream tag; inferred from `vn` when it is unambiguous

- `timespan` (`tuple`): (start, end), as either ints or 'YYYY-MM'-style strings

- `slicing` (`bool`): additionally `.sel` the time axis to `timespan`

- `regrid` (`bool`): regrid to a regular `dlat` x `dlon` grid

---

#### • `save_means`

`save_means(vn, comp=None, output_dirpath=None, timespan=None, hstr=None, slicing=False, regrid=False, dlat=1, dlon=1, overwrite=False)`

Save seasonal and annual mean files for `vn` into `output_dirpath`.

Writes files for ANN, DJF, MAM, JJA and SON for the given
`timespan` and optionally regrids results.

---

#### • `clear_ds`

`clear_ds(vn=None)`

Clear the existing `.ds` property

---

#### • `copy`

`copy()`

Return a deep copy of this Timeseries instance.

---

### `Logs`

Manage CESM log files for a case and extract time series variables.

This helper locates compressed component log files (e.g. `ocn.log.*.gz`),
parses monthly reported diagnostics, and provides plotting helpers for
inspected variables.

#### • `get_vars`

`get_vars(vn=[...])`

Parse log files and extract listed variables into a DataFrame.

**Parameters**

- `vn` (`list or str`): variables to extract from logs. If a string,
  it will be converted to a single-element list.

**Side-effects**

Sets `self.df`, `self.df_ann` and `self.vn` with parsed results.

---

#### • `plot_vars`

`plot_vars(vn=None, annualize=True, xlim=None, ylim_dict=None, unit_dict=None, clr_dict=None, figsize=[20, 5], ncol=4, nrow=None, wspace=0.5, hspace=0.5, kws=None, title=None)`

Plot one or more variables parsed from the logs.

**Parameters**

- `vn` (`list or str`): variables to plot; defaults to the parsed set.

- `annualize` (`bool`): plot annual means when True.

- `xlim, ylim_dict, unit_dict, clr_dict`: plotting customizations.

- `figsize, ncol, nrow`: layout options.

**Returns**

- (fig, ax) Matplotlib figure and dict of axes keyed by variable name.

---

#### • `compare_vars`

`compare_vars(L_ref, vn=None, annualize=True, xlim=None, unit_dict=None, clr_dict=None, figsize=[20, 5], ncol=4, nrow=None, wspace=0.3, hspace=0.5, kws=None, title=None)`

Overlay variables from a reference `Logs` instance for comparison.

**Parameters**

- `L_ref` (`Logs`): reference Logs instance whose variables will be
  plotted over the current instance's plots in black.

---

## The Spell Mini-Language

### `Spell`

The Spell System

A "spell" is a string that summarizes a series of data processing steps.
A basic sentence should be in the form: "vn:ann_method:sa_method", where

- `vn`: a variable name
- `ann_method`: annualization method
- `sa_method`: spatial average method

One may also add more operations after the `vn` part:

- "|plev": to interpolate the data from the model z levels to the pressure levels
- "|regrid": to regrid the data from the model grid to the regular lat/lon grid
- "|zavg": to vertically average an ocean field over a depth range
- ".isel(...)" / ".sel(...)": to slice the variable before anything else

An optional "alias ~ " prefix renames the result.

Examples::

    'TS:ann:gm'
    'GMSST ~ SST:ann:gm'
    'TEMP.isel(z_t=0):ann:gm'
    'T|regrid(1,1)|plev(500):climo'
    'TEMP|zavg(0,1000):ann:gm'
    'SST.sel(lat=slice(-5,5)):ann'

Arguments are parsed into structured form (`regrid_args`, `slicing_kwargs`,
`plev_levels`, ...) so callers never have to `eval` the string. The raw
fragments remain available as `slicing`, `regrid`, `plev` and `zavg` for
display. Note `:` is the field separator and cannot appear inside arguments.

**Attributes**

- `sentence` (`str`): the spell with any alias stripped

- `alias` (`str`): the requested output name, or None

- `vn` (`str`): the bare variable name, with any slicing call removed

- `vn_raw` (`str`): the variable fragment as written, slicing included

- `ann_method` (`str`): annualization method, or None

- `sa_method` (`str`): spatial-average method, or None

- `slicing` (`str`): e.g. ``'isel(z_t=0)'``, or None

- `slicing_method` (`str`): ``'isel'`` or ``'sel'``, or None

- `plev` (`str`): e.g. ``'plev(500)'``, or None

- `plev_levels` (`list`): requested pressure levels, or None for a bare ``|plev``

#### • `parse_sentence`

`parse_sentence()`

Split the ``vn[:ann_method[:sa_method]]`` skeleton

---

#### • `parse_slicing`

`parse_slicing()`

Extract a leading ``.isel(...)`` / ``.sel(...)`` on the variable

---

## Derived-Variable Registry

### `F`

`F(func=None, name=None)`

Decorator to register a diagnostic function, with optional custom key.

---

### `Registry`

#### • `get_F`

`get_F(cls, name)`

Retrieve a diagnostic function by name.

---

## Visualization Helpers

### `set_style`

`set_style(style='journal', font_scale=1.0)`

Modify the visualization style

This function is inspired by [Seaborn](https://github.com/mwaskom/seaborn).
See a demo in the example_notebooks folder on GitHub to look at the different styles

**Parameters**

- `style` (`(journal, web, nature, agu, presentation, dark, minimal, matplotlib, _spines, _nospines, _grid, _nogrid)`): set the styles for the figure:
      - journal (default): fonts appropriate for paper
      - web: web-like font (e.g. ggplot)
      - nature: clean style inspired by Nature/Science figures with Helvetica-like fonts
      - agu: style following AGU journal conventions with minor ticks and enclosed axes
      - presentation: bold, high-contrast style for talks and posters
      - dark: modern dark theme for screen display and dashboards
      - minimal: ultra-clean style with thin lines and open layout
      - matplotlib: the original matplotlib style
      In addition, the following options are available:
      - _spines/_nospines: allow to show/hide spines
      - _grid/_nogrid: allow to show gridlines (default: _grid)

- `font_scale` (`float`): Default is 1. Corresponding to 12 Font Size.

---

### `subplots`

`subplots(nrow, ncol, ax_loc, projs=None, projs_kws=None, figsize=None, wspace=None, hspace=None, annotation=False, annotation_kws=None, annotation_separate=False, annotation_skip=None)`

Create a named grid of axes, with optional Cartopy projections

A thin wrapper over `matplotlib.gridspec.GridSpec` that returns the axes in a
dict keyed by name instead of an array, so that a subplot can be referred to as
`ax['ts']` rather than by position. Any subset of the axes can be given a Cartopy
projection, which makes mixed layouts (maps next to timeseries) straightforward.

**Parameters**

- `nrow` (`int`): number of rows in the grid

- `ncol` (`int`): number of columns in the grid

- `ax_loc` (`dict`): maps an axes name to its slot in the grid, e.g.
  `{'map': (0, slice(0, 2)), 'ts': (1, 0)}`. Each value is anything
  `GridSpec.__getitem__` accepts -- an int, a `(row, col)` tuple, or tuples
  containing `slice` objects for axes spanning several cells.

- `projs` (`dict`): maps an axes name to a `cartopy.crs` class name, e.g.
  `{'map': 'Robinson'}`. Names absent from this dict get a plain
  (non-geographic) axes. `None` means no projections at all.

- `projs_kws` (`dict`): maps an axes name to the keyword arguments passed to its
  projection class, e.g. `{'map': {'central_longitude': 180}}`

- `figsize` (`tuple`): figure size in inches, passed to `matplotlib.pyplot.figure`

- `wspace` (`float`): width of the padding between subplots, in units of the average
  axes width; passed to `GridSpec.update`

- `hspace` (`float`): height of the padding between subplots, in units of the average
  axes height; passed to `GridSpec.update`

- `annotation` (`bool`): if True, label the axes with (a), (b), ... via
  `add_annotation`

- `annotation_kws` (`dict`): keyword arguments for `add_annotation`. When
  `annotation_separate` is False, this is a single dict applied to all axes.
  When it is True, this must be a dict of dicts keyed by axes name, e.g.
  `{'map': {'loc_x': -0.1}, 'ts': {}}` -- every labeled axes needs an entry.
  `style` defaults to `')'` in both cases.

- `annotation_separate` (`bool`): if True, call `add_annotation` once per axes so
  that each label can be positioned individually; if False, one call labels
  them all together

- `annotation_skip` (`list`): axes names to leave unlabeled; only honored when
  `annotation_separate` is True. Note that the letters still advance over the
  skipped axes, since they are assigned by position in `ax_loc`.

**Returns**

- (fig, ax) the Matplotlib figure and a dict of axes keyed by the names in `ax_loc`.

---

### `savefig`

`savefig(fig, path, verbose=True, **kws)`

Save a figure to a path

**Parameters**

- `fig` (`matplotlib.pyplot.figure`): the figure to save

- `path` (`str`): the path to save the figure, can be ignored and specify in "settings" instead

- `settings` (`dict`): the dictionary of arguments for plt.savefig(); some notes below:
  - "path" must be specified in settings if not assigned with the keyword argument;
    it can be any existed or non-existed path, with or without a suffix;
    if the suffix is not given in "path", it will follow "format"
  - "format" can be one of {"pdf", "eps", "png", "ps"}

---

### `showfig`

`showfig(fig, close=True)`

Show the figure

**Parameters**

- `fig` (`matplotlib.pyplot.figure`): The matplotlib figure object

- `close` (`bool`): if True, close the figure automatically

---

### `closefig`

`closefig(fig=None)`

Show the figure

**Parameters**

- `fig` (`matplotlib.pyplot.figure`): The matplotlib figure object

---

### `add_annotation`

`add_annotation(ax, fs=20, loc_x=-0.15, loc_y=1.03, start=0, style=None)`

Label axes with (a), (b), ... in order

**Parameters**

- `ax`: a single axes, a dict of axes (as returned by `subplots`), or any
  list/tuple/array of axes -- including the 2-D array `plt.subplots` returns

- `fs` (`float or list`): font size, per-axes if a list

- `loc_x, loc_y` (`float`): label position in axes coordinates

- `start` (`int`): index into the alphabet to start from

- `style` (`str`): None for a bare letter, `')'` for `a)`, `'()'` for `(a)`

---

### `infer_cmap`

`infer_cmap(da)`

Guess a sensible colormap from a DataArray's `long_name`

Matches keywords against `da.attrs['long_name']`, lowercased, and returns the
colormap for the first keyword that matches -- so the order below is a precedence
order, e.g. "sea ice temperature" resolves to the temperature map, not the ice one.
Diverging maps are used for fields that are naturally read about a center value
(temperature, pressure, precipitation, correlation) and sequential ones otherwise.

============= ==========
keyword       colormap
============= ==========
temperature   `RdBu_r`
pressure      `bwr_r`
precipitation `BrBG`
correlation   `RdBu_r`
r2            `Reds`
salinity      `PiYG`
circulation   `RdBu_r`
depth         `GnBu`
height        `PiYG`
kmt           `BrBG`
ice           `Blues`
============= ==========

**Parameters**

- `da` (`xarray.DataArray`): the field to pick a colormap for; only its
  `long_name` attribute is inspected, never the values

**Returns**

- a Matplotlib colormap name, or `'viridis'` if `long_name` is absent or matches no keyword

---

## Utilities

### `fetch_sample_data`

`fetch_sample_data(case=DEFAULT_SAMPLE_CASE, url=None, sha256=None, verbose=True, force=False)`

Return a local path to a tutorial sample case, downloading it if needed

The tutorial notebooks run against a reduced copy of a real CESM case. It lives in
the cache directory (see `cache_dir`), not in the repository.

Resolution order:

1. ``$X4C_SAMPLE_DIR``, if set and it contains the case -- use this to point at a
   copy you already have, e.g. on a shared filesystem.
2. the cache directory, if the case is already extracted there
3. otherwise download the archive and extract it

**Parameters**

- `case` (`str`): which sample case to fetch; a key of `SAMPLE_DATA`

- `url` (`str`): override the download location. Defaults to
  `sample_data_url`, or ``$X4C_SAMPLE_URL`` if that is set.

- `sha256` (`str`): expected checksum of the archive. Defaults to the case's
  registered checksum; `None` skips verification.

- `verbose` (`bool`): report what is being used or fetched

- `force` (`bool`): re-download even if the case is already present

**Returns**

- path to the case directory, ready to hand to `x4c.Timeseries`

[(<DocstringSectionKind.examples: 'examples'>, ">>> import x4c\n>>> case_dir = x4c.fetch_sample_data(case='cesm1')")]

---

### `cache_dir`

`cache_dir()`

The directory x4c downloads regrid weight files into

Resolution order:

1. ``$X4C_CACHE_DIR``, if set
2. ``$XDG_CACHE_HOME/x4c``, if set
3. ``~/.cache/x4c``

Deliberately *not* the installed package directory: that fails outright on a
read-only or shared ``site-packages``, and in an editable install it drops
multi-MB binaries into the source tree.

