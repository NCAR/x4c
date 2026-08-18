---
title: Core Features
---

`x4c` attaches a `.x` accessor to `xarray.Dataset` and `xarray.DataArray` that
understands CESM conventions — where the grid-cell areas live, how to find latitude
and longitude on an unstructured grid, and how the vertical coordinate is spaced.

Every notebook below runs against the reduced sample case bundled with these docs,
so they work from a fresh clone. See
[the sample case README](https://github.com/NCAR/x4c/tree/main/docsrc/notebooks) for
what was trimmed out of it. Each notebook is a single, self-contained example — small
on purpose, so a page never mixes unrelated features.

- [Overview](notebooks/core-overview.ipynb) — the `.x` accessor and what it attaches
- [Annualization and seasonalization](notebooks/core-annualization.ipynb) — `.x.annualize()`, including DJF wraparound
- [Seasonal maps](notebooks/core-annualization-seasonal.ipynb) — DJF vs JJA, composed with `.x.plot()`
- [Spatial means and climate indices](notebooks/core-geo_mean.ipynb) — area-weighted `.x.gm`, `.x.nhm`, `.x.shm`
- [Regridding: atmosphere](notebooks/core-regridding-atm.ipynb) — SE grid to regular lat/lon
- [Regridding: ocean](notebooks/core-regridding-ocean.ipynb) — POP grid to regular lat/lon
- [Hybrid levels to pressure levels](notebooks/core-plev.ipynb) — `.x.get_plev()`
- [Model coastlines for a paleo run](notebooks/core-visualization-coastlines.ipynb) — the `ssv` trick
- [Map projections at a glance](notebooks/core-visualization-projections.ipynb) — any Cartopy projection
- [Contour levels, colormaps, gridlines](notebooks/core-visualization-contour.ipynb)
- [Regional maps and saving figures](notebooks/core-visualization-regional.ipynb) — `latlon_range`, `x4c.savefig()`
- [Vertical sections](notebooks/core-visualization-vertical.ipynb) — the meridional overturning circulation
- [Zonal means](notebooks/core-visualization-zonal.ipynb) — a Hovmöller-style contour
- [Timeseries](notebooks/core-visualization-timeseries.ipynb)
- [Publication styles](notebooks/core-visualization-styles.ipynb) — `x4c.set_style()`
- [EOF analysis](notebooks/core-analysis-eof.ipynb) — grid predicates, `.x.eof()`
- [Extracting sites](notebooks/core-analysis-sites.ipynb) — `.x.nearest2d()`, `.x.nearest3d()`
- [Saving derived fields](notebooks/core-analysis-saving.ipynb) — `.x.to_netcdf()`
