---
title: Core Features
---

`x4c` attaches a `.x` accessor to `xarray.Dataset` and `xarray.DataArray` that
understands CESM conventions — where the grid-cell areas live, how to find latitude
and longitude on an unstructured grid, and how the vertical coordinate is spaced.

Every notebook below runs against the reduced sample case bundled with these docs,
so they work from a fresh clone. See
[the sample case README](https://github.com/NCAR/x4c/tree/main/docsrc/notebooks) for
what was trimmed out of it.

- [Overview](notebooks/core-overview.ipynb) — the `.x` accessor and what it attaches
- [Annualization and seasonalization](notebooks/core-annualization.ipynb) — `.x.annualize()`, including DJF wraparound
- [Spatial means and climate indices](notebooks/core-geo_mean.ipynb) — area-weighted `.x.gm`, `.x.nhm`, `.x.shm`
- [Regridding](notebooks/core-regridding.ipynb) — SE and POP grids onto regular lat/lon
- [Visualization](notebooks/core-visualization.ipynb) — `.x.plot()`, projections, and styles
- [Analysis: EOFs, site extraction, saving](notebooks/core-analysis.ipynb)
