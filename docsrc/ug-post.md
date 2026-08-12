---
title: CESM Postprocessing
---

This section illustrates the postprocessing of CESM output using `x4c`.

- [History files to timeseries](notebooks/post-gen_ts.ipynb) — splitting and merging with `ncks`/`ncrcat` via `History.gen_ts()`
- [PBS scripts](notebooks/post-pbs.ipynb) — production Zsh scripts that generate and submit PBS jobs for `gen_ts()` on CESM1/2/3/CMIP7 cases
