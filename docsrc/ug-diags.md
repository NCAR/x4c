---
title: CESM Diagnostics
---

`x4c.Timeseries` indexes a whole post-processed case and gives you variables by name,
including *derived* variables that are computed on demand. The `spell` notation
compresses a processing chain into a single string.

- [The `Timeseries` case object](notebooks/diags-overview.ipynb)
- [Loading variables](notebooks/diags-load_variables.ipynb) — `case.load(vn)`
- [The derived-variable registry](notebooks/diags-variables.ipynb) — variables that are computed rather than read
- [The spell mini-language](notebooks/diags-spell.ipynb) — a processing chain as one string
- [Adding a derived variable](notebooks/diags-new_vars.ipynb) — the `@F` decorator
- [`quickview`: a diagnostics dashboard in one call](notebooks/diags-quickview.ipynb)
- [Reading CESM logs](notebooks/diags-logs.ipynb) — `x4c.Logs`
