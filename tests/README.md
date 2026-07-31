# x4c test suite

Regression tests for the defects found in the 2026-07-28 code review
(`.claude/code-review-2026-07-28.md`). Every test names the finding it guards in
its docstring.

Everything is synthetic — no CESM output, no network, no MPI ranks. The whole
suite runs in about 15 seconds.

## Running

```bash
pytest                    # everything
pytest tests/test_weights.py -v
pytest -k annualize
```

Most of the suite runs on the PyPI-installable dependencies alone, since the
conda-only packages are now lazily imported (review §4.1). The regrid tests in
`test_weights.py` do need `xesmf`/`esmpy`; `ci/environment.yml` installs the full
stack so everything runs.

If you run from a venv layered on a conda env rather than the activated conda env
itself, set `ESMFMKFILE` by hand — `esmpy` resolves it from `sys.prefix`, which
the venv changes:

```bash
export ESMFMKFILE=$CONDA_PREFIX/lib/esmf.mk
```

## Coverage

| File | Findings covered |
|---|---|
| `test_annualize.py` | 1.1 empty `days_weighted` result; 1.2 calendar-year weight normalization |
| `test_weights.py` | 1.3 stale `gw` after `regrid`; 2.7 `nhm`/`shm` KeyError after `regrid` |
| `test_io.py` | 1.4 `to_netcdf` mutating the live object and variable-level attrs |
| `test_diags.py` | 1.5 `NINO3.4` mislabelled as kelvin |
| `test_update_ds.py` | 2.2 `comp='rof'` KeyError |
| `test_case.py` | 2.1 dead `calc` reuse branch; 2.3 `comps_info` string; 2.4 ignored `avoid_list`; 2.5 broken `get_ts`/`save_means`; 2.6 `get_paths` returning `None` |
| `test_plot.py` | 2.8 `NameError` in the UXarray branch |
| `test_spell.py` | 3.1 `eval()` on spell strings; 3.2 cryptic parse failures |
| `test_robustness.py` | 3.3–3.14 (shell-out, download, cache dir, lazy MPI, `keep_attrs` docs, `geo_mean`, `Logs`, `nearest2d`, `eof`, `gen_ts`, `avoid_list`) |
| `test_hygiene.py` | 4.1–4.15 (declared deps, `savefig`, plot kwargs, `vns` duplicates, path validation, `quickview`, dead code, unused imports, bare excepts, docs) |

Validated against the pre-fix code (commit `0e68ae0`) in a detached worktree. A
regression test that passes both before and after its fix is worthless, so re-run
that comparison if you refactor these:

| suite | pre-fix `0e68ae0` | fixed |
|---|---|---|
| groups 1 + 2 | 81 failed, 60 passed, 6 errors | 147 passed |
| groups 1–3 | 112 failed, 69 passed, 7 errors | 241 passed |
| groups 1–4 | 136 failed, 81 passed, 7 errors | **277 passed** |

Per-module failures at HEAD: 31/36 in `test_robustness.py` (group 3) and 24 in
`test_hygiene.py` (group 4) — at least one per finding. Tests that pass at HEAD
assert behaviour that was already correct, so they are guards rather than
regression proofs.

## Conventions

- Hand-computed expectations, not golden values recorded from the
  implementation. `conftest.weighted_mean` and `conftest.coslat_mean` are
  independent reference implementations.
- The `ramp_series` fixture makes value equal time index, and the `ts_tree`
  fixture writes value equal calendar month, so seasonal means are exact
  integers (ANN 6.5, MAM 4.0, JJA 7.0, SON 10.0). That is what catches a season
  defined with the wrong months.
- A few tests deliberately assert on source text (`test_no_diagnostic_labelled_kelvin`,
  `test_ux_consumer_reads_ax_projection`) where the runtime path needs an absent
  optional dependency or where the trap is a label rather than a behaviour. They
  say so in their docstrings.

## Known gaps

- **`ux=True` is not exercised end-to-end.** `uxarray` is not in the environment,
  and the `ImportError` guard fires before the fixed line, so `test_plot.py`
  checks the property the fix relies on plus the source. The original `NameError`
  was reachable only with `uxarray` installed, for the same reason.
- **The SE (`ne*`) regrid path is not covered.** It needs a multi-MB ESMF weight
  file that `x4c` downloads on demand; the tests stay offline and use the FV and
  POP paths, which need only `xesmf`'s `grid_global`.
- **No coverage of `History.bigbang`/`gen_ts`.** Those shell out to `ncks`/`ncrcat`
  under MPI. Worth a separate, marked, opt-in suite.
- **`test_spell.py` cannot be validated against `0e68ae0`.** It imports
  `find_call`/`parse_call_args`, which the old `spell.py` does not define, so
  collection aborts there. Pre-fix evidence for findings 3.1/3.2 is instead: the
  old `case.py` contained 4 `= eval(` call sites (now 0), and
  `Spell('TS:ann:gm:extra')` used to raise
  `TypeError: argument of type 'NoneType' is not iterable`.
- **§3.7 (`keep_attrs`) is asserted, not fixed.**
  `test_keep_attrs_is_set_and_documented` pins that the global is still set and
  that both the call site and the package docstring explain why. Converting it to
  scoped context managers is deliberately out of scope — see that entry in the
  review.
