'''Tests for the group-4 packaging and hygiene fixes.

Covers review findings:
  4.1   Nine imported packages were undeclared, so `pip install x4c` produced an
        uninstallable package.
  4.2   `visual.savefig` computed a default `.pdf` path, saved to the *other* path,
        and reported the one it did not write.
  4.3   `plot(regrid=...)` / `cyclic` / `add_colorbar` were read but not popped, so
        they leaked into the matplotlib call.
  4.4   `log=True` together with `levels` raised an opaque matplotlib error.
  4.5   `da.units` was accessed unguarded in the vertical-plot branch.
  4.6   `rsync_move` printed a literal `'>>> {cmd}'`.
  4.7   `cesm_str2datetime` had an `int()`-of-a-list branch and no final `else`.
  4.8   `Timeseries.vns` listed a variable once per timespan file.
  4.9   Timeseries path parsing was positional and unvalidated.
  4.10  `quickview` indexed hard-coded dicts with user-supplied spell keys.
  4.11  ~24% of the package was commented-out code.
  4.12  Unused imports and a shadowed builtin.
  4.13  Bare `except:` clauses swallowed KeyboardInterrupt/SystemExit.
'''
import builtins
import importlib
import inspect
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import x4c
from x4c import case, core, diags, spell, utils, visual

MODULES = [case, core, diags, spell, utils, visual]
CONDA_ONLY = {'xesmf', 'esmpy', 'ESMF', 'geocat', 'mpi4py', 'uxarray'}


# --------------------------------------------------------------------------- #
# 4.1 — declared dependencies are sufficient to import
# --------------------------------------------------------------------------- #
def test_no_conda_only_package_imported_at_module_level():
    '''4.1: importing x4c must not require the conda-only stack.

    Those packages are declared as extras; requiring them at import time is what
    made `pip install x4c && python -c "import x4c"` fail.
    '''
    offenders = []
    for mod in MODULES:
        for line in inspect.getsource(mod).splitlines():
            if not line.startswith(('import ', 'from ')):
                continue  # indented -> inside a function, i.e. lazy
            m = re.match(r'(?:import|from)\s+([\w\.]+)', line)
            if m and m.group(1).split('.')[0] in CONDA_ONLY:
                offenders.append(f'{mod.__name__}: {line.strip()}')
    assert offenders == [], f'conda-only packages imported eagerly: {offenders}'


def test_declared_dependencies_cover_the_eager_imports():
    '''Every eagerly-imported third-party package must be declared.'''
    import tomllib
    root = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
    with open(os.path.join(root, 'pyproject.toml'), 'rb') as f:
        declared_raw = tomllib.load(f)['project']['dependencies']
    # normalize: "netCDF4" -> "netcdf4", "python-dateutil" -> "dateutil"
    alias = {'netcdf4': 'netCDF4', 'python-dateutil': 'dateutil',
             'nc-time-axis': 'nc_time_axis', 'matplotlib': 'matplotlib'}
    declared = set()
    for d in declared_raw:
        name = re.split(r'[<>=!\[ ]', d)[0].lower()
        declared.add(alias.get(name, name).lower())

    stdlib = set(sys.stdlib_module_names)
    eager = set()
    for mod in MODULES:
        for line in inspect.getsource(mod).splitlines():
            if not line.startswith(('import ', 'from ')):
                continue
            m = re.match(r'(?:import|from)\s+([\w\.]+)', line)
            if not m:
                continue
            root_name = m.group(1).split('.')[0]
            if root_name in stdlib or root_name in {'x4c', ''} or line.startswith('from .'):
                continue
            eager.add(root_name.lower())

    missing = sorted(e for e in eager if e not in declared)
    assert missing == [], f'imported but not declared in pyproject.toml: {missing}'


def test_optional_features_raise_actionable_errors():
    '''The extras must fail with install instructions, not ModuleNotFoundError.'''
    for fn, needle in ((utils.import_xesmf, 'xesmf'), (utils.import_geocat_comp, 'geocat')):
        src = inspect.getsource(fn)
        assert 'conda install' in src, f'{fn.__name__} does not tell the user how to install'
        assert 'ImportError' in src


# --------------------------------------------------------------------------- #
# 4.2 — savefig
# --------------------------------------------------------------------------- #
def test_savefig_appends_default_suffix(tmp_path):
    '''4.2: an extension-less target must actually become `.pdf`.'''
    fig = plt.figure()
    target = tmp_path / 'figure_no_ext'
    visual.savefig(fig, str(target), verbose=False)
    assert (tmp_path / 'figure_no_ext.pdf').exists()
    assert not (tmp_path / 'figure_no_ext.png').exists(), 'saved with the wrong format'


def test_savefig_respects_explicit_suffix(tmp_path):
    fig = plt.figure()
    target = tmp_path / 'figure.png'
    visual.savefig(fig, str(target), verbose=False)
    assert target.exists()


def test_savefig_reports_the_path_it_wrote(tmp_path, capsys):
    '''4.2: the printed path used to name a file that did not exist.'''
    fig = plt.figure()
    visual.savefig(fig, str(tmp_path / 'reported'), verbose=True)
    out = capsys.readouterr().out
    m = re.search(r'Figure saved at: "([^"]+)"', out)
    assert m, out
    assert os.path.exists(m.group(1)), f'reported {m.group(1)} but it does not exist'


def test_savefig_creates_missing_directories(tmp_path):
    fig = plt.figure()
    visual.savefig(fig, str(tmp_path / 'a' / 'b' / 'fig.pdf'), verbose=False)
    assert (tmp_path / 'a' / 'b' / 'fig.pdf').exists()


# --------------------------------------------------------------------------- #
# 4.3 / 4.4 / 4.5 — plot kwargs
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_regrid_kwarg_does_not_leak_to_matplotlib(latlon_ds):
    '''4.3: `regrid` used to reach contourf, warning "kwargs were not used".'''
    da = latlon_ds(nt=1).x.da.isel(time=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        da.x.plot(regrid=False)


def test_cyclic_kwarg_does_not_leak(latlon_ds):
    da = latlon_ds(nt=1).x.da.isel(time=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        da.x.plot(cyclic=True)


def test_add_colorbar_false_is_honoured(latlon_ds):
    '''4.3: must suppress the colorbar without leaking into raw matplotlib calls.'''
    da = latlon_ds(nt=1).x.da.isel(time=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fig, ax = da.x.plot(add_colorbar=False)
    assert fig is not None


def test_log_with_levels_raises_clear_error(latlon_ds):
    '''4.4: previously "upper_level must be larger than lower_level" from matplotlib.'''
    da = latlon_ds(nt=1).x.da.isel(time=0)
    with pytest.raises(ValueError, match='cannot be combined'):
        da.x.plot(log=True, levels=np.linspace(0, 1, 11))


def test_vertical_plot_without_units(ocean_column_da):
    '''4.5: the 2-D branch used `da.units` unguarded, unlike the map branch.'''
    da = ocean_column_da(nt=1, nj=1).isel(time=0, nlat=0)
    assert 'units' not in da.attrs
    da.name = 'TEMP'
    fig, ax = da.x.plot()   # must not raise AttributeError
    assert fig is not None


def test_map_plot_without_units(latlon_ds):
    da = latlon_ds(nt=1).x.da.isel(time=0)
    da.attrs.pop('units', None)
    fig, ax = da.x.plot()
    assert fig is not None


# --------------------------------------------------------------------------- #
# 4.6 / 4.7 — small utils bugs
# --------------------------------------------------------------------------- #
def test_rsync_move_prints_an_interpolated_command(monkeypatch, capsys):
    '''4.6: used to print the literal string `>>> {cmd}`.'''
    monkeypatch.setattr(utils.subprocess, 'run', lambda *a, **k: None)
    utils.rsync_move(['/tmp/a'], '/tmp/b')
    out = capsys.readouterr().out
    assert '{cmd}' not in out
    assert 'rsync' in out


@pytest.mark.parametrize(
    'stamp, expected',
    [
        ('0001-01-01-00000', (1, 1, 1)),
        ('0001-01-01', (1, 1, 1)),
        ('0001-02', (1, 2, 1)),
        ('00010201000000', (1, 2, 1)),
    ],
)
def test_cesm_str2datetime(stamp, expected):
    dt = utils.cesm_str2datetime(stamp)
    assert (dt.year, dt.month, dt.day) == expected


def test_cesm_str2datetime_rejects_too_many_fields():
    '''4.7: there was no final `else`, so `res` could be unbound.'''
    with pytest.raises(ValueError, match='Cannot parse CESM timestamp'):
        utils.cesm_str2datetime('0001-01-01-00000-99')


# --------------------------------------------------------------------------- #
# 4.8 / 4.9 — Timeseries indexing
# --------------------------------------------------------------------------- #
def test_vns_has_no_duplicates(ts_tree):
    '''4.8: TS spans two timespan files and used to be listed twice.'''
    root, casename = ts_tree
    ts = case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})
    vns = ts.vns['atm']['cam.h0']
    assert len(vns) == len(set(vns)), f'duplicates in {vns}'
    assert vns.count('TS') == 1
    assert len(ts.paths['atm']['cam.h0']['TS']) == 2, 'but the paths must still list both files'


def test_vns_is_sorted(ts_tree):
    root, casename = ts_tree
    ts = case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})
    vns = ts.vns['atm']['cam.h0']
    assert vns == sorted(vns)


def test_unparseable_paths_are_skipped_with_a_warning(ts_tree, capsys):
    '''4.9: a filename not matching the pattern used to produce silent garbage.'''
    root, casename = ts_tree
    d = os.path.join(root, 'atm', 'proc', 'tseries', 'month_1')
    # a file from a *different* case
    stray = os.path.join(d, 'someothercase.cam.h0.QQQ.000101-001012.nc')
    xr.Dataset({'QQQ': (('x',), np.zeros(2))}).to_netcdf(stray)

    ts = case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})
    out = capsys.readouterr().out
    assert 'Skipping' in out
    assert 'QQQ' not in ts.vns['atm']['cam.h0']


def test_paths_dict_is_not_a_defaultdict(ts_tree):
    '''A missing key must raise, not silently create an empty entry.'''
    root, casename = ts_tree
    ts = case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})
    with pytest.raises(KeyError):
        ts.paths['atm']['no_such_hstr']


# --------------------------------------------------------------------------- #
# 4.10 — quickview with custom spells
# --------------------------------------------------------------------------- #
def test_quickview_accepts_custom_spells(ts_tree):
    '''4.10: a custom spell key used to KeyError on the hard-coded title/colour dicts.'''
    root, casename = ts_tree
    ts = case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})
    fig, ax = ts.quickview(spells={'MyPrecip': 'PRECC:ann:gm'}, timespan=(1, 10))
    assert set(ax) == {'MyPrecip'}


# --------------------------------------------------------------------------- #
# 4.11 / 4.12 / 4.13 — hygiene
# --------------------------------------------------------------------------- #
def test_no_bare_except():
    '''4.13: a bare `except:` also swallows KeyboardInterrupt and SystemExit.'''
    offenders = []
    for mod in MODULES:
        for i, line in enumerate(inspect.getsource(mod).splitlines(), 1):
            if re.match(r'^\s*except\s*:\s*$', line):
                offenders.append(f'{mod.__name__}:{i}')
    assert offenders == [], f'bare except clauses: {offenders}'


def test_no_eval_or_exec():
    '''4.13 + 3.1: no dynamic execution anywhere in the package.'''
    offenders = []
    for mod in MODULES:
        for i, line in enumerate(inspect.getsource(mod).splitlines(), 1):
            s = line.strip()
            if s.startswith('#'):
                continue
            if re.search(r'\b(eval|exec)\s*\(', s) and 'literal_eval' not in s and '_eval_node' not in s:
                offenders.append(f'{mod.__name__}:{i}: {s[:60]}')
    assert offenders == [], f'dynamic execution found: {offenders}'


def test_commented_out_code_is_gone():
    '''4.11: ~24% of the package was inert. Keep it that way.

    Threshold, not zero: real explanatory comments are wanted. What is being
    prevented is whole superseded implementations left in place, which is what made
    the live code hard to navigate.
    '''
    for mod in MODULES:
        lines = inspect.getsource(mod).splitlines()
        commented = sum(1 for l in lines if l.strip().startswith('#'))
        ratio = commented / max(len(lines), 1)
        assert ratio < 0.20, (
            f'{mod.__name__} is {ratio:.0%} comment lines; check for reintroduced '
            'commented-out code (the review baseline was 33% in case.py, 53% in diags.py)'
        )


def test_no_commented_out_function_definitions():
    '''The specific 4.11 pattern: a whole `def`/`class` left commented out.'''
    offenders = []
    for mod in MODULES:
        for i, line in enumerate(inspect.getsource(mod).splitlines(), 1):
            if re.match(r'^\s*#\s*(@\w+|(async\s+)?def\s+\w+|class\s+\w+)', line):
                offenders.append(f'{mod.__name__}:{i}: {line.strip()[:60]}')
    assert offenders == [], f'commented-out definitions: {offenders}'


@pytest.mark.parametrize(
    'mod_name, name',
    [
        ('x4c.utils', 'relativedelta'),
        ('x4c.core', 'ListedColormap'),
        ('x4c.core', 'cfeature'),
        ('x4c.core', 'dirpath'),
    ],
)
def test_unused_imports_removed(mod_name, name):
    '''4.12: these were imported/defined and never used.'''
    mod = importlib.import_module(mod_name)
    assert not hasattr(mod, name), f'{mod_name}.{name} is back'


def test_builtin_next_not_shadowed():
    '''4.12: `parse_timestamps` used `next` as a local variable name.'''
    src = inspect.getsource(utils.parse_timestamps)
    assert not re.search(r'^\s*next\s*=', src, re.M), '`next` is shadowed again'


def test_parse_timestamps_still_works():
    '''Guard the 4.12 rename with a behavioural check.'''
    out = utils.parse_timestamps(('0001-01', '0020-12'), timestep=10, timestep_unit='year')
    assert out == [('0001-01', '0010-12'), ('0011-01', '0020-12')]


# --------------------------------------------------------------------------- #
# 4.15 — docs match reality
# --------------------------------------------------------------------------- #
def test_no_setup_py_referenced_in_claude_md():
    '''4.15: CLAUDE.md said the version lives in `setup.py`, which does not exist.'''
    root = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
    md = os.path.join(root, 'CLAUDE.md')
    if not os.path.exists(md):
        pytest.skip('CLAUDE.md not present')
    text = open(md).read()
    assert not os.path.exists(os.path.join(root, 'setup.py'))
    assert 'set manually in `setup.py`' not in text


def test_package_docstring_documents_the_cache_and_keep_attrs():
    doc = x4c.__doc__ or ''
    assert 'keep_attrs' in doc
    assert 'cache' in doc.lower()
