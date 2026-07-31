'''Regression tests for `History` and `Timeseries` against a synthetic CESM tree.

Covers review findings:
  2.1  `Timeseries.calc`'s "already calculated" branch assigned `da` and then
       overwrote it one line later (the assignment was one dedent short), so the
       reuse path was dead and could `KeyError` on `self.ds[vn]`.
  2.3  A `comps_info` value given as a plain string was iterated character by
       character, so `{'atm': 'h0'}` looked for hstrs `'h'` and `'0'`.
  2.4  `History` discovered hstrs with `self.avoid_list` but then built the
       per-hstr path lists with the bare constructor argument, letting `once`
       files back in.
  2.5  `Timeseries.get_ts` called `get_paths` with the wrong signature and passed
       a long-removed `adjust_month` kwarg, so it raised `TypeError` on every
       call; `save_means` did the same and additionally defined MAM as JFM.
  2.6  `get_paths` returned `None` for an unknown variable while its caller
       called `len()` on the result.
'''
import glob
import os

import numpy as np
import pytest
import xarray as xr

from x4c import case


@pytest.fixture
def ts(ts_tree):
    root, casename = ts_tree
    return case.Timeseries(root, casename=casename, grid_dict={'atm': 'fv1x1'})


# --------------------------------------------------------------------------- #
# 2.6
# --------------------------------------------------------------------------- #
def test_get_paths_returns_empty_list_for_unknown_variable(ts):
    '''2.6: `[]`, not `None` — the caller does `len(paths)`.'''
    assert ts.get_paths('atm', 'cam.h0', 'NO_SUCH_VAR') == []


def test_load_unknown_variable_raises_valueerror(ts):
    '''2.6: the intended error, not a TypeError from `len(None)`.'''
    with pytest.raises(ValueError, match='unknown'):
        ts.load('NO_SUCH_VAR')


def test_get_paths_finds_both_timespan_files(ts):
    '''Sanity on the fixture: TS is split across two decades.'''
    assert len(ts.get_paths('atm', 'cam.h0', 'TS')) == 2


def test_get_paths_timespan_filter(ts):
    '''Only files fully inside the requested window are returned.'''
    paths = ts.get_paths('atm', 'cam.h0', 'TS', timespan=('0001-01', '0010-12'))
    assert len(paths) == 1 and '000101-001012' in paths[0]


# --------------------------------------------------------------------------- #
# 2.2 end-to-end
# --------------------------------------------------------------------------- #
def test_rof_variable_loads(ts):
    '''2.2: loading a river-runoff timeseries used to raise KeyError.'''
    ts.load('QRUNOFF', verbose=False)
    assert 'gw' in ts.ds['QRUNOFF'].attrs
    assert ts.ds['QRUNOFF'].attrs['comp'] == 'rof'


# --------------------------------------------------------------------------- #
# 2.5 — get_ts
# --------------------------------------------------------------------------- #
def test_get_ts_is_callable(ts):
    '''2.5: this raised TypeError on every call.'''
    ds = ts.get_ts('TS')
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes['time'] == 240  # 20 years of monthly data


def test_get_ts_infers_comp_and_hstr(ts):
    '''An unambiguous variable should not require comp/hstr.'''
    ds = ts.get_ts('PRECC')
    assert ds.attrs['comp'] == 'atm'
    assert ds.attrs['hstr'] == 'cam.h0'


def test_get_ts_attaches_grid_attrs(ts):
    '''Going through `open_mfdataset` is what attaches `gw`; setting `.attrs`
    by hand afterwards would leave the accessors without a weight.'''
    ds = ts.get_ts('TS')
    assert 'gw' in ds.attrs
    assert np.isfinite(float(ds.x.da.x.gm[0]))


def test_get_ts_timespan_selects_files(ts):
    '''An int timespan is converted to the string form internally.'''
    assert ts.get_ts('TS', timespan=(1, 10)).sizes['time'] == 120


def test_get_ts_timespan_requires_full_containment(ts):
    '''`get_paths` only returns files *entirely* inside the window.

    Asking for years 1-5 when the only file spans 1-10 legitimately matches
    nothing. Documented here because it is easy to read `timespan` as "overlap".
    '''
    assert ts.get_paths('atm', 'cam.h0', 'TS', timespan=('0001', '0005')) == []
    with pytest.raises(ValueError, match='No timeseries files found'):
        ts.get_ts('TS', timespan=(1, 5))


def test_get_ts_slicing(ts):
    '''`slicing=True` runs and preserves the data.

    Note it is effectively a no-op given the containment rule above: any file
    `get_paths` returns is already inside the window, so there is nothing left to
    trim. Kept as a smoke test of the code path.
    '''
    ds = ts.get_ts('TS', timespan=(1, 10), slicing=True)
    assert ds.sizes['time'] == 120
    assert np.isfinite(ds['TS'].values).all()


def test_get_ts_slicing_without_timespan_raises(ts):
    '''Slicing to `None` would be a TypeError deep in xarray; fail clearly.'''
    with pytest.raises(ValueError, match='timespan'):
        ts.get_ts('TS', slicing=True)


def test_get_ts_unknown_variable_raises(ts):
    with pytest.raises(ValueError, match='unknown'):
        ts.get_ts('NO_SUCH_VAR')


def test_get_ts_does_not_cache(ts):
    '''`get_ts` is documented as not touching `self.ds`.'''
    ts.get_ts('PRECC')
    assert 'PRECC' not in ts.ds


# --------------------------------------------------------------------------- #
# 2.5 — save_means
# --------------------------------------------------------------------------- #
@pytest.fixture
def means_dir(ts, tmp_path):
    out = tmp_path / 'means'
    ts.save_means('PRECC', output_dirpath=str(out), timespan=(1, 10))
    return out


def test_save_means_writes_every_season(means_dir):
    '''2.5: the function is callable at all, and produces all five files.'''
    for sn in ('ANN', 'DJF', 'MAM', 'JJA', 'SON'):
        assert glob.glob(os.path.join(str(means_dir), sn, '*.nc')), f'{sn} missing'


@pytest.mark.parametrize(
    'season, expected',
    [('ANN', 6.5), ('MAM', 4.0), ('JJA', 7.0), ('SON', 10.0)],
)
def test_save_means_seasonal_values(means_dir, season, expected):
    '''The fixture writes value == month number, so each season mean is exact.

    MAM is the important one: it was defined as `[1, 2, 3]` (JFM), which returned
    2.0 instead of 4.0.
    '''
    path = glob.glob(os.path.join(str(means_dir), season, '*.nc'))[0]
    with xr.open_dataset(path) as r:
        assert float(r['PRECC'].isel(time=0).mean()) == pytest.approx(expected, abs=1e-9)


def test_save_means_output_is_readable(means_dir):
    '''2.5 + 1.4: `annualize` carries the grid attrs, so the write needs
    `.x.to_netcdf`; plain `to_netcdf` raised TypeError.'''
    path = glob.glob(os.path.join(str(means_dir), 'ANN', '*.nc'))[0]
    with xr.open_dataset(path) as r:
        assert 'gw' not in r.attrs
        assert 'PRECC' in r


def test_save_means_requires_output_dirpath(ts):
    with pytest.raises(ValueError, match='output_dirpath'):
        ts.save_means('PRECC', timespan=(1, 10))


def test_save_means_requires_timespan(ts, tmp_path):
    with pytest.raises(ValueError, match='timespan'):
        ts.save_means('PRECC', output_dirpath=str(tmp_path / 'x'))


# --------------------------------------------------------------------------- #
# 2.1
# --------------------------------------------------------------------------- #
def test_calc_reuses_cached_diagnostic(ts):
    '''2.1: with `vn` cached in `.diags` but absent from `.ds`, this KeyError'd.'''
    ts.calc('TS', verbose=False)
    assert 'TS' in ts.diags
    ts.clear_ds('TS')
    out = ts.calc('TS:ann:gm', verbose=False)
    assert out.dims == ('time',)


def test_calc_annual_global_mean_value(ts):
    '''TS is written as month + 273.15 in kelvin, so the annual mean is 6.5 °C.

    Also a check that the K -> °C conversion is applied exactly once: a double
    conversion would give -266.65.
    '''
    out = ts.calc('TS:ann:gm', verbose=False)
    assert float(out[0]) == pytest.approx(6.5, abs=1e-6)


def test_calc_caches_by_spell(ts):
    '''The cache key is the spell, not the variable.'''
    ts.calc('PRECC:ann:gm', verbose=False)
    assert 'PRECC:ann:gm' in ts.diags


def test_calc_reuse_does_not_double_convert_units(ts):
    '''Reusing a cached diag must not apply the unit conversion twice.

    This is why the °C label from finding 1.5 matters: the second pass sees
    '°C' and leaves the values alone.
    '''
    first = float(ts.calc('TS', verbose=False).mean())
    ts.clear_ds('TS')
    second = float(ts.calc('TS:ann:gm', verbose=False).mean())
    assert second == pytest.approx(first, abs=1e-6)


# --------------------------------------------------------------------------- #
# 2.3 / 2.4
# --------------------------------------------------------------------------- #
def test_history_accepts_comps_info_string(hist_tree):
    '''2.3: a plain string must be treated as one hstr, not a character sequence.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    assert h.comps_info['atm'] == ['cam.h0']
    assert sorted(h.paths['atm']) == ['cam.h0']


def test_history_string_comps_info_finds_files(hist_tree):
    '''2.3: the per-character keys found nothing at all.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    assert len(h.paths['atm']['cam.h0']) == 3


def test_history_applies_default_avoid_list(hist_tree):
    '''2.4: the default `avoid_list` must apply to the per-hstr path lists too.

    NB: this test's *name* must not contain any default avoid token — see
    `test_avoid_list_matches_full_path_not_basename` below for why.
    '''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    leaked = [p for p in h.paths['atm']['cam.h0'] if '.once.' in p]
    assert leaked == [], f'{len(leaked)} static files leaked into the path list'
    assert len(h.paths['atm']['cam.h0']) == 3


def test_avoid_list_matches_basename_not_full_path(tmp_path):
    '''3.14: an avoid token in a *parent directory* name must not drop all files.

    Found while writing these tests: a test named `..._once_files` put "once" in
    pytest's temp directory, and the default `avoid_list=['once']` then excluded
    every history file. Any real case stored under a path containing "once"
    (or a user-supplied token) hit the same silent emptiness.
    '''
    casename = 'testcase'
    root = tmp_path / 'run_once_experiment' / casename
    d = root / 'atm' / 'hist'
    d.mkdir(parents=True)
    lat, lon = np.arange(-60, 61, 40.0), np.arange(0, 360, 90.0)
    for ti in xr.date_range('0001-01-01', periods=2, freq='MS', calendar='noleap'):
        da = xr.DataArray(
            np.ones((1, len(lat), len(lon))), dims=('time', 'lat', 'lon'),
            coords={'time': [ti], 'lat': lat, 'lon': lon},
        )
        xr.Dataset({'TS': da}).to_netcdf(d / f'{casename}.cam.h0.{ti.year:04d}-{ti.month:02d}.nc')

    h = case.History(str(root), comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    assert len(h.paths['atm']['cam.h0']) == 2


def test_history_extra_avoid_list_is_additive(hist_tree):
    '''A user-supplied `avoid_list` extends the default rather than replacing it.'''
    root, casename = hist_tree
    h = case.History(
        root, comps=['atm'], comps_info={'atm': 'cam.h0'},
        casename=casename, avoid_list=['0001-02'],
    )
    kept = h.paths['atm']['cam.h0']
    assert all('.once.' not in p for p in kept)
    assert all('0001-02' not in p for p in kept)


def test_history_wildcard_discovery(hist_tree):
    '''The default `'*'` path still discovers hstrs by scanning filenames.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], casename=casename)
    assert 'cam.h0' in h.comps_info['atm']


def test_history_variable_listing(hist_tree):
    '''`vns` is built by inspecting the first history file.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    assert 'TS' in h.vns['atm']['cam.h0']


# --------------------------------------------------------------------------- #
# indexing
# --------------------------------------------------------------------------- #
def test_timeseries_indexes_all_components(ts):
    '''Both components in the fixture must be discovered.'''
    assert set(ts.paths) == {'atm', 'rof'}
    assert 'TS' in ts.vns['atm']['cam.h0']
    assert 'QRUNOFF' in ts.vns['rof']['mosart.h0']


def test_get_comp_hstr(ts):
    assert ts.get_comp_hstr('TS') == [('atm', 'cam.h0')]
    assert ts.get_comp_hstr('NO_SUCH_VAR') == []
