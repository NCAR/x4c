'''Tests for the group-3 fragility and error-handling fixes.

Covers review findings:
  3.3   `rm_timespan` built `rm -f` commands with `shell=True`, interpolating
        `root_dir` unquoted.
  3.4   `download` did not check the HTTP status, so an error page was written into
        the weight-file cache and never re-fetched.
  3.5   Weight files were downloaded into the installed package directory.
  3.6   `mpi4py` was imported at module load, initializing MPI for every user.
  3.8   `geo_mean` fell through to an unbound local (`UnboundLocalError`).
  3.9   `Logs.get_vars` used `lines.index(line)` inside a loop over `lines`.
  3.10  Its variable scan skipped the first line of every file by construction.
  3.11  `nearest2d` left `lats2d` unbound for mixed dimensionality and assumed a
        (lat, lon) axis order in the NaN mask.
  3.12  `eof(weight=True)` left `coslat` unbound when no latitude was available.
  3.13  `gen_ts(timestep=None)` — the default — failed deep inside `parse_timestamps`.
'''
import gzip
import inspect
import os

import numpy as np
import pytest
import xarray as xr

from x4c import case, core, utils


# --------------------------------------------------------------------------- #
# 3.6 — MPI must be imported lazily
# --------------------------------------------------------------------------- #
def test_mpi_not_imported_at_module_load():
    '''3.6: `import x4c` must not initialize the MPI runtime.'''
    src = inspect.getsource(case)
    module_level = [
        line for line in src.splitlines()
        if line.startswith(('import mpi4py', 'from mpi4py'))
    ]
    assert module_level == [], f'mpi4py imported at module level: {module_level}'


def test_mpi_helper_exists_and_is_used():
    '''The three parallel methods must go through the lazy helper.'''
    assert hasattr(case, '_get_mpi_comm')
    for meth in (case.History.bigbang, case.History.bigcrunch, case.History.gen_ts):
        assert '_get_mpi_comm()' in inspect.getsource(meth), f'{meth.__name__} bypasses the helper'


# --------------------------------------------------------------------------- #
# 3.13 — gen_ts argument validation
# --------------------------------------------------------------------------- #
def test_gen_ts_requires_timestep(hist_tree):
    '''3.13: `timestep=None` is the default and used to raise a TypeError deep inside
    `parse_timestamps`; it must be rejected up front.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    with pytest.raises(ValueError, match='timestep'):
        h.gen_ts(output_dirpath='/tmp/x4c_unused', timespan=(1, 1), comps=['atm'])


def test_gen_ts_requires_timespan(hist_tree):
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    with pytest.raises(ValueError, match='timespan'):
        h.gen_ts(output_dirpath='/tmp/x4c_unused', comps=['atm'])


# --------------------------------------------------------------------------- #
# timespan normalization (shared by gen_ts / load / get_ts)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'timespan, expected',
    [
        ((1, 20), ('0001', '0020')),
        (('0001-01', '0020-12'), ('0001-01', '0020-12')),
        (('0001-01', 20), ('0001-01', '0020')),   # mixed: used to be skipped entirely
        ((1, '0020-12'), ('0001', '0020-12')),
        (None, None),
    ],
)
def test_normalize_timespan(timespan, expected):
    '''Element-wise, so an already-formatted string is never re-zero-padded.

    The all-or-nothing check this replaced left a mixed tuple unconverted; naively
    loosening it to `or` would have fed `'0001-01'` through `int_to_timestamp`,
    producing `'0000-1-01'`.
    '''
    assert utils.normalize_timespan(timespan) == expected


# --------------------------------------------------------------------------- #
# 3.3 — rm_timespan must not shell out
# --------------------------------------------------------------------------- #
def test_rm_timespan_does_not_use_shell(hist_tree):
    '''3.3: no `shell=True`, so `root_dir` can never be reinterpreted as shell syntax.'''
    src = inspect.getsource(case.History.rm_timespan)
    assert 'shell=True' not in src.replace('`rm -f ... shell=True`', '')
    assert 'rm -f' not in src.replace('`rm -f ... shell=True`', '')


def test_rm_timespan_rehearsal_deletes_nothing(hist_tree):
    '''The default must be a dry run.'''
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    before = sorted(os.listdir(os.path.join(root, 'atm', 'hist')))
    found = h.rm_timespan((1, 1), comps=['atm'])
    assert len(found) > 0
    assert sorted(os.listdir(os.path.join(root, 'atm', 'hist'))) == before


def test_rm_timespan_actually_deletes_when_asked(hist_tree):
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    hist = os.path.join(root, 'atm', 'hist')
    targets = h.rm_timespan((1, 1), comps=['atm'], rehearsal=False)
    assert len(targets) > 0
    for p in targets:
        assert not os.path.exists(p)
    # the `.once.` static files are matched by the year glob too, so just check that
    # nothing outside the returned list was touched
    assert all(os.path.exists(p) or p in targets
               for p in (os.path.join(hist, f) for f in os.listdir(hist)))


def test_rm_timespan_handles_paths_with_spaces(tmp_path):
    '''3.3: the exact failure mode of shell interpolation.'''
    casename = 'testcase'
    root = tmp_path / 'my run dir' / casename
    hist = root / 'atm' / 'hist'
    hist.mkdir(parents=True)
    lat, lon = np.arange(-60, 61, 40.0), np.arange(0, 360, 90.0)
    for ti in xr.date_range('0001-01-01', periods=2, freq='MS', calendar='noleap'):
        da = xr.DataArray(
            np.ones((1, len(lat), len(lon))), dims=('time', 'lat', 'lon'),
            coords={'time': [ti], 'lat': lat, 'lon': lon},
        )
        xr.Dataset({'TS': da}).to_netcdf(hist / f'{casename}.cam.h0.{ti.year:04d}-{ti.month:02d}.nc')

    h = case.History(str(root), comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    found = h.find_timespan_files((1, 1), comps=['atm'])
    assert len(found) == 2, f'glob failed on a path containing spaces: {found}'


def test_rm_timespan_empty_match_is_not_an_error(hist_tree):
    root, casename = hist_tree
    h = case.History(root, comps=['atm'], comps_info={'atm': 'cam.h0'}, casename=casename)
    assert h.rm_timespan((9000, 9001), comps=['atm']) == []


# --------------------------------------------------------------------------- #
# 3.4 / 3.5 — download and cache location
# --------------------------------------------------------------------------- #
def test_download_raises_for_status(monkeypatch, tmp_path):
    '''3.4: an HTTP error must raise, not be written into the cache as data.'''
    import requests

    class FakeResp:
        headers = {'content-length': '9'}
        def raise_for_status(self):
            raise requests.HTTPError('404 Not Found')
        def iter_content(self, chunk_size=1):
            yield b'<html>404'

    monkeypatch.setattr(requests, 'get', lambda *a, **k: FakeResp())
    target = tmp_path / 'wgt.nc.gz'
    with pytest.raises(requests.HTTPError):
        utils.download('https://example.invalid/wgt.nc.gz', str(target))
    assert not target.exists(), 'an error body was cached as if it were data'


def test_download_is_atomic_on_failure(monkeypatch, tmp_path):
    '''3.4: an interrupted transfer must not leave a truncated file behind.'''
    import requests

    class FakeResp:
        headers = {'content-length': '100'}
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=1):
            yield b'partial'
            raise OSError('connection reset')

    monkeypatch.setattr(requests, 'get', lambda *a, **k: FakeResp())
    target = tmp_path / 'wgt.nc.gz'
    with pytest.raises(OSError):
        utils.download('https://example.invalid/wgt.nc.gz', str(target))
    assert not target.exists()
    assert not (tmp_path / 'wgt.nc.gz.part').exists(), 'temp file left behind'


def test_download_success_writes_the_file(monkeypatch, tmp_path):
    import requests

    class FakeResp:
        headers = {'content-length': '5'}
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size=1):
            yield b'hello'

    monkeypatch.setattr(requests, 'get', lambda *a, **k: FakeResp())
    target = tmp_path / 'sub' / 'wgt.nc.gz'
    utils.download('https://example.invalid/wgt.nc.gz', str(target), show_bar=False)
    assert target.read_bytes() == b'hello'


def test_cache_dir_is_not_the_package_dir(monkeypatch):
    '''3.5: never write multi-MB binaries into site-packages or the source tree.'''
    monkeypatch.delenv('X4C_CACHE_DIR', raising=False)
    monkeypatch.delenv('XDG_CACHE_HOME', raising=False)
    pkg_dir = os.path.dirname(core.__file__)
    assert not utils.cache_dir().startswith(pkg_dir)
    assert utils.cache_dir().endswith(os.path.join('.cache', 'x4c'))


def test_cache_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv('X4C_CACHE_DIR', str(tmp_path / 'mycache'))
    assert utils.cache_dir() == str(tmp_path / 'mycache')

    monkeypatch.delenv('X4C_CACHE_DIR')
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    assert utils.cache_dir() == os.path.join(str(tmp_path / 'xdg'), 'x4c')


def test_fetch_wgt_file_prefers_cache(monkeypatch, tmp_path):
    '''An existing cached file must short-circuit the download.'''
    monkeypatch.setenv('X4C_CACHE_DIR', str(tmp_path))
    (tmp_path / 'map_test.nc.gz').write_bytes(b'x')

    def boom(*a, **k):
        raise AssertionError('should not download when the cache is populated')

    monkeypatch.setattr(utils, 'download', boom)
    assert utils.fetch_wgt_file('map_test.nc.gz') == str(tmp_path / 'map_test.nc.gz')


def test_fetch_wgt_file_downloads_into_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('X4C_CACHE_DIR', str(tmp_path / 'c'))
    calls = {}

    def fake_download(url, fname, **k):
        calls['url'], calls['fname'] = url, fname
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        # must be real gzip: `fetch_wgt_file` rejects anything else, so that GitHub's
        # 200-plus-HTML answer to a missing /raw/ path cannot end up cached
        open(fname, 'wb').write(gzip.compress(b'x'))

    monkeypatch.setattr(utils, 'download', fake_download)
    out = utils.fetch_wgt_file('map_new.nc.gz', verbose=False)
    assert out == str(tmp_path / 'c' / 'map_new.nc.gz')
    assert calls['url'].endswith('map_new.nc.gz')


# --------------------------------------------------------------------------- #
# 3.8 — geo_mean
# --------------------------------------------------------------------------- #
def test_geo_mean_raises_when_metadata_is_incomplete(latlon_ds):
    '''3.8: `gw` without `lat`/`lon` matched no branch and hit an unbound local.'''
    da = latlon_ds(nt=1).x.da
    da.attrs.pop('lat', None)
    da.attrs.pop('lon', None)
    with pytest.raises(ValueError, match='Cannot compute a geographical mean'):
        utils.geo_mean(da)


def test_geo_mean_works_with_full_metadata(latlon_ds):
    da = latlon_ds(nt=1).x.da
    assert np.isfinite(float(utils.geo_mean(da).isel(time=0)))


def test_geo_mean_coslat_fallback(latlon_ds):
    '''With no `gw` at all it uses the cos(lat) branch on the coordinates.'''
    da = latlon_ds(nt=1).x.da
    for k in ('gw', 'lat', 'lon'):
        da.attrs.pop(k, None)
    assert np.isfinite(float(utils.geo_mean(da).isel(time=0)))


# --------------------------------------------------------------------------- #
# 3.11 — nearest2d
# --------------------------------------------------------------------------- #
@pytest.fixture
def site_da():
    def _make(transpose=False, nan_corner=False):
        lat = np.arange(-80, 81, 20.0)
        lon = np.arange(0, 360, 30.0)
        data = np.tile(np.arange(len(lon), dtype=float), (len(lat), 1))
        da = xr.DataArray(data, dims=('lat', 'lon'),
                          coords={'lat': lat, 'lon': lon}, name='X')
        if nan_corner:
            da = da.where(~((da.lat < -60) & (da.lon < 60)))
        return da.transpose('lon', 'lat') if transpose else da
    return _make


def test_nearest2d_picks_the_nearest_cell(site_da):
    out = site_da().x.nearest2d(lat=[0.0], lon=[90.0])
    assert float(out.isel(site=0).lat) == 0.0
    assert float(out.isel(site=0).lon) == 90.0


def test_nearest2d_is_dim_order_independent(site_da):
    '''3.11: `.values` follows the DataArray's dim order, so the mask needed a
    transpose; without it a (lon, lat) array indexed the wrong cell.'''
    a = site_da(transpose=False).x.nearest2d(lat=[40.0], lon=[120.0])
    b = site_da(transpose=True).x.nearest2d(lat=[40.0], lon=[120.0])
    assert float(a.isel(site=0)) == float(b.isel(site=0))
    assert float(b.isel(site=0).lat) == 40.0
    assert float(b.isel(site=0).lon) == 120.0


def test_nearest2d_skips_nan_cells(site_da):
    '''A NaN target region must resolve to the nearest *valid* cell.'''
    out = site_da(nan_corner=True).x.nearest2d(lat=[-80.0], lon=[0.0])
    assert np.isfinite(float(out.isel(site=0)))


def test_nearest2d_mixed_dimensionality_raises():
    '''3.11: used to leave `lats2d` unbound for an `UnboundLocalError`.'''
    nj, ni = 4, 5
    da = xr.DataArray(
        np.ones((nj, ni)), dims=('nlat', 'nlon'),
        coords={
            'lat': (('nlat', 'nlon'), np.zeros((nj, ni))),   # 2-D
            'lon': (('nlon',), np.arange(float(ni))),        # 1-D
        },
        name='X',
    )
    with pytest.raises(ValueError, match='both be 1-D or both 2-D'):
        da.x.nearest2d(lat=[0.0], lon=[1.0], lat_dim='nlat', lon_dim='nlon')


def test_nearest2d_all_nan_raises(site_da):
    da = site_da() * np.nan
    with pytest.raises(ValueError, match='No valid'):
        da.x.nearest2d(lat=[0.0], lon=[0.0])


# --------------------------------------------------------------------------- #
# 3.12 — eof
# --------------------------------------------------------------------------- #
def test_eof_without_latitude_raises(latlon_ds):
    '''3.12: used to leave `coslat` unbound for a `NameError`.'''
    ds = latlon_ds(nt=24)
    da = ds.x.da.rename({'lat': 'y'})
    da.attrs.pop('lat', None)
    with pytest.raises(ValueError, match='EOF weighting needs a latitude'):
        da.x.eof()


def test_eof_without_time_raises(latlon_ds):
    da = latlon_ds(nt=24).x.da.isel(time=0)
    with pytest.raises(ValueError, match='needs a `time` dimension'):
        da.x.eof()


def test_eof_runs_on_a_normal_field():
    '''Smoke test with a field that has real variance to decompose.'''
    rng = np.random.default_rng(0)
    lat = np.arange(-60, 61, 20.0)
    lon = np.arange(0, 360, 40.0)
    t = xr.date_range('0001-01-01', periods=36, freq='MS', calendar='noleap')
    data = rng.standard_normal((len(t), len(lat), len(lon)))
    da = xr.DataArray(data, dims=('time', 'lat', 'lon'),
                      coords={'time': t, 'lat': lat, 'lon': lon}, name='X')
    pcs, eofs, var = da.x.eof(n=3)
    assert pcs.sizes['mode'] == 3
    assert float(var.sum()) <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# 3.9 / 3.10 — Logs.get_vars
# --------------------------------------------------------------------------- #
LOG_HEADER = [
    'some preamble\n',
    'This run        started from\n',
    ' date(month-day-year): 01-01-0001\n',
]


@pytest.fixture
def ocn_log(tmp_path):
    '''Write a synthetic gzipped POP log.

    `repeat_line` inserts a duplicated line before the header, which is what broke
    `lines.index(line)`.
    '''
    def _make(nmonths=3, repeat_line=False, first_line_is_a_var=False, omit_header=False):
        lines = []
        if first_line_is_a_var:
            # exercises 3.10: the old `elif` never tested the first line for a match
            lines.append('TEMP:  1.0\n')
        if repeat_line:
            lines += ['duplicated\n', 'duplicated\n']
        if not omit_header:
            lines += LOG_HEADER
        for i in range(nmonths):
            lines.append(f'TEMP:  {float(i)}\n')
            lines.append(f'SALT:  {float(i) * 0.001}\n')
        p = tmp_path / 'ocn.log.000000-000000.gz'
        with gzip.open(p, 'wt') as f:
            f.writelines(lines)
        return str(tmp_path)
    return _make


def test_logs_parses_values(ocn_log):
    L = case.Logs(ocn_log(nmonths=4), comp='ocn')
    L.get_vars(vn=['TEMP', 'SALT'])
    assert list(L.df['TEMP']) == [0.0, 1.0, 2.0, 3.0]


def test_logs_handles_duplicated_lines(ocn_log):
    '''3.9: `lines.index(line)` returned the first equal line, not the current one.'''
    L = case.Logs(ocn_log(nmonths=3, repeat_line=True), comp='ocn')
    L.get_vars(vn=['TEMP'])
    assert list(L.df['TEMP']) == [0.0, 1.0, 2.0]


def test_logs_reads_a_match_on_the_first_line(ocn_log):
    '''3.10: initializing inside the scan loop skipped line 0 of every file.'''
    L = case.Logs(ocn_log(nmonths=2, first_line_is_a_var=True), comp='ocn')
    L.get_vars(vn=['TEMP'])
    assert list(L.df['TEMP']) == [1.0, 0.0, 1.0]  # the leading line is now counted


def test_logs_missing_header_raises_clearly(ocn_log):
    '''`start_date` used to stay unbound, giving a bare NameError.'''
    L = case.Logs(ocn_log(omit_header=True), comp='ocn')
    with pytest.raises(ValueError, match='Could not find the run start date'):
        L.get_vars(vn=['TEMP'])


def test_logs_absent_variable_is_skipped(ocn_log):
    '''A variable not in the log must not break the DataFrame construction.'''
    L = case.Logs(ocn_log(nmonths=2), comp='ocn')
    L.get_vars(vn=['TEMP', 'NOT_IN_LOG'])
    assert 'TEMP' in L.df.columns
    assert 'NOT_IN_LOG' not in L.df.columns


def test_logs_empty_directory_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError, match='No `ocn.log'):
        case.Logs(str(tmp_path), comp='ocn')


def test_logs_annual_means(ocn_log):
    L = case.Logs(ocn_log(nmonths=12), comp='ocn')
    L.get_vars(vn=['TEMP'])
    assert len(L.df_ann) == 1
    assert float(L.df_ann['TEMP'].iloc[0]) == pytest.approx(np.mean(np.arange(12.0)))


# --------------------------------------------------------------------------- #
# 3.7 — the global keep_attrs is documented rather than silent
# --------------------------------------------------------------------------- #
def test_keep_attrs_is_set_and_documented():
    '''3.7: the process-wide side effect stays, but must be discoverable.'''
    assert xr.get_options()['keep_attrs'] is True
    import x4c
    assert 'keep_attrs' in (x4c.__doc__ or ''), 'package docstring does not mention it'
    assert 'PROCESS-WIDE' in inspect.getsource(core)[:2000]
