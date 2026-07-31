'''Shared fixtures for the x4c test suite.

Everything here is synthetic — no CESM output, no network. That keeps the suite
runnable in CI and fast enough to be worth running on every push.

Hand-computed expectations use the `noleap` month lengths unless a test says
otherwise, since that is CESM's default calendar.
'''
import numpy as np
import pytest
import xarray as xr

from x4c import utils

# noleap month lengths, used for hand-computing day-weighted means
DAYS_NOLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def weighted_mean(values, days):
    '''The day-weighted mean of `values`, normalized over the days present.

    This is the reference implementation the tests compare against: weights are
    normalized over exactly what is in the bin, not over a calendar year.
    '''
    values = np.asarray(values, dtype=float)
    days = np.asarray(days, dtype=float)
    return float((values * days).sum() / days.sum())


def coslat_mean(values2d, lat, lon_size=None):
    '''cos(lat)-weighted mean of a (lat, lon) field, skipping NaNs.'''
    values2d = np.asarray(values2d, dtype=float)
    n = values2d.shape[1] if lon_size is None else lon_size
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[:, None] * np.ones((1, n))
    m = ~np.isnan(values2d)
    return float((values2d[m] * w[m]).sum() / w[m].sum())


# --------------------------------------------------------------------------- #
# time series
# --------------------------------------------------------------------------- #
@pytest.fixture
def ramp_series():
    '''A 48-month series whose value equals its 0-based time index.

    Makes day-weighted means trivially hand-computable: the value in month
    `i` is just `i`.
    '''
    def _make(calendar='noleap', start='0001-01-01', periods=48):
        t = xr.date_range(start, periods=periods, freq='MS', calendar=calendar)
        return xr.DataArray(
            np.arange(float(periods)), dims='time', coords={'time': t}, name='v',
        )
    return _make


@pytest.fixture
def nan_series():
    '''24 months x 4 cells, with a different missing-data pattern per cell.

    cell 0: fully valid
    cell 1: all NaN            (a land cell in an ocean field)
    cell 2: NaN in year 1 only (a record that starts late)
    cell 3: Jul+Aug of year 1 NaN (a partially missing season)
    '''
    t = xr.date_range('0001-01-01', periods=24, freq='MS', calendar='noleap')
    v = np.tile(np.arange(1.0, 25.0)[:, None], (1, 4))
    v[:, 1] = np.nan
    v[:12, 2] = np.nan
    v[6:8, 3] = np.nan
    return xr.DataArray(v, dims=('time', 'cell'), coords={'time': t}, name='v')


# --------------------------------------------------------------------------- #
# gridded datasets
# --------------------------------------------------------------------------- #
@pytest.fixture
def latlon_ds():
    '''A small FV-style atm Dataset with the x4c grid attrs attached.

    The field varies with latitude so that area weighting actually matters —
    a uniform field would pass even with a wrong weight.
    '''
    def _make(nt=6, dlat=5.0, dlon=5.0, comp='atm', grid='fv5x5'):
        lat = np.arange(-90 + dlat / 2, 90, dlat)
        lon = np.arange(dlon / 2, 360, dlon)
        field = 290.0 - 30.0 * np.sin(np.deg2rad(lat))[:, None] ** 2 + np.zeros((1, len(lon)))
        t = xr.date_range('0001-01-01', periods=nt, freq='MS', calendar='noleap')
        ds = xr.Dataset(
            {'TS': (('time', 'lat', 'lon'), np.tile(field, (nt, 1, 1)))},
            coords={'time': t, 'lat': lat, 'lon': lon},
        )
        ds['TS'].attrs.update({'units': 'K', 'long_name': 'Surface Temperature'})
        return utils.update_ds(ds, path='synthetic.nc', vn='TS', comp=comp, grid=grid)
    return _make


@pytest.fixture
def pop_ds():
    '''A POP-style ocean Dataset: 2-D TLAT/TLONG, TAREA weights, nlat/nlon dims.'''
    def _make(nt=4, nj=40, ni=80, value=15.0):
        tlat = np.linspace(-79, 89, nj)[:, None] * np.ones((1, ni))
        tlon = np.ones((nj, 1)) * np.linspace(0.5, 359.5, ni)[None, :]
        ds = xr.Dataset(
            {
                'SST': (('time', 'nlat', 'nlon'), np.full((nt, nj, ni), value)),
                'TAREA': (('nlat', 'nlon'), np.full((nj, ni), 1e14)),
            },
            coords={
                'time': np.arange(nt),
                'TLAT': (('nlat', 'nlon'), tlat),
                'TLONG': (('nlat', 'nlon'), tlon),
            },
        )
        return utils.update_ds(ds, path='synthetic_pop.nc', vn='SST', comp='ocn', grid='g16')
    return _make


@pytest.fixture
def ocean_column_da():
    '''A (time, z_t, nlat, nlon) ocean DataArray carrying `gw` and `dz` attrs.'''
    def _make(nt=2, nz=5, nj=10, ni=12, seed=0):
        rng = np.random.default_rng(seed)
        z_t = np.arange(nz, dtype=float) * 100.0
        da = xr.DataArray(
            rng.random((nt, nz, nj, ni)),
            dims=('time', 'z_t', 'nlat', 'nlon'),
            coords={'z_t': z_t},
            name='TEMP',
        )
        da.attrs['gw'] = xr.DataArray(np.ones((nj, ni)), dims=('nlat', 'nlon'))
        da.attrs['dz'] = xr.DataArray(np.full(nz, 100.0), dims='z_t', coords={'z_t': z_t})
        return da
    return _make


# --------------------------------------------------------------------------- #
# synthetic CESM directory trees
# --------------------------------------------------------------------------- #
@pytest.fixture
def ts_tree(tmp_path):
    '''Build a synthetic CESM *timeseries* tree and return (root, casename).

    Layout matches the pattern `Timeseries` expects:
        <root>/<comp>/proc/tseries/month_1/<case>.<hstr>.<vn>.<YYYYMM-YYYYMM>.nc

    Written values equal the calendar month number (plus an optional offset), so
    seasonal means are exactly checkable: ANN=6.5, MAM=4.0, JJA=7.0, SON=10.0.
    '''
    casename = 'testcase'
    root = tmp_path / casename
    lat = np.arange(-60, 61, 40.0)
    lon = np.arange(0, 360, 90.0)

    def _write(comp, hstr, vn, y0, y1, offset=0.0, units='1'):
        d = root / comp / 'proc' / 'tseries' / 'month_1'
        d.mkdir(parents=True, exist_ok=True)
        t = xr.date_range(f'{y0:04d}-01-01', f'{y1:04d}-12-01', freq='MS', calendar='noleap')
        month = np.array([x.month for x in t], dtype=float) + offset
        da = xr.DataArray(
            month[:, None, None] * np.ones((1, len(lat), len(lon))),
            dims=('time', 'lat', 'lon'),
            coords={'time': t, 'lat': lat, 'lon': lon},
            name=vn,
        )
        da.attrs.update({'units': units, 'long_name': vn})
        fname = f'{casename}.{hstr}.{vn}.{y0:04d}01-{y1:04d}12.nc'
        xr.Dataset({vn: da}).to_netcdf(d / fname)

    # TS in kelvin across two decades; PRECC unitless; QRUNOFF on the rof component
    _write('atm', 'cam.h0', 'TS', 1, 10, offset=273.15, units='K')
    _write('atm', 'cam.h0', 'TS', 11, 20, offset=273.15, units='K')
    _write('atm', 'cam.h0', 'PRECC', 1, 10)
    _write('rof', 'mosart.h0', 'QRUNOFF', 1, 10)
    return str(root), casename


@pytest.fixture
def hist_tree(tmp_path):
    '''Build a synthetic CESM *history* tree and return (root, casename).

    Writes both `<case>.cam.h0.<date>.nc` and `<case>.cam.h0.once.<date>.nc`, so
    the default `avoid_list` (which excludes `once`) is observable.
    '''
    casename = 'testcase'
    root = tmp_path / casename
    d = root / 'atm' / 'hist'
    d.mkdir(parents=True, exist_ok=True)
    lat = np.arange(-60, 61, 40.0)
    lon = np.arange(0, 360, 90.0)

    for ti in xr.date_range('0001-01-01', periods=3, freq='MS', calendar='noleap'):
        da = xr.DataArray(
            np.ones((1, len(lat), len(lon))),
            dims=('time', 'lat', 'lon'),
            coords={'time': [ti], 'lat': lat, 'lon': lon},
        )
        for tag in ('', '.once'):
            fname = f'{casename}.cam.h0{tag}.{ti.year:04d}-{ti.month:02d}.nc'
            xr.Dataset({'TS': da}).to_netcdf(d / fname)
    return str(root), casename
