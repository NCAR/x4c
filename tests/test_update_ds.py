'''Regression tests for component metadata resolution.

Covers review finding:
  2.2  `update_ds` looked up the component in dicts that covered only
       atm/ocn/ice/lnd, so `comp='rof'` raised `KeyError` — even though `rof` is
       a default component of both `History` and `Timeseries`.
'''
import numpy as np
import pytest
import xarray as xr

from x4c import utils

CESM_COMPS = ['atm', 'ocn', 'lnd', 'ice', 'rof']


@pytest.fixture
def plain_latlon():
    '''A minimal lat/lon dataset with no area variable.'''
    def _make(varname='Q'):
        return xr.Dataset(
            {varname: (('time', 'lat', 'lon'), np.ones((2, 4, 8)))},
            coords={
                'time': np.arange(2),
                'lat': np.arange(-60, 61, 40.0),
                'lon': np.arange(0, 360, 45.0),
            },
        )
    return _make


# --------------------------------------------------------------------------- #
# 2.2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('comp', CESM_COMPS)
def test_every_default_component_resolves(plain_latlon, comp):
    '''2.2: all five components CESM writes must be accepted.

    `rof` used to raise KeyError here.
    '''
    ds = utils.update_ds(plain_latlon(), path='s.nc', vn='Q', comp=comp)
    assert ds.attrs['comp'] == comp
    assert 'gw' in ds.attrs, f'no weight attached for comp={comp}'


def test_rof_specifically(plain_latlon):
    '''2.2: named explicitly, since this is the reported failure.'''
    ds = utils.update_ds(plain_latlon(), path='s.nc', vn='QRUNOFF', comp='rof')
    assert tuple(ds.attrs['gw'].dims) == ('lat', 'lon')
    assert 'lat' in ds.attrs and 'lon' in ds.attrs


def test_unlisted_component_degrades_gracefully(plain_latlon):
    '''An unrecognized component should fall back, not raise.

    The lookups are all guarded by `in ds`, so a wrong guess costs nothing while
    a hard failure would block a whole workflow.
    '''
    ds = utils.update_ds(plain_latlon(), path='s.nc', vn='Q', comp='some_future_comp')
    assert 'gw' in ds.attrs
    assert tuple(ds.attrs['gw'].dims) == ('lat', 'lon')


# --------------------------------------------------------------------------- #
# the weight actually chosen
# --------------------------------------------------------------------------- #
def test_ocn_prefers_tarea():
    '''For POP output the cell area is authoritative, not cos(lat).'''
    nj, ni = 6, 8
    ds = xr.Dataset(
        {
            'SST': (('nlat', 'nlon'), np.ones((nj, ni))),
            'TAREA': (('nlat', 'nlon'), np.full((nj, ni), 3.0)),
        },
        coords={
            'TLAT': (('nlat', 'nlon'), np.zeros((nj, ni))),
            'TLONG': (('nlat', 'nlon'), np.zeros((nj, ni))),
        },
    )
    ds = utils.update_ds(ds, path='s.nc', vn='SST', comp='ocn')
    assert np.allclose(ds.attrs['gw'].values, 3.0)
    assert tuple(ds.attrs['gw'].dims) == ('nlat', 'nlon')


def test_atm_prefers_area_variable():
    '''An `area` variable wins over the cos(lat) fallback.'''
    ncol = 10
    ds = xr.Dataset(
        {'TS': (('ncol',), np.ones(ncol)), 'area': (('ncol',), np.full(ncol, 7.0))},
        coords={'lat': (('ncol',), np.linspace(-80, 80, ncol)),
                'lon': (('ncol',), np.linspace(0, 350, ncol))},
    )
    ds = utils.update_ds(ds, path='s.nc', vn='TS', comp='atm', grid='ne30pg3')
    assert np.allclose(ds.attrs['gw'].values, 7.0)


def test_coslat_fallback_when_no_area(plain_latlon):
    '''Without an area variable the weight is cos(lat) over (lat, lon).'''
    ds = utils.update_ds(plain_latlon(), path='s.nc', vn='Q', comp='atm')
    expected = np.cos(np.deg2rad(ds['lat'].values))
    assert np.allclose(ds.attrs['gw'].isel(lon=0).values, expected)


def test_explicit_names_override_the_component_defaults():
    '''`gw_name`/`lat_name`/`lon_name` must take precedence.'''
    ds = xr.Dataset(
        {'X': (('y', 'x'), np.ones((3, 4))), 'myarea': (('y', 'x'), np.full((3, 4), 2.0))},
        coords={'mylat': (('y',), np.array([-45.0, 0.0, 45.0])),
                'mylon': (('x',), np.arange(4.0))},
    )
    ds = utils.update_ds(
        ds, path='s.nc', vn='X', comp='atm',
        gw_name='myarea', lat_name='mylat', lon_name='mylon',
    )
    assert np.allclose(ds.attrs['gw'].values, 2.0)
    assert ds.attrs['lat'].dims == ('y',)


# --------------------------------------------------------------------------- #
# path bookkeeping
# --------------------------------------------------------------------------- #
def test_path_is_absolute(plain_latlon):
    '''`Timeseries.load` compares `ds.path` against a glob result.'''
    ds = utils.update_ds(plain_latlon(), path='relative.nc', vn='Q', comp='atm')
    assert ds.attrs['path'].startswith('/')


def test_path_list_is_preserved(plain_latlon):
    '''A multi-file open records every path.'''
    ds = utils.update_ds(plain_latlon(), path=['a.nc', 'b.nc'], vn='Q', comp='atm')
    assert isinstance(ds.attrs['path'], list) and len(ds.attrs['path']) == 2
