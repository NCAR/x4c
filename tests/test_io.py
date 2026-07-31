'''Regression tests for the netCDF write path.

Covers review finding:
  1.4  `.x.to_netcdf()` stripped the grid attrs (`gw`/`lat`/`lon`/`dz`) from the
       live object rather than from a copy, so every subsequent `.x.gm` on the
       object the caller still held raised `KeyError`. Separately, the strip only
       touched the *dataset* attrs, so a variable carrying its own copy of them
       made the write fail with `TypeError`.
'''
import numpy as np
import pytest
import xarray as xr

from x4c import core, utils

GRID_ATTRS = ('gw', 'lat', 'lon', 'dz')


# --------------------------------------------------------------------------- #
# the object must survive the write
# --------------------------------------------------------------------------- #
def test_dataset_attrs_survive_write(latlon_ds, tmp_path):
    '''1.4: writing must not mutate the Dataset being written.'''
    ds = latlon_ds()
    before = sorted(ds.attrs)
    ds.x.to_netcdf(tmp_path / 'a.nc')
    assert sorted(ds.attrs) == before


def test_dataset_still_reducible_after_write(latlon_ds, tmp_path):
    '''1.4: the concrete symptom — `.x.gm` used to raise KeyError afterwards.'''
    ds = latlon_ds()
    before = float(ds.x.da.x.gm[0])
    ds.x.to_netcdf(tmp_path / 'a.nc')
    assert float(ds.x.da.x.gm[0]) == pytest.approx(before, abs=0)


def test_dataarray_attrs_survive_write(latlon_ds, tmp_path):
    '''1.4: same guarantee on the DataArray accessor.'''
    da = latlon_ds().x.da
    before = float(da.x.gm[0])
    da.x.to_netcdf(tmp_path / 'b.nc')
    assert 'gw' in da.attrs
    assert float(da.x.gm[0]) == pytest.approx(before, abs=0)


def test_repeated_writes_are_stable(latlon_ds, tmp_path):
    '''Writing twice must work — the first write used to consume the attrs.'''
    ds = latlon_ds()
    ds.x.to_netcdf(tmp_path / 'c1.nc')
    ds.x.to_netcdf(tmp_path / 'c2.nc')
    assert 'gw' in ds.attrs


# --------------------------------------------------------------------------- #
# variable-level attrs
# --------------------------------------------------------------------------- #
def test_variable_level_grid_attrs_are_stripped(latlon_ds, tmp_path):
    '''1.4: a variable from `da.x.ds` carries its own `gw`; the write must handle it.

    Stripping only the dataset attrs left this raising
    `TypeError: Invalid value for attr 'gw'`.
    '''
    ds = latlon_ds().x.da.x.ds
    assert 'gw' in ds['TS'].attrs, 'fixture no longer reproduces the trap'
    ds.x.to_netcdf(tmp_path / 'd.nc')  # must not raise
    # and the live object keeps them
    assert 'gw' in ds['TS'].attrs


def test_written_file_has_no_grid_attrs(latlon_ds, tmp_path):
    '''netCDF attrs must be scalars/strings, so the grid attrs must not appear.'''
    p = tmp_path / 'e.nc'
    latlon_ds().x.to_netcdf(p)
    with xr.open_dataset(p) as r:
        assert not (set(GRID_ATTRS) & set(r.attrs))
        for v in r.variables:
            assert not (set(GRID_ATTRS) & set(r[v].attrs)), f'{v} leaked grid attrs'


def test_written_data_is_intact(latlon_ds, tmp_path):
    '''The strip must not disturb the data or the coordinates.'''
    ds = latlon_ds()
    p = tmp_path / 'f.nc'
    ds.x.to_netcdf(p)
    with xr.open_dataset(p) as r:
        assert np.allclose(r['TS'].values, ds['TS'].values)
        assert np.allclose(r['lat'].values, ds['lat'].values)


def test_roundtrip_reattaches_weights(latlon_ds, tmp_path):
    '''A written file must reload into a fully usable x4c object.'''
    ds = latlon_ds()
    expected = float(ds.x.da.x.gm[0])
    p = tmp_path / 'g.nc'
    ds.x.to_netcdf(p)
    reloaded = core.open_dataset(p, vn='TS', comp='atm', grid='fv5x5')
    assert float(reloaded.x.da.x.gm[0]) == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# the shared helper
# --------------------------------------------------------------------------- #
def test_drop_grid_attrs_leaves_source_untouched():
    '''`utils.drop_grid_attrs` must be non-destructive at every level.'''
    ds = xr.Dataset({'TS': (('lat',), np.ones(4))}, coords={'lat': np.arange(4.0)})
    ds.attrs = {'gw': xr.DataArray(np.ones(4), dims='lat'), 'comp': 'atm', 'keepme': 'yes'}
    ds['TS'].attrs['gw'] = xr.DataArray(np.ones(4), dims='lat')

    out = utils.drop_grid_attrs(ds)

    assert 'gw' in ds.attrs and 'gw' in ds.variables['TS'].attrs, 'source was mutated'
    assert 'gw' not in out.attrs and 'gw' not in out.variables['TS'].attrs


def test_drop_grid_attrs_preserves_other_metadata():
    '''Only the grid attrs come off; everything else must be kept.'''
    ds = xr.Dataset({'TS': (('lat',), np.ones(4))}, coords={'lat': np.arange(4.0)})
    ds.attrs = {'gw': xr.DataArray(np.ones(4), dims='lat'), 'comp': 'atm', 'title': 'keep'}
    out = utils.drop_grid_attrs(ds)
    assert out.attrs == {'comp': 'atm', 'title': 'keep'}


def test_drop_grid_attrs_shares_data_buffers():
    '''The copy must be shallow, or writing a large dataset would double memory.'''
    ds = xr.Dataset({'TS': (('lat',), np.ones(1000))}, coords={'lat': np.arange(1000.0)})
    ds.attrs = {'gw': xr.DataArray(np.ones(1000), dims='lat')}
    out = utils.drop_grid_attrs(ds)
    assert np.shares_memory(out['TS'].values, ds['TS'].values)


def test_drop_grid_attrs_handles_dataarray():
    '''Both accessors route through this helper.'''
    da = xr.DataArray(np.ones(4), dims='lat', coords={'lat': np.arange(4.0)}, name='TS')
    da.attrs = {'gw': xr.DataArray(np.ones(4), dims='lat'), 'units': 'K'}
    out = utils.drop_grid_attrs(da)
    assert out.attrs == {'units': 'K'}
    assert 'gw' in da.attrs


def test_grid_attrs_constant_matches_helper():
    '''The named constant is what both `to_netcdf` methods rely on.'''
    assert set(utils.GRID_ATTRS) == set(GRID_ATTRS)
