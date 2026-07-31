'''Regression tests for the stale-area-weight bugs.

Covers review findings:
  1.3  `regrid()` copied the source grid's `gw` onto the regridded output. Since
       the weight lives in `.attrs`, xarray could not catch the mismatch, and a
       following `.x.gm` *broadcast* instead of reducing — silently turning a
       global mean into a full outer product.
  2.7  `regrid()` deleted `attrs['lat']`/`attrs['lon']`, so the hemispheric
       means raised `KeyError: 'lat'` afterwards.

The invariant behind both: **`gw.dims` must always be a non-empty subset of the
data dims.** That single property is what makes the whole class of stale-weight
bugs detectable, so it is asserted directly.
'''
import numpy as np
import pytest
import xarray as xr

from x4c import utils

from conftest import coslat_mean

REDUCTIONS = ['gm', 'gs', 'nhm', 'shm', 'nhs', 'shs']


# --------------------------------------------------------------------------- #
# the core invariant
# --------------------------------------------------------------------------- #
def test_gw_dims_subset_invariant_after_regrid(latlon_ds):
    '''1.3: the weight must span dims the data actually has.'''
    rgd = latlon_ds().x.regrid(dlon=5, dlat=5)
    gw_dims = set(rgd.attrs['gw'].dims)
    assert gw_dims, 'weight has no dims'
    assert gw_dims <= set(rgd['TS'].dims), f'{gw_dims} not a subset of {rgd["TS"].dims}'


@pytest.mark.parametrize('op', REDUCTIONS)
@pytest.mark.parametrize(
    'bad_gw',
    [
        xr.DataArray(np.ones(9), dims='ncol'),                       # stale SE weight
        xr.DataArray(np.ones((4, 5)), dims=('nlat', 'nlon')),        # stale POP weight
        xr.DataArray(1.0),                                           # degenerate scalar
    ],
    ids=['stale_ncol', 'stale_nlat_nlon', 'scalar'],
)
def test_mismatched_weight_raises(latlon_ds, op, bad_gw):
    '''1.3: every weighted reduction must refuse a weight it cannot reduce over.

    Silently broadcasting is the actual bug — an error here is the fix.
    '''
    da = latlon_ds().x.da
    da.attrs['gw'] = bad_gw
    with pytest.raises(ValueError, match='stale or mismatched|not all'):
        getattr(da.x, op)


@pytest.mark.parametrize('op', REDUCTIONS)
def test_reductions_collapse_horizontal_dims(latlon_ds, op):
    '''A weighted reduction must remove lat/lon and keep time.'''
    da = latlon_ds(nt=6).x.da
    out = getattr(da.x, op)
    assert out.dims == ('time',), f'{op} returned dims {out.dims}'
    assert out.sizes['time'] == 6


# --------------------------------------------------------------------------- #
# regrid re-derives the grid attrs
# --------------------------------------------------------------------------- #
def test_regrid_rederives_gw_on_fv_grid(latlon_ds):
    '''1.3: an FV source keeps (lat, lon) but the weight must be for the target.'''
    src = latlon_ds()
    rgd = src.x.regrid(dlon=10, dlat=10)
    assert tuple(rgd.attrs['gw'].dims) == ('lat', 'lon')
    assert rgd.attrs['gw'].sizes['lat'] == rgd.sizes['lat']
    assert rgd.attrs['gw'].sizes['lon'] == rgd.sizes['lon']


def test_regrid_rederives_gw_on_pop_grid(pop_ds):
    '''1.3: a POP source has (nlat, nlon); the output must have (lat, lon).'''
    src = pop_ds()
    assert tuple(src.attrs['gw'].dims) == ('nlat', 'nlon')
    rgd = src.x.regrid(dlon=5, dlat=5)
    assert tuple(rgd.attrs['gw'].dims) == ('lat', 'lon')


def test_regrid_repopulates_lat_lon_attrs(latlon_ds):
    '''2.7: lat/lon must be re-attached, not deleted.'''
    rgd = latlon_ds().x.regrid(dlon=5, dlat=5)
    assert 'lat' in rgd.attrs and 'lon' in rgd.attrs
    assert rgd.attrs['lat'].sizes['lat'] == rgd.sizes['lat']


def test_nhm_shm_work_after_regrid(latlon_ds):
    '''2.7: hemispheric means used to raise KeyError after a regrid.'''
    da = latlon_ds().x.regrid(dlon=5, dlat=5).x.da
    for op in ('nhm', 'shm', 'nhs', 'shs'):
        out = getattr(da.x, op)
        assert np.isfinite(out.values).all(), f'{op} produced non-finite values'


def test_nh_and_sh_differ_for_asymmetric_field():
    '''Sanity: hemispheric means must actually split at the equator.'''
    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(2.5, 360, 5.0)
    # linear in latitude -> NH mean > 0 > SH mean
    field = lat[:, None] * np.ones((1, len(lon)))
    ds = xr.Dataset({'X': (('lat', 'lon'), field)}, coords={'lat': lat, 'lon': lon})
    ds = utils.update_ds(ds, path='s.nc', vn='X', comp='atm', grid='fv5x5')
    da = ds.x.da
    assert float(da.x.nhm) > 0 > float(da.x.shm)


# --------------------------------------------------------------------------- #
# numerical correctness
# --------------------------------------------------------------------------- #
def test_gm_matches_hand_computed_coslat_mean(latlon_ds):
    '''The weight must be right, not merely dimensionally valid.'''
    ds = latlon_ds(nt=1)
    got = float(ds.x.da.x.gm[0])
    expected = coslat_mean(ds['TS'].isel(time=0).values, ds['lat'].values)
    assert got == pytest.approx(expected, abs=1e-12)


def test_gm_matches_hand_computed_after_regrid(latlon_ds):
    '''1.3: isolate weight correctness from the remapping itself.

    Compare `gm` of the regridded field against a hand cos(lat) mean of that
    same regridded field, so any interpolation error cancels out.
    '''
    rgd = latlon_ds(nt=1).x.regrid(dlon=5, dlat=5)
    got = float(rgd.x.da.x.gm[0])
    expected = coslat_mean(rgd['TS'].isel(time=0).values, rgd['lat'].values)
    assert got == pytest.approx(expected, abs=1e-10)


def test_gm_of_uniform_field_is_that_value(pop_ds):
    '''A constant field must survive regrid + area mean exactly.'''
    rgd = pop_ds(value=15.0).x.regrid(dlon=5, dlat=5)
    assert float(rgd.x.da.x.gm[0]) == pytest.approx(15.0, abs=1e-9)


def test_dataarray_and_dataset_regrid_paths_agree(latlon_ds):
    '''1.3: `da.x.regrid()` and `ds.x.regrid().x.da` are meant to be equivalent.

    They diverged before, because the DataArray path deleted the lat/lon attrs
    that the Dataset path had just re-derived.
    '''
    ds = latlon_ds()
    from_da = ds.x.da.x.regrid(dlon=5, dlat=5)
    from_ds = ds.x.regrid(dlon=5, dlat=5).x.da
    assert set(from_da.attrs) >= {'gw', 'lat', 'lon'}
    assert np.allclose(from_da.x.gm.values, from_ds.x.gm.values)
    assert np.allclose(from_da.x.nhm.values, from_ds.x.nhm.values)


# --------------------------------------------------------------------------- #
# zavg's volume weight must remain valid
# --------------------------------------------------------------------------- #
def test_zavg_volume_weight_still_reduces(ocean_column_da):
    '''`zavg` folds the vertical into `gw`; the result must still pass validation.'''
    da = ocean_column_da()
    zavg = da.x.zavg(0, 400)
    assert 'z_t' not in zavg.dims
    assert set(zavg.attrs['gw'].dims) == {'nlat', 'nlon'}
    assert zavg.x.gm.dims == ('time',)


def test_zavg_then_gm_is_volume_weighted(ocean_column_da):
    '''`zavg(...).x.gm` must equal a dz*area weighted mean over the whole volume.'''
    da = ocean_column_da()
    got = float(da.x.zavg(0, 400).x.gm[0])
    # uniform dz and uniform area -> a plain mean over z_t, nlat, nlon
    expected = float(da.isel(time=0).mean().values)
    assert got == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- #
# the shared helper
# --------------------------------------------------------------------------- #
def test_coslat_weight_shape_and_values():
    '''`utils.coslat_weight` backs both the load-time and post-regrid paths.'''
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 180.0])
    ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
    gw = utils.coslat_weight(ds)
    assert tuple(gw.dims) == ('lat', 'lon')
    assert gw.isel(lon=0).values == pytest.approx(np.cos(np.deg2rad(lat)))


def test_coslat_weight_without_lon():
    '''A lat-only dataset yields the 1-D weight.'''
    ds = xr.Dataset(coords={'lat': np.array([-45.0, 45.0])})
    gw = utils.coslat_weight(ds)
    assert tuple(gw.dims) == ('lat',)


def test_update_ds_and_regrid_agree_on_convention(latlon_ds):
    '''Load-time and post-regrid weights must use the same convention.

    If these drift apart, `gm` silently changes meaning depending on whether the
    data was regridded.
    '''
    rgd = latlon_ds().x.regrid(dlon=5, dlat=5)
    reloaded = utils.coslat_weight(rgd)
    assert np.allclose(rgd.attrs['gw'].values, reloaded.values)
