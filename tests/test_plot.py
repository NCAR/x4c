'''Regression tests for the plotting entry points.

Covers review finding:
  2.8  The UXarray branch of `XDataArray.plot` referenced `_projection`, a local
       only bound when `plot` creates its own axes. A caller-supplied `ax` — the
       normal case for multi-panel figures, and what `Timeseries.quickview` does —
       left it unbound, raising `NameError`.

Coverage limit worth stating plainly: the `ux=True` branch raises `ImportError`
before reaching the fixed line when `uxarray` is absent, so the end-to-end path is
only exercised where `uxarray` is installed. The original `NameError` was
reachable under the same condition. `test_ux_consumer_reads_ax_projection`
therefore inspects the source, and the behavioural tests cover the property the
fix depends on: that `ax.projection` is available on both axes origins.
'''
import inspect

import matplotlib
matplotlib.use('Agg')

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

from x4c import core, visual


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


# --------------------------------------------------------------------------- #
# 2.8 — the property the fix relies on
# --------------------------------------------------------------------------- #
def test_subplots_axes_expose_projection():
    '''A caller-supplied `visual.subplots` axes must carry `.projection`.'''
    _, ax = visual.subplots(
        1, 1, {'a': 0}, projs={'a': 'Robinson'}, projs_kws={'a': {'central_longitude': 180}},
    )
    assert hasattr(ax['a'], 'projection')
    assert isinstance(ax['a'].projection, ccrs.Robinson)


def test_self_created_axes_expose_projection():
    '''The axes `plot` builds itself exposes the same attribute.'''
    plt.figure()
    ax = plt.subplot(projection=ccrs.Robinson(central_longitude=180))
    assert isinstance(ax.projection, ccrs.Robinson)


def test_projection_kwargs_reach_the_axes():
    '''`projs_kws` must be forwarded, or the re-derived projection differs.'''
    _, ax = visual.subplots(
        1, 1, {'a': 0}, projs={'a': 'PlateCarree'}, projs_kws={'a': {'central_longitude': 90}},
    )
    assert ax['a'].projection.proj4_params['lon_0'] == pytest.approx(90)


def test_ux_consumer_reads_ax_projection():
    '''2.8: the `to_polycollection` call must not read the unbound local.

    Source-level, because the runtime path needs `uxarray`. The `plt.subplot`
    creation site legitimately still uses the local and must stay.
    '''
    lines = [l.strip() for l in inspect.getsource(core.XDataArray.plot).splitlines()]
    consumers = [l for l in lines if 'to_polycollection' in l]
    assert len(consumers) == 1, f'expected one to_polycollection call, found {consumers}'
    assert 'projection=ax.projection' in consumers[0]

    creation = [l for l in lines if l.startswith('ax = plt.subplot')]
    assert creation and 'projection=_projection' in creation[0], (
        'the axes creation site should still build its own projection'
    )


def test_ux_without_uxarray_raises_importerror(latlon_ds):
    '''Without `uxarray` the branch must fail on the import, not on a NameError.'''
    pytest.importorskip  # noqa: B018  (documents the dependency explicitly)
    try:
        import uxarray  # noqa: F401
        pytest.skip('uxarray installed; the ImportError branch is not taken')
    except ImportError:
        pass

    ncol = 12
    import xarray as xr
    from x4c import utils
    ds = xr.Dataset(
        {'TS': (('ncol',), np.linspace(250, 300, ncol))},
        coords={'lat': (('ncol',), np.linspace(-80, 80, ncol)),
                'lon': (('ncol',), np.linspace(0, 350, ncol))},
    )
    ds = utils.update_ds(ds, path='s.nc', vn='TS', comp='atm', grid='ne30pg3')
    _, ax = visual.subplots(1, 1, {'a': 0}, projs={'a': 'Robinson'})
    with pytest.raises(ImportError, match='UXarray'):
        ds.x.da.x.plot(ax=ax['a'], ux=True)


# --------------------------------------------------------------------------- #
# the ordinary map path still works
# --------------------------------------------------------------------------- #
def test_latlon_map_plot_returns_axes(latlon_ds):
    '''A regular lat/lon map is the common case; keep it covered.'''
    da = latlon_ds(nt=1).x.da.isel(time=0)
    fig, ax = da.x.plot()
    assert fig is not None and ax is not None


def test_map_plot_onto_supplied_axes(latlon_ds):
    '''Passing `ax` must return just the axes, not a (fig, ax) pair.'''
    da = latlon_ds(nt=1).x.da.isel(time=0)
    _, axd = visual.subplots(1, 1, {'a': 0}, projs={'a': 'Robinson'})
    out = da.x.plot(ax=axd['a'])
    assert out is axd['a']


def test_timeseries_plot_path(latlon_ds):
    '''A 1-D result takes the non-map branch.'''
    da = latlon_ds(nt=12).x.da.x.gm
    fig, ax = da.x.plot()
    assert fig is not None


def test_is_map_detection(latlon_ds, pop_ds, ocean_column_da):
    '''`is_map` drives which branch `plot` takes.'''
    assert latlon_ds(nt=1).x.da.isel(time=0).x.is_map()
    assert pop_ds(nt=1).x.da.isel(time=0).x.is_pop()
    assert not latlon_ds(nt=12).x.da.x.gm.x.is_map()


def test_add_annotation_letters():
    '''`visual.add_annotation` labels panels a), b), ... in order.'''
    _, ax = visual.subplots(1, 2, {'a': 0, 'b': 1})
    visual.add_annotation(ax, style=')')
    texts = [t.get_text() for a in ax.values() for t in a.texts]
    assert texts == ['a)', 'b)']
