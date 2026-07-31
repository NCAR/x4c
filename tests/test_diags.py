'''Regression tests for the diagnostic unit labels.

Covers review finding:
  1.5  `get_NINO34` labelled its result `'K'`, but it is built on `SST`, which is
       already in °C. `Timeseries.calc` converts anything tagged `'K'` by
       subtracting 273.15, so the index came back offset by that amount.

The blanket conversion in `Timeseries.calc` is still a design trap (review §1.5,
"wider design problem"), so `test_no_diagnostic_labelled_kelvin` guards against
the next diagnostic falling into it.
'''
import inspect

import numpy as np
import pytest
import xarray as xr

from x4c import diags, utils


class StubCase:
    '''The minimum surface `diags.Registry` functions need.

    Deliberately not a real `Timeseries`: these tests are about the units
    bookkeeping, and a stub keeps them independent of file discovery.
    '''

    def __init__(self, **arrays):
        self.ds = dict(arrays)

    def load(self, vn, **kws):  # everything is pre-loaded
        pass


@pytest.fixture
def sst_case():
    '''A stub case holding a uniform 27 °C SST field, as `get_SST` would.'''
    def _make(value=27.0, units='°C'):
        lat = np.arange(-87.5, 90, 5.0)
        lon = np.arange(2.5, 360, 5.0)
        ds = xr.Dataset(
            {'SST': (('time', 'lat', 'lon'), np.full((6, len(lat), len(lon)), value))},
            coords={'time': np.arange(6), 'lat': lat, 'lon': lon},
        )
        ds = utils.update_ds(ds, path='s.nc', vn='SST', comp='atm', grid='fv5x5')
        da = ds.x.da
        da.attrs['units'] = units
        return StubCase(SST=da)
    return _make


def apply_calc_unit_block(da):
    '''Replay `Timeseries.calc`'s unit conversion, so the test exercises the trap.

    Kept in sync with `case.py`; `test_calc_unit_block_matches_source` asserts
    the real code still has this shape.
    '''
    if da.units == 'degC':
        da.attrs['units'] = '°C'
    elif da.units == 'K':
        da -= 273.15
        da.attrs['units'] = '°C'
    return da


# --------------------------------------------------------------------------- #
# 1.5
# --------------------------------------------------------------------------- #
def test_nino34_is_labelled_celsius(sst_case):
    '''1.5: the label drives the conversion, so it must say °C not K.'''
    da = diags.Registry.funcs['NINO3.4'](sst_case())
    assert da.attrs['units'] == '°C'


def test_nino34_value_is_not_offset(sst_case):
    '''1.5: a uniform 27 °C ocean gives a Niño-3.4 value of 27.'''
    da = diags.Registry.funcs['NINO3.4'](sst_case(value=27.0))
    assert float(da[0]) == pytest.approx(27.0, abs=1e-9)


def test_nino34_survives_the_calc_unit_block(sst_case):
    '''1.5: passing through `calc`'s conversion must be a no-op.'''
    da = diags.Registry.funcs['NINO3.4'](sst_case(value=27.0))
    before = float(da[0])
    after = float(apply_calc_unit_block(da.copy())[0])
    assert after == pytest.approx(before, abs=1e-12)


def test_kelvin_label_would_reintroduce_the_bug(sst_case):
    '''Pin the failure mode, so the test documents what it is protecting against.'''
    da = diags.Registry.funcs['NINO3.4'](sst_case(value=27.0))
    mislabelled = da.copy()
    mislabelled.attrs['units'] = 'K'
    assert float(apply_calc_unit_block(mislabelled)[0]) == pytest.approx(27.0 - 273.15, abs=1e-9)


def test_nino34_name_and_long_name(sst_case):
    '''The registry key and the DataArray name must agree.'''
    da = diags.Registry.funcs['NINO3.4'](sst_case())
    assert da.name == 'NINO3.4'
    assert 'NINO3.4' in da.attrs['long_name']


def test_nino34_registered_under_dotted_key():
    '''`@F(name='NINO3.4')` is the only dotted registry key; guard the override.'''
    assert 'NINO3.4' in diags.Registry.funcs
    assert diags.Registry.get_F('NINO3.4') is diags.Registry.funcs['NINO3.4']


# --------------------------------------------------------------------------- #
# guards on the wider trap
# --------------------------------------------------------------------------- #
def test_no_diagnostic_labelled_kelvin():
    '''No diagnostic should hard-code `units = 'K'`.

    `Timeseries.calc` subtracts 273.15 from anything so labelled, which is wrong
    for differences, anomalies and indices. A genuine absolute temperature should
    inherit its units from the raw variable instead of asserting them here.
    '''
    offenders = [
        line.strip()
        for line in inspect.getsource(diags).splitlines()
        if "attrs['units'] = 'K'" in line and not line.strip().startswith('#')
    ]
    assert offenders == [], f'diagnostics labelled K: {offenders}'


def test_calc_unit_block_matches_source():
    '''If `calc`'s conversion changes shape, `apply_calc_unit_block` is stale.'''
    from x4c import case
    src = inspect.getsource(case.Timeseries.calc)
    assert "== 'K'" in src and '273.15' in src, (
        'Timeseries.calc no longer converts K -> °C; update apply_calc_unit_block '
        'in this test file to match.'
    )


# --------------------------------------------------------------------------- #
# neighbouring diagnostics keep their labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'vn, expected_units',
    [
        ('SST', '°C'),
        ('SSS', 'gram/kilogram'),
        ('MLD', 'm'),
        ('d18Op', 'permil'),
        ('d18Osw', 'permil'),
        ('dDsw', 'permil'),
        ('RESTOM', 'W/m$^2$'),
        ('DP', 'Pa'),
    ],
)
def test_expected_unit_labels_are_present(vn, expected_units):
    '''A cheap guard that the label set has not drifted.'''
    src = inspect.getsource(diags.Registry.funcs[vn])
    assert repr(expected_units) in src or f'"{expected_units}"' in src, (
        f'{vn} no longer sets units to {expected_units!r}'
    )
