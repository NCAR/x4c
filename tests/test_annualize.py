'''Regression tests for the day-weighted annualization bugs.

Covers review findings:
  1.1  `annualize(days_weighted=True)` returned an empty array, because the NaN
       mask was applied on the monthly index against an annual result.
  1.2  Day weights were normalized per calendar year rather than per resample
       bin, which mis-weights wraparound seasons (DJF), incomplete edge bins,
       and any bin with missing months.
'''
import warnings

import numpy as np
import pytest
import xarray as xr

from x4c import utils

from conftest import DAYS_NOLEAP, weighted_mean


# --------------------------------------------------------------------------- #
# 1.1 — the day-weighted branch must return data at all
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'months, n_expected',
    [
        (None, 4),          # calendar-year annual mean over 48 months
        ([6, 7, 8], 4),     # JJA
        ([-12, 1, 2], 5),   # DJF: 4 years of monthly data spans 5 YE-FEB bins
        ([3, 4, 5], 4),     # MAM
    ],
    ids=['ann', 'jja', 'djf', 'mam'],
)
def test_days_weighted_returns_data(ramp_series, months, n_expected):
    '''1.1: the result must be non-empty and have one value per resample bin.'''
    out = utils.annualize(ramp_series(), months=months, days_weighted=True)
    assert out.size == n_expected, f'expected {n_expected} bins, got shape {out.shape}'
    assert np.isfinite(out.values).all()


def test_days_weighted_matches_unweighted_shape(ramp_series):
    '''1.1: weighted and unweighted branches must agree on binning.'''
    s = ramp_series()
    for months in (None, [6, 7, 8], [-12, 1, 2]):
        uw = utils.annualize(s, months=months)
        wt = utils.annualize(s, months=months, days_weighted=True)
        assert uw.shape == wt.shape
        assert (uw.time.values == wt.time.values).all()


# --------------------------------------------------------------------------- #
# 1.2 — weights must be normalized over the resample bin
# --------------------------------------------------------------------------- #
def test_annual_mean_matches_hand_computed(ramp_series):
    '''1.2: calendar-year mean is sum(days*val)/365 for a noleap year.'''
    got = utils.annualize(ramp_series(), days_weighted=True).values
    expected = weighted_mean(np.arange(12.0), DAYS_NOLEAP)
    assert got[0] == pytest.approx(expected, abs=1e-12)
    # later years are the same profile shifted by 12
    assert got[1] == pytest.approx(weighted_mean(np.arange(12.0, 24.0), DAYS_NOLEAP), abs=1e-12)


def test_jja_matches_hand_computed(ramp_series):
    '''1.2: a non-wraparound season normalizes over its own 92 days.'''
    got = utils.annualize(ramp_series(), months=[6, 7, 8], days_weighted=True).values
    expected = weighted_mean([5.0, 6.0, 7.0], DAYS_NOLEAP[5:8])
    assert got[0] == pytest.approx(expected, abs=1e-12)


def test_djf_leading_partial_bin(ramp_series):
    '''1.2: the leading DJF bin has no December; it must renormalize over Jan+Feb.

    Before the fix the weights still summed to 59/90, collapsing the value
    toward zero (0.3111 instead of 0.4746).
    '''
    got = utils.annualize(ramp_series(), months=[-12, 1, 2], days_weighted=True).values
    expected = weighted_mean([0.0, 1.0], [31, 28])  # Jan yr1, Feb yr1
    assert got[0] == pytest.approx(expected, abs=1e-12)
    assert got[0] != pytest.approx(0.3111, abs=1e-3), 'regressed to calendar-year weighting'


def test_djf_interior_bin(ramp_series):
    '''1.2: an interior DJF bin spans {Dec Y-1, Jan Y, Feb Y}.'''
    got = utils.annualize(ramp_series(), months=[-12, 1, 2], days_weighted=True).values
    expected = weighted_mean([11.0, 12.0, 13.0], [31, 31, 28])
    assert got[1] == pytest.approx(expected, abs=1e-12)


def test_djf_trailing_partial_bin(ramp_series):
    '''1.2: the trailing DJF bin holds only December, so it equals that value.

    Before the fix this returned 16.19 instead of 47.
    '''
    got = utils.annualize(ramp_series(), months=[-12, 1, 2], days_weighted=True).values
    assert got[-1] == pytest.approx(47.0, abs=1e-12)


def test_leap_february_uses_29_days(ramp_series):
    '''1.2: in a leap year the DJF bin must weight February by 29 days.

    Calendar-year normalization got this wrong because the calendar-year
    denominator no longer matches the bin denominator.
    '''
    s = ramp_series(calendar='standard', start='1999-01-01')
    got = utils.annualize(s, months=[-12, 1, 2], days_weighted=True).values
    # bin 1 = {Dec 1999 (v=11), Jan 2000 (v=12), Feb 2000 (v=13, 29 days)}
    expected = weighted_mean([11.0, 12.0, 13.0], [31, 31, 29])
    assert got[1] == pytest.approx(expected, abs=1e-12)
    # bin 2 = {Dec 2000 (v=23), Jan 2001 (v=24), Feb 2001 (v=25, 28 days)}
    expected2 = weighted_mean([23.0, 24.0, 25.0], [31, 31, 28])
    assert got[2] == pytest.approx(expected2, abs=1e-12)
    # the leap bin must be numerically distinguishable from its neighbour
    assert not np.isclose(got[1], got[2])


# --------------------------------------------------------------------------- #
# NaN semantics: must match the unweighted branch
# --------------------------------------------------------------------------- #
def test_all_nan_cell_stays_nan(nan_series):
    '''`sum` counts NaN as zero, so an all-NaN cell must be masked back to NaN.'''
    wt = utils.annualize(nan_series, months=[6, 7, 8], days_weighted=True).values
    assert np.isnan(wt[:, 1]).all(), 'all-NaN cell came back as a number (probably 0.0)'


def test_partially_missing_record_masked_per_year(nan_series):
    '''A cell whose first year is missing must be NaN in year 1 and valid in year 2.'''
    wt = utils.annualize(nan_series, months=[6, 7, 8], days_weighted=True).values
    assert np.isnan(wt[0, 2])
    assert np.isfinite(wt[1, 2])


def test_partial_bin_renormalizes_like_unweighted(nan_series):
    '''1.2: a bin missing months renormalizes over what is present.

    cell 3 has Jul+Aug of year 1 missing, so JJA year 1 is just June. The
    unweighted branch already behaved this way; the weighted branch used to be
    biased low because the absent weights were never removed.
    '''
    uw = utils.annualize(nan_series, months=[6, 7, 8]).values
    wt = utils.annualize(nan_series, months=[6, 7, 8], days_weighted=True).values
    assert wt[0, 3] == pytest.approx(uw[0, 3], abs=1e-12)
    assert wt[0, 3] == pytest.approx(6.0, abs=1e-12)  # June only


def test_no_divide_by_zero_warning(nan_series):
    '''An empty bin must not produce a RuntimeWarning on the way to NaN.'''
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        utils.annualize(nan_series, months=[6, 7, 8], days_weighted=True).compute()


# --------------------------------------------------------------------------- #
# the unweighted branch must not have changed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'months, expected',
    [
        (None, [5.5, 17.5, 29.5, 41.5]),
        ([6, 7, 8], [6.0, 18.0, 30.0, 42.0]),
        ([-12, 1, 2], [0.5, 12.0, 24.0, 36.0, 47.0]),
    ],
    ids=['ann', 'jja', 'djf'],
)
def test_unweighted_branch_unchanged(ramp_series, months, expected):
    '''Guard: the fixes touched shared code, so pin the unweighted results too.'''
    got = utils.annualize(ramp_series(), months=months).values
    assert got == pytest.approx(expected, abs=1e-12)


def test_negative_month_convention(ramp_series):
    '''`[-12,1,2]` and `[12,1,2]` are equivalent — `annualize` takes abs().

    Documenting this because the negative form is what the docstrings advertise
    for DJF, and it is easy to assume the sign changes the binning.
    '''
    s = ramp_series()
    a = utils.annualize(s, months=[-12, 1, 2], days_weighted=True).values
    b = utils.annualize(s, months=[12, 1, 2], days_weighted=True).values
    assert a == pytest.approx(b, abs=1e-12)


# --------------------------------------------------------------------------- #
# accessor plumbing
# --------------------------------------------------------------------------- #
def test_accessors_reach_the_fixed_code(latlon_ds):
    '''Both `ds.x.annualize` and `da.x.annualize` must return data.'''
    ds = latlon_ds(nt=36)
    out_ds = ds.x.annualize(days_weighted=True)
    out_da = ds.x.da.x.annualize(days_weighted=True)
    assert out_ds['TS'].sizes['time'] == 3
    assert out_da.sizes['time'] == 3
    assert np.isfinite(out_da.values).all()


def test_annualize_preserves_grid_attrs(latlon_ds):
    '''The grid attrs must survive annualization, or downstream `.x.gm` breaks.'''
    ds = latlon_ds(nt=36)
    out = ds.x.annualize(days_weighted=True)
    assert 'gw' in out.attrs
    assert out.attrs['comp'] == 'atm'
    assert np.isfinite(float(out.x.da.x.gm[0]))
