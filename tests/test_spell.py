'''Tests for the spell mini-language parser.

Covers review findings:
  3.1  Spell fragments were executed with `eval()`, so any spell was arbitrary code
       execution and malformed spells failed with confusing interpreter errors.
  3.2  A spell with an unexpected number of colon-separated fields left `vn` as
       `None` and failed later with
       `TypeError: argument of type 'NoneType' is not iterable`.
'''
import numpy as np
import pytest

from x4c.spell import Spell, find_call, parse_call_args


# --------------------------------------------------------------------------- #
# 3.1 — no eval
# --------------------------------------------------------------------------- #
def test_no_eval_in_spell_or_calc():
    '''The parser and its consumer must not call `eval`.'''
    import inspect
    from x4c import case, spell
    for mod in (spell, case):
        src = inspect.getsource(mod)
        bad = [
            line.strip() for line in src.splitlines()
            if 'eval(' in line
            and 'literal_eval' not in line
            and '_eval_node' not in line
            and not line.strip().startswith('#')
        ]
        assert bad == [], f'{mod.__name__} still calls eval: {bad}'


@pytest.mark.parametrize(
    'payload',
    [
        '__import__("os").system("echo pwned")',
        'open("/etc/passwd").read()',
        'print(1)',
        'lambda: 1',
        'globals()',
    ],
)
def test_code_in_arguments_is_rejected(payload):
    '''3.1: an argument that is not a literal or `slice(...)` must be refused.'''
    with pytest.raises(ValueError, match='Cannot call|Unsupported expression|Cannot parse'):
        Spell(f'TS|regrid({payload}):ann:gm')


def test_slice_is_still_allowed():
    '''`slice` must remain callable, or `.sel(lat=slice(...))` breaks.'''
    S = Spell('SST.sel(lat=slice(-5, 5)):ann')
    assert S.slicing_method == 'sel'
    assert S.slicing_kwargs == {'lat': slice(-5, 5)}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'argstring, expected_args, expected_kwargs',
    [
        ('', (), {}),
        ('1, 2', (1, 2), {}),
        ('1.5', (1.5,), {}),
        ('-3', (-3,), {}),
        ('[500, 850]', ([500, 850],), {}),
        ("dlon=1, dlat=2", (), {'dlon': 1, 'dlat': 2}),
        ("z_t=0", (), {'z_t': 0}),
        ("gs='U'", (), {'gs': 'U'}),
        ('lat=slice(-5, 5)', (), {'lat': slice(-5, 5)}),
        ('time=slice(None, 10)', (), {'time': slice(None, 10)}),
    ],
)
def test_parse_call_args(argstring, expected_args, expected_kwargs):
    args, kwargs = parse_call_args(argstring, 'test')
    assert args == expected_args
    assert kwargs == expected_kwargs


def test_parse_call_args_syntax_error_names_the_fragment():
    with pytest.raises(ValueError, match='Cannot parse the arguments of `regrid'):
        parse_call_args('1,,2', 'regrid(1,,2)')


def test_find_call_handles_nesting():
    '''A regex on `[^)]+` cannot do this; the scanner must.'''
    assert find_call('SST.sel(lat=slice(-5,5))', '.sel')[2] == 'lat=slice(-5,5)'


def test_find_call_bare_name():
    assert find_call('T|plev:climo', 'plev')[2] is None


def test_find_call_missing():
    assert find_call('TS:ann:gm', 'regrid') is None


def test_find_call_unbalanced():
    with pytest.raises(ValueError, match='Unbalanced parentheses'):
        find_call('TS|regrid(1,1:ann', 'regrid')


# --------------------------------------------------------------------------- #
# 3.2 — clear errors on malformed spells
# --------------------------------------------------------------------------- #
def test_too_many_fields_raises_clearly():
    '''3.2: used to be `TypeError: argument of type 'NoneType' is not iterable`.'''
    with pytest.raises(ValueError, match='at most 3'):
        Spell('TS:ann:gm:extra')


def test_empty_variable_name_raises():
    with pytest.raises(ValueError, match='no variable name'):
        Spell(':ann:gm')


def test_empty_alias_raises():
    with pytest.raises(ValueError, match='Empty alias'):
        Spell(' ~ TS:ann:gm')


def test_unknown_sa_method_raises_at_parse_time():
    '''Fail while parsing, not after loading and annualizing the data.'''
    with pytest.raises(ValueError, match='Unknown spatial average method'):
        Spell('TS:ann:not_a_method')


def test_non_string_spell_raises_typeerror():
    with pytest.raises(TypeError, match='must be a string'):
        Spell(['TS', 'ann'])


def test_empty_ann_method_normalized_in_two_field_spell():
    '''`'TS:'` must give `ann_method=None`, not `''` (which broke `ann_modifier`).'''
    assert Spell('TS:').ann_method is None


# --------------------------------------------------------------------------- #
# the documented grammar
# --------------------------------------------------------------------------- #
def test_bare_variable():
    S = Spell('TS')
    assert (S.vn, S.ann_method, S.sa_method) == ('TS', None, None)


def test_two_fields():
    S = Spell('TS:ann')
    assert (S.vn, S.ann_method, S.sa_method) == ('TS', 'ann', None)


def test_three_fields():
    S = Spell('TS:ann:gm')
    assert (S.vn, S.ann_method, S.sa_method) == ('TS', 'ann', 'gm')


def test_alias():
    S = Spell('GMSST ~ SST:ann:gm')
    assert S.alias == 'GMSST'
    assert S.vn == 'SST'


def test_regrid_with_args():
    S = Spell('TS|regrid(1,1):ann:gm')
    assert S.regrid == 'regrid(1,1)'
    assert S.regrid_args == (1, 1)


def test_regrid_bare():
    S = Spell('TS|regrid:ann:gm')
    assert S.regrid == 'regrid()'
    assert S.regrid_args == ()


def test_regrid_with_kwargs():
    S = Spell("TS|regrid(dlon=2, dlat=2):ann:gm")
    assert S.regrid_kwargs == {'dlon': 2, 'dlat': 2}


def test_zavg():
    S = Spell('TEMP|zavg(0,1000):ann:gm')
    assert S.zavg_args == (0, 1000)


@pytest.mark.parametrize(
    'spell, expected',
    [
        ('T|plev:climo', None),
        ('T|plev(500):climo', [500]),
        ('T|plev([500,850]):climo', [500, 850]),
        ('T|plev(500,850):climo', [500, 850]),
    ],
)
def test_plev_levels(spell, expected):
    assert Spell(spell).plev_levels == expected


def test_plev_rejects_kwargs():
    with pytest.raises(ValueError, match='positional levels only'):
        Spell('T|plev(levels=500):climo')


def test_isel_slicing():
    S = Spell('TEMP.isel(z_t=0):ann:gm')
    assert S.vn == 'TEMP'
    assert S.slicing_method == 'isel'
    assert S.slicing_kwargs == {'z_t': 0}


def test_dotted_variable_name_survives_slicing():
    '''`S.vn.split('.')[0]` used to truncate `NINO3.4` to `NINO3`.'''
    assert Spell('NINO3.4:ann').vn == 'NINO3.4'
    S = Spell('NINO3.4.isel(time=0):ann')
    assert S.vn == 'NINO3.4'
    assert S.slicing_kwargs == {'time': 0}


def test_combined_modifiers():
    S = Spell('PLEV500 ~ T.isel(time=0)|regrid(1,1)|plev(500):climo:gm')
    assert S.alias == 'PLEV500'
    assert S.vn == 'T'
    assert S.slicing_kwargs == {'time': 0}
    assert S.regrid_args == (1, 1)
    assert S.plev_levels == [500]
    assert S.ann_method == 'climo'
    assert S.sa_method == 'gm'


def test_repr_is_informative():
    assert 'vn=' in repr(Spell('TS:ann:gm'))


@pytest.mark.parametrize('method', list(Spell.SA_METHODS))
def test_all_documented_sa_methods_parse(method):
    assert Spell(f'TS:ann:{method}').sa_method == method
