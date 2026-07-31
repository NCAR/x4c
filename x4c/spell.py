import ast

#: the only callables a spell argument may invoke. `slice` is needed for
#: `.sel(lat=slice(-5, 5))`, which is the whole point of supporting `sel` at all.
_ALLOWED_CALLS = {'slice': slice}


def _eval_node(node, where):
    ''' Evaluate a restricted AST node

    Accepts literals, list/tuple/set/dict displays, unary +/- (so negative numbers
    work), and calls to the names in :data:`_ALLOWED_CALLS`. Anything else raises.

    This exists so spells do not have to be `eval`-ed. A spell is a user-facing
    string that may come from a config file, a notebook parameter or a CLI flag;
    running it as Python meant any spell was arbitrary code execution, and it also
    made malformed spells fail with confusing interpreter errors rather than a
    message naming the spell.
    '''
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, where)
        return +value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.List):
        return [_eval_node(e, where) for e in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, where) for e in node.elts)

    if isinstance(node, ast.Set):
        return {_eval_node(e, where) for e in node.elts}

    if isinstance(node, ast.Dict):
        return {_eval_node(k, where): _eval_node(v, where)
                for k, v in zip(node.keys, node.values)}

    if isinstance(node, ast.Call):
        name = getattr(node.func, 'id', None)
        if name not in _ALLOWED_CALLS:
            allowed = ', '.join(sorted(_ALLOWED_CALLS))
            raise ValueError(
                f'Cannot call `{name}` in `{where}`: the only callable allowed in a '
                f'spell argument is `{allowed}`.'
            )
        args = [_eval_node(a, where) for a in node.args]
        kwargs = {kw.arg: _eval_node(kw.value, where) for kw in node.keywords}
        return _ALLOWED_CALLS[name](*args, **kwargs)

    raise ValueError(
        f'Unsupported expression in `{where}`: spell arguments may only be numbers, '
        'strings, True/False/None, lists/tuples/dicts of those, or `slice(...)`.'
    )


def parse_call_args(argstring, where):
    ''' Parse an argument list into (args, kwargs)

    ``'1, 2'``               -> ``((1, 2), {})``
    ``'lat=slice(-5, 5)'``   -> ``((), {'lat': slice(-5, 5)})``
    ``''`` or ``None``       -> ``((), {})``

    Args:
        argstring (str or None): the text between the parentheses
        where (str): the spell fragment, used in error messages

    Returns:
        tuple: (args, kwargs)
    '''
    if argstring is None or argstring.strip() == '':
        return (), {}

    # wrapping in a call is the tidiest way to get Python to parse an argument list
    try:
        tree = ast.parse(f'_f({argstring})', mode='eval')
    except SyntaxError as e:
        raise ValueError(f'Cannot parse the arguments of `{where}`: {e.msg}') from e

    call = tree.body
    args = tuple(_eval_node(a, where) for a in call.args)
    kwargs = {kw.arg: _eval_node(kw.value, where) for kw in call.keywords}
    return args, kwargs


def find_call(text, name):
    ''' Locate `name` or `name(...)` in `text`, honouring nested parentheses

    A regex like ``name\\([^)]+\\)`` cannot handle ``sel(lat=slice(-5,5))``, so scan
    for the matching close paren instead.

    Returns:
        tuple or None: (start, end, argstring), with `end` exclusive and `argstring`
        `None` for a bare `name` written without parentheses.
    '''
    i = text.find(name)
    if i < 0:
        return None

    j = i + len(name)
    if j >= len(text) or text[j] != '(':
        return i, j, None  # bare, e.g. `|plev`

    depth = 0
    for k in range(j, len(text)):
        if text[k] == '(':
            depth += 1
        elif text[k] == ')':
            depth -= 1
            if depth == 0:
                return i, k + 1, text[j + 1:k]

    raise ValueError(f'Unbalanced parentheses after `{name}` in `{text}`.')


class Spell:
    ''' The Spell System

    A "spell" is a string that summarizes a series of data processing steps.
    A basic sentence should be in the form: "vn:ann_method:sa_method", where

    - `vn`: a variable name
    - `ann_method`: annualization method
    - `sa_method`: spatial average method

    One may also add more operations after the `vn` part:

    - "|plev": to interpolate the data from the model z levels to the pressure levels
    - "|regrid": to regrid the data from the model grid to the regular lat/lon grid
    - "|zavg": to vertically average an ocean field over a depth range
    - ".isel(...)" / ".sel(...)": to slice the variable before anything else

    An optional "alias ~ " prefix renames the result.

    Examples::

        'TS:ann:gm'
        'GMSST ~ SST:ann:gm'
        'TEMP.isel(z_t=0):ann:gm'
        'T|regrid(1,1)|plev(500):climo'
        'TEMP|zavg(0,1000):ann:gm'
        'SST.sel(lat=slice(-5,5)):ann'

    Arguments are parsed into structured form (`regrid_args`, `slicing_kwargs`,
    `plev_levels`, ...) so callers never have to `eval` the string. The raw
    fragments remain available as `slicing`, `regrid`, `plev` and `zavg` for
    display. Note `:` is the field separator and cannot appear inside arguments.

    Attributes:
        sentence (str): the spell with any alias stripped
        alias (str): the requested output name, or None
        vn (str): the bare variable name, with any slicing call removed
        vn_raw (str): the variable fragment as written, slicing included
        ann_method (str): annualization method, or None
        sa_method (str): spatial-average method, or None
        slicing (str): e.g. ``'isel(z_t=0)'``, or None
        slicing_method (str): ``'isel'`` or ``'sel'``, or None
        slicing_args (tuple), slicing_kwargs (dict)
        regrid (str), regrid_args (tuple), regrid_kwargs (dict)
        zavg (str), zavg_args (tuple), zavg_kwargs (dict)
        plev (str): e.g. ``'plev(500)'``, or None
        plev_levels (list): requested pressure levels, or None for a bare ``|plev``
    '''

    #: the spatial-average methods `Timeseries.calc` knows how to apply
    SA_METHODS = ('gm', 'nhm', 'shm', 'zm', 'gs', 'nhs', 'shs', 'somin', 'yz')

    def __init__(self, sentence: str):
        if not isinstance(sentence, str):
            raise TypeError(f'A spell must be a string; got {type(sentence).__name__}.')

        self.sentence = sentence

        self.alias = None
        self.vn = None
        self.vn_raw = None
        self.ann_method = None
        self.sa_method = None

        self.slicing = None
        self.slicing_method = None
        self.slicing_args = ()
        self.slicing_kwargs = {}

        self.regrid = None
        self.regrid_args = ()
        self.regrid_kwargs = {}

        self.plev = None
        self.plev_levels = None

        self.zavg = None
        self.zavg_args = ()
        self.zavg_kwargs = {}

        self.parse_alias()
        self.parse_sentence()
        self.parse_slicing()
        self.parse_regrid()
        self.parse_plev()
        self.parse_zavg()

    def __repr__(self):
        bits = [f'vn={self.vn!r}']
        for name in ('alias', 'ann_method', 'sa_method', 'slicing', 'regrid', 'plev', 'zavg'):
            value = getattr(self, name)
            if value is not None:
                bits.append(f'{name}={value!r}')
        return f'Spell({", ".join(bits)})'

    def parse_alias(self):
        if '~' in self.sentence:
            head, _, tail = self.sentence.partition('~')
            self.alias = head.strip()
            self.sentence = tail.strip()
            if self.alias == '':
                raise ValueError(f'Empty alias in spell `{self.sentence}`.')

    def parse_sentence(self):
        ''' Split the ``vn[:ann_method[:sa_method]]`` skeleton '''
        elements = [e.strip() for e in self.sentence.split(':')]

        if len(elements) > 3:
            raise ValueError(
                f'Cannot parse spell `{self.sentence}`: expected at most 3 '
                f'colon-separated fields (vn:ann_method:sa_method), got {len(elements)}. '
                'Note that `:` is the field separator and cannot appear inside arguments.'
            )

        self.vn_raw = elements[0]
        if len(elements) >= 2:
            self.ann_method = elements[1] or None
        if len(elements) == 3:
            self.sa_method = elements[2] or None

        if self.vn_raw == '':
            raise ValueError(f'Cannot parse spell `{self.sentence}`: no variable name.')

        if self.sa_method is not None and self.sa_method not in self.SA_METHODS:
            raise ValueError(
                f'Unknown spatial average method `{self.sa_method}` in spell '
                f'`{self.sentence}`. Options: {", ".join(self.SA_METHODS)}.'
            )

        # the modifiers live after the first `|`; the variable fragment precedes it
        self.vn_raw = self.vn_raw.split('|')[0].strip()
        self.vn = self.vn_raw

    def parse_slicing(self):
        ''' Extract a leading ``.isel(...)`` / ``.sel(...)`` on the variable '''
        for method in ('isel', 'sel'):
            found = find_call(self.vn_raw, f'.{method}')
            if found is None:
                continue

            start, end, argstring = found
            self.slicing_method = method
            self.slicing = f'{method}({argstring or ""})'
            self.slicing_args, self.slicing_kwargs = parse_call_args(argstring, self.slicing)
            # strip by position rather than splitting on '.', so a dotted variable
            # name such as `NINO3.4` survives
            self.vn = (self.vn_raw[:start] + self.vn_raw[end:]).strip()
            if self.vn == '':
                raise ValueError(f'Cannot parse spell `{self.sentence}`: no variable name.')
            return

    def _parse_modifier(self, name):
        ''' Shared handling for the pipe-delimited modifiers '''
        if f'|{name}' not in self.sentence:
            return None
        found = find_call(self.sentence, name)
        if found is None:
            return None
        _, _, argstring = found
        raw = f'{name}({argstring})' if argstring is not None else f'{name}()'
        args, kwargs = parse_call_args(argstring, raw)
        return raw, args, kwargs

    def parse_regrid(self):
        parsed = self._parse_modifier('regrid')
        if parsed is not None:
            self.regrid, self.regrid_args, self.regrid_kwargs = parsed

    def parse_zavg(self):
        parsed = self._parse_modifier('zavg')
        if parsed is not None:
            self.zavg, self.zavg_args, self.zavg_kwargs = parsed

    def parse_plev(self):
        parsed = self._parse_modifier('plev')
        if parsed is None:
            return

        raw, args, kwargs = parsed
        self.plev = raw
        if kwargs:
            raise ValueError(f'`plev` takes positional levels only; got {sorted(kwargs)}.')

        if len(args) == 0:
            self.plev_levels = None           # bare `|plev`: use the defaults
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            self.plev_levels = list(args[0])  # `plev([500, 850])`
        else:
            self.plev_levels = list(args)     # `plev(500)` or `plev(500, 850)`
