#!/usr/bin/env python
''' Generate docsrc/api.md from the x4c source docstrings.

mystmd (Jupyter Book 2) has no autodoc directive like classic Sphinx, so the API
reference is pre-rendered to plain Markdown by introspecting x4c with `griffe`
and re-running after code changes: `python docsrc/scripts/gen_api_docs.py`.

Docstrings in x4c are a mix of Google style ("Args:") and NumPy style
("Parameters\\n----------"); `parse_docstring` picks the right griffe parser
per-function by sniffing for a NumPy-style underline.
'''
import re
from pathlib import Path

import griffe

DOCSRC = Path(__file__).resolve().parent.parent

# (section title, module name, [(kind, member name), ...])
TARGETS = [
    ('Core Features', 'core', [
        ('function', 'load_dataset'),
        ('function', 'open_dataset'),
        ('function', 'open_mfdataset'),
        ('class', 'XDataset'),
        ('class', 'XDataArray'),
    ]),
    ('CESM Postprocessing', 'case', [
        ('class', 'History'),
    ]),
    ('CESM Diagnostics', 'case', [
        ('class', 'Timeseries'),
        ('class', 'Logs'),
    ]),
    ('The Spell Mini-Language', 'spell', [
        ('class', 'Spell'),
    ]),
    ('Derived-Variable Registry', 'diags', [
        ('function', 'F'),
        ('class', 'Registry'),
    ]),
    ('Visualization Helpers', 'visual', [
        ('function', 'set_style'),
        ('function', 'subplots'),
        ('function', 'savefig'),
        ('function', 'showfig'),
        ('function', 'closefig'),
        ('function', 'add_annotation'),
        ('function', 'infer_cmap'),
    ]),
    ('Utilities', 'utils', [
        ('function', 'fetch_sample_data'),
        ('function', 'cache_dir'),
    ]),
]

NUMPY_UNDERLINE = re.compile(r'\n *-{3,}\s*\n')


def parse_docstring(ds):
    if ds is None:
        return []
    style = 'numpy' if NUMPY_UNDERLINE.search(ds.value) else 'google'
    return ds.parse(style)


MAX_DEFAULT_LEN = 60


def render_signature(member):
    parts = []
    for p in member.parameters:
        if p.name == 'self':
            continue
        name = p.name
        if p.kind is not None and 'variadic keyword' in p.kind.value:
            name = f'**{name}'
        elif p.kind is not None and 'variadic positional' in p.kind.value:
            name = f'*{name}'
        elif p.default is not None:
            default = str(p.default)
            if len(default) > MAX_DEFAULT_LEN:
                opener = default[0] if default[0] in '[({' else ''
                default = f'{opener}...{default[-1]}' if opener else '...'
            name = f'{name}={default}'
        parts.append(name)
    return f"{member.name}({', '.join(parts)})"


def format_bullet(prefix, desc):
    ''' `- prefix: line1\\n  line2\\n  line3` -- continuation lines indented so
    CommonMark keeps them inside the same list item instead of starting a new
    paragraph or a stray unindented line.
    '''
    lines = desc.splitlines() or ['']
    out = f'- {prefix}{lines[0]}' if prefix else f'- {lines[0]}'
    for line in lines[1:]:
        out += f'\n  {line}'
    return out


def render_sections(sections, level):
    out = []
    for section in sections:
        kind = section.kind.value
        if kind == 'text':
            out.append(section.value.strip())
        elif kind in ('parameters', 'attributes'):
            out.append('**Parameters**' if kind == 'parameters' else '**Attributes**')
            for p in section.value:
                ann = f' (`{p.annotation}`)' if p.annotation else ''
                out.append(format_bullet(f'`{p.name}`{ann}: ', p.description))
        elif kind in ('returns', 'raises'):
            heading = '**Returns**' if kind == 'returns' else '**Raises**'
            out.append(heading)
            bullets = []
            for item in section.value:
                ann = str(item.annotation) if item.annotation else ''
                name = getattr(item, 'name', '') or ''
                # Griffe's Google-style parser splits same-indent continuation lines
                # into separate unnamed/untyped items -- rejoin those into the prior bullet.
                if not ann and not name and bullets:
                    bullets[-1] = f'{bullets[-1]} {item.description}'
                    continue
                label = f'`{name}` (`{ann}`): ' if name and ann else (f'`{ann}`: ' if ann else '')
                bullets.append(f'{label}{item.description}')
            for b in bullets:
                out.append(format_bullet('', b))
        elif kind == 'admonition':
            title = section.value.annotation.strip().capitalize()
            out.append(f'**{title}**')
            out.append(section.value.description)
        else:
            # examples, etc. -- keep as prose rather than dropping content
            out.append(section.value if isinstance(section.value, str) else str(section.value))
    return '\n\n'.join(out)


def render_function(fn, level, bullet=False):
    heading = '#' * level
    # The heading text drives the page's auto-generated Contents outline, so it stays
    # to the bare name -- the full signature (with every argument/default) moves into
    # the body as its own line instead of bloating that outline.
    title = f'• `{fn.name}`' if bullet else f'`{fn.name}`'
    lines = [f'{heading} {title}', '', f'`{render_signature(fn)}`', '']
    body = render_sections(parse_docstring(fn.docstring), level)
    if body:
        lines.append(body)
        lines.append('')
    return '\n'.join(lines)


def render_class(cls, level):
    heading = '#' * level
    lines = [f'{heading} `{cls.name}`', '']
    body = render_sections(parse_docstring(cls.docstring), level)
    if body:
        lines.append(body)
        lines.append('')

    methods = [
        m for name, m in cls.members.items()
        if not name.startswith('_') and getattr(m, 'kind', None) is not None
        and m.kind.value == 'function' and m.docstring is not None
    ]
    for i, m in enumerate(methods):
        if i > 0:
            lines.append('---')
            lines.append('')
        lines.append(render_function(m, level + 1, bullet=True))
    return '\n'.join(lines)


def main():
    pkg = griffe.load('x4c')
    out = ['# API Reference', '']
    for i, (title, modname, members) in enumerate(TARGETS):
        if i > 0:
            out.append('---')
            out.append('')
        out.append(f'## {title}')
        out.append('')
        mod = pkg[modname]
        for j, (kind, name) in enumerate(members):
            if j > 0:
                out.append('---')
                out.append('')
            member = mod[name]
            if kind == 'class':
                out.append(render_class(member, 3))
            else:
                out.append(render_function(member, 3))
    text = '\n'.join(out)
    # Docstrings carry Sphinx cross-reference roles (:func:`x`, :class:`x`, ...) left
    # over from the old sphinx.ext.autodoc build; mystmd has no such role, so drop the
    # role prefix and keep the target as an inline code span.
    text = re.sub(r':(?:func|class|meth|mod|data|attr|obj):`([^`]*)`', r'`\1`', text)
    text = re.sub(r'\n{3,}', '\n\n', text) + '\n'
    out_path = DOCSRC / 'api.md'
    out_path.write_text(text)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
