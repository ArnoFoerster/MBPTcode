"""Every dotted name the source text gives in backticks resolves in this tree.

A comment or docstring that names `gradients.gwbse.state_following` or
`Base.factorization.separable_ri.fit_rows` points a reader at code; after a
module moves, such a pointer is the one thing no import checks. Every
backticked `a.b.c` whose first part is `src` or a top-level package
(Base, Solvers, SingleReference, gradients, properties, environments,
fragments, photophysics, CodeGeneration, MultiReference, autograd_oracle) is
resolved: the longest prefix that is a module, then each remaining part a
name that module (or the class before it) defines. The scanner is shown to
fail on a planted stale name. Scanned: the packages listed in SCANNED.
"""
import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOTS = ('Base', 'Solvers', 'SingleReference', 'gradients', 'properties',
         'environments', 'fragments', 'photophysics', 'CodeGeneration',
         'MultiReference', 'autograd_oracle')
TICKED = re.compile(r'`(?:src\.)?((?:' + '|'.join(ROOTS) + r')(?:\.[A-Za-z_]\w*)+)')
_DEFS = {}


def defined(path):
    """{top-level name: {its class members}} of a module file."""
    if path not in _DEFS:
        tree = ast.parse(path.read_text())
        out = {}
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                out[n.name] = {m.name for m in n.body
                               if isinstance(m, (ast.FunctionDef, ast.ClassDef))} | {
                    t.id for m in n.body if isinstance(m, (ast.Assign, ast.AnnAssign))
                    for t in (m.targets if isinstance(m, ast.Assign) else [m.target])
                    if isinstance(t, ast.Name)}
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[n.name] = set()
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        out[t.id] = set()
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    out[(a.asname or a.name).split('.')[0]] = set()
        _DEFS[path] = out
    return _DEFS[path]


def resolves(dotted, src=REPO / 'src'):
    parts = dotted.split('.')
    for k in range(len(parts), 0, -1):
        base = src.joinpath(*parts[:k])
        module = base.with_suffix('.py')
        if base.is_dir() and (base / '__init__.py').exists():
            module = base / '__init__.py'
            if k == len(parts):
                return True
        elif not module.exists():
            continue
        rest = parts[k:]
        if not rest:
            return True
        names = defined(module)
        if rest[0] not in names:
            return base.is_dir() and (base / f'{rest[0]}.py').exists()
        return len(rest) == 1 or rest[1] in names[rest[0]] or not names[rest[0]]
    return False


def unresolved(text):
    return sorted({m for m in TICKED.findall(text) if not resolves(m)})


def test_the_scanner_fails_on_a_planted_stale_name():
    assert unresolved('see `Base.factorization.separable_ri.fit_rows`') == []
    assert unresolved('see `Base.separable_ri.fit_rows`') == ['Base.separable_ri.fit_rows']
    assert unresolved('see `gradients.no_such_module`') == ['gradients.no_such_module']


#: The packages whose text is held to it (the foundation and the method
#: packages built on it).
SCANNED = ('Base', 'Solvers', 'SingleReference/GW', 'SingleReference/BSE',
           'SingleReference/LinearResponse', 'SingleReference/Periodic',
           'SingleReference/common')
FILES = sorted(p for package in SCANNED for p in (REPO / 'src' / package).rglob('*.py'))


@pytest.mark.parametrize('path', FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_every_backticked_name_resolves(path):
    assert unresolved(path.read_text()) == []
