"""The downfolding package keeps its general layer free of method code.

`src/MultiReference/Downfolding/` is a method-agnostic layer (`terms`,
`partition`, `two_body`, `one_body`, `energy`, and the driver
`effective_hamiltonian`) plus one file per low-level method under `methods/`.
The two sides meet only in `DownfoldingTerms`, and the driver reaches the
methods only through the `LOW_LEVEL` registry. Both rules are read off the
source by AST, so no module is imported to check them. A module that does
not exist yet is skipped, so the test tightens as files land.
"""
import ast
import pathlib

import pytest

from src.MultiReference.Downfolding.terms import choice

PACKAGE = (pathlib.Path(__file__).resolve().parent.parent
           / 'src' / 'MultiReference' / 'Downfolding')
GENERAL = ['terms', 'partition', 'two_body', 'one_body', 'energy']
FORBIDDEN = ('src.SingleReference.LinearResponse', 'src.SingleReference.GW',
             'src.MultiReference.Downfolding.methods')
METHODS = 'src.MultiReference.Downfolding.methods'


def _imports(path):
    """(module, names) of every import in `path`; names is [] for a plain
    `import x`. Relative imports are resolved against the package."""
    package = ['src', 'MultiReference', 'Downfolding']
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                base = package[:len(package) - (node.level - 1)]
                module = '.'.join(base + ([module] if module else []))
            found.append((module, [a.name for a in node.names]))
    return found


def _under(module, prefix):
    return module == prefix or module.startswith(prefix + '.')


def test_general_layer_imports_no_method_code():
    checked = 0
    for name in GENERAL:
        path = PACKAGE / f'{name}.py'
        if not path.exists():
            continue
        checked += 1
        for module, names in _imports(path):
            # `from src.MultiReference.Downfolding import methods` too
            targets = [module] + [f'{module}.{n}' for n in names]
            for target in targets:
                for banned in FORBIDDEN:
                    assert not _under(target, banned), (
                        f'{name}.py imports {target}, method code')
    assert checked >= 1  # terms.py at the least


def test_driver_imports_only_the_registry():
    path = PACKAGE / 'effective_hamiltonian.py'
    if not path.exists():
        pytest.skip('effective_hamiltonian.py is not written yet')
    for module, names in _imports(path):
        assert not _under(module, 'src.SingleReference.LinearResponse')
        assert not _under(module, 'src.SingleReference.GW')
        if _under(module, METHODS):
            assert module == METHODS and names == ['LOW_LEVEL'], (
                f'the driver takes only LOW_LEVEL from methods, not '
                f'{module}: {names}')


def test_choice_refuses_and_normalizes():
    assert choice('Imag_Frequency', ('casida', 'imagfrequency'),
                  'mode') == 'imagfrequency'
    # a choice spelt with '_' is matched in either spelling and returned as listed
    for value in ('crpa_ph', 'CRPA-PH'):
        assert choice(value, ('crpa', 'crpa_ph'), 'screening') == 'crpa_ph'
    assert choice('Space_Time', ('space-time',), 'mode') == 'space-time'
    with pytest.raises(ValueError, match="mode"):
        choice('x', ('casida',), 'mode')
