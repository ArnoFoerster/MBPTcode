"""The downfolding package keeps its general layer free of method code.

`src/EffectiveHamiltonian/Downfolding/` is a method-agnostic layer (`terms`,
`partition`, `two_body`, `one_body`, `energy`, and the driver
`effective_hamiltonian`) plus one file per low-level method under `methods/`.
The two sides meet only in `DownfoldingTerms`, and the driver reaches the
methods only through the `LOW_LEVEL` registry. Both rules are read off the
source by AST, so no module is imported to check them. Every import is
checked by its module and by each module.name it takes (`from . import
methods` reaches the methods as surely as `from .methods import rpa`). Every
module named here exists: a renamed or missing one fails, it is not skipped.
"""
import ast
import pathlib

import pytest

from src.EffectiveHamiltonian.Downfolding.options import choice

PACKAGE = (pathlib.Path(__file__).resolve().parent.parent
           / 'src' / 'EffectiveHamiltonian' / 'Downfolding')
GENERAL = ['low_level', 'options', 'masks', 'two_body', 'one_body',
           'energy']
KERNELS = ('src.SingleReference.LinearResponse', 'src.SingleReference.GW')
METHODS = 'src.EffectiveHamiltonian.Downfolding.methods'
FORBIDDEN = KERNELS + (METHODS,)


def _imports(path):
    """(module, names) of every import in `path`; names is [] for a plain
    `import x`. Relative imports are resolved against the package."""
    package = ['src', 'EffectiveHamiltonian', 'Downfolding']
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


def _targets(module, names):
    """The module an import names and each module.name it takes: `from
    src.EffectiveHamiltonian.Downfolding import methods` reaches the methods
    package through its name, not its module."""
    return [module] + [f'{module}.{n}' for n in names]


def _source(name):
    path = PACKAGE / f'{name}.py'
    assert path.exists(), f'{path} is gone: rename it here too'
    return path


def test_general_layer_imports_no_method_code():
    for name in GENERAL:
        for module, names in _imports(_source(name)):
            for target in _targets(module, names):
                for banned in FORBIDDEN:
                    assert not _under(target, banned), (
                        f'{name}.py imports {target}, method code')


def test_driver_imports_only_the_registry():
    for module, names in _imports(_source('effective_hamiltonian')):
        for target in _targets(module, names):
            for banned in KERNELS:
                assert not _under(target, banned), (
                    f'the driver imports {target}, method code')
            if _under(target, METHODS):
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
