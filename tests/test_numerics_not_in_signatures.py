"""No numerical choice is a keyword of a src function: `Numerics` holds it.

One immutable object, built once by the driver and handed down, defines the grids, quadratures,
fit and Davidson settings of a run; a function that takes `ntau`, `counts`, `conv_tol` ... as its
own keyword is a second way in, and the two disagree the moment a caller passes one and the object
holds another. Read off the source by AST over the GW-BSE, gradient and property packages: no
parameter of a function, method or lambda carries one of `SIGNATURE_NAMES`, except

  * `derive`, the one sanctioned way to change a field, and `src/Base/numerics/numerics.py`;
  * the rows of `numerics_signature_debt.json`, the signatures not yet converted. The list can only
    shrink: a signature that gains a name fails, and an entry whose name has left its signature
    fails until the row is deleted.

The scanner is shown on planted sources to fail on the pattern it forbids and pass on the other.
"""
import ast
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

#: The packages the object covers: GW-BSE, gradients and properties, and what they build on.
PACKAGES = ('src/gradients', 'src/properties', 'src/Base',
            'src/SingleReference/GW', 'src/SingleReference/LinearResponse',
            'src/SingleReference/BSE',
            'src/photophysics', 'src/environments', 'src/fragments',
            'src/SingleReference/common')

#: Parameter names that are a numerical choice of a run: grids, quadratures, the ISDF fit,
#: the Davidson, the memory and layout knobs.
SIGNATURE_NAMES = frozenset(
    'ntau ntau_gw ntau_w nfreq nfreq_cd nfreq_rpa ntau_rpa npade e_min_below_gap '
    'tau_target n_poles sop_stride conv_tol conv_tol_grad max_cycle nroots guess_factor '
    'preconditioner tile_gb fit_block fit_gb block_memory_gb counts radii n_start '
    'grid_accuracy l_max_second pair_tol regularization orbital_tol'.split())

#: Functions and modules exempt by construction.
EXEMPT_FUNCTIONS = frozenset({'derive'})
EXEMPT_MODULES = frozenset({'src/Base/numerics/numerics.py'})

DEBT = json.loads((REPO / 'tests' / 'numerics_signature_debt.json').read_text())


def numerical_parameters(tree):
    """{'function': {names}} of every function or lambda parameter in `SIGNATURE_NAMES`."""
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        name = getattr(node, 'name', '<lambda>')
        if name in EXEMPT_FUNCTIONS:
            continue
        args = node.args
        names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        hit = names & SIGNATURE_NAMES
        if hit:
            found.setdefault(name, set()).update(hit)
    return found


def violations():
    """{'path::function': [names]} over every scanned module."""
    out = {}
    for package in PACKAGES:
        for path in sorted((REPO / package).rglob('*.py')):
            rel = str(path.relative_to(REPO))
            if rel in EXEMPT_MODULES:
                continue
            for name, hit in numerical_parameters(ast.parse(path.read_text())).items():
                out[f'{rel}::{name}'] = sorted(hit)
    return out


def test_no_new_numerical_keyword_in_a_signature():
    new = {k: v for k, v in violations().items()
           if not set(v) <= set(DEBT.get(k, ()))}
    assert not new, (
        'a src signature takes a numerical choice that `Numerics` holds; read it from the '
        f'object instead of adding a keyword: {new}')


def test_the_debt_list_holds_only_signatures_that_still_carry_the_name():
    found = violations()
    stale = {k: sorted(set(v) - set(found.get(k, ()))) for k, v in DEBT.items()
             if set(v) - set(found.get(k, ()))}
    assert not stale, f'converted signatures still listed; delete these rows: {stale}'


def test_the_scanner_fails_on_a_planted_keyword_and_passes_on_the_object():
    planted = ast.parse('def solve(mf, ntau=12, *, tile_gb=4.0):\n    pass\n'
                        'g = lambda x, conv_tol=1e-8: x\n')
    assert numerical_parameters(planted) == {'solve': {'ntau', 'tile_gb'},
                                             '<lambda>': {'conv_tol'}}
    clean = ast.parse('def solve(mf, numerics=None, comm=None):\n    pass\n'
                      'def derive(self, reason, **changes):\n    pass\n'
                      'def other(self, ntau):\n    pass\n')
    assert numerical_parameters(clean) == {'other': {'ntau'}}
    exempt = ast.parse('def derive(self, ntau=3):\n    pass\n')
    assert numerical_parameters(exempt) == {}
