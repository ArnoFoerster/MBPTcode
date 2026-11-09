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
PACKAGES = ('src/gradients', 'src/properties', 'src/Base', 'src/Solvers',
            'src/SingleReference/GW', 'src/SingleReference/LinearResponse',
            'src/SingleReference/BSE', 'src/SingleReference/Periodic',
            'src/photophysics', 'src/environments', 'src/fragments',
            'src/SingleReference/common')

#: Parameter names that are a numerical choice of a run: grids, quadratures, the ISDF fit,
#: the Davidson, the self-consistent loops' tolerances and mixing, the memory and layout
#: knobs.
SIGNATURE_NAMES = frozenset(
    'ntau ntau_gw ntau_w nfreq nfreq_cd nfreq_rpa ntau_rpa npade e_min_below_gap '
    'tau_target n_poles sop_stride conv_tol conv_tol_grad max_cycle nroots guess_factor '
    'preconditioner tile_gb fit_block fit_gb block_memory_gb counts radii n_start '
    'grid_accuracy l_max_second pair_tol regularization orbital_tol '
    'tol max_iter tile_memory_gb stride npoints_max w0 w0_cd damping diis_size dm_tol '
    'threshold'.split())

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


#: The orbital-response (Z-vector) solves and their entry points: each takes the run's `Numerics`
#: with no default, so a caller that hands none is refused instead of solving on defaults the run
#: did not declare.
Z_VECTOR_FUNCTIONS = {
    'src/Base/pyscf/cphf.py': ('solve_operator', 'solve'),
    'src/gradients/multipliers.py': ('solve_orbital_multipliers',),
    'src/gradients/mean_field_response.py': ('solve_lambda', 'eps_chain_gradient',
                                             'relaxed_density_mo', 'relaxed_multiplier'),
    'src/gradients/grad_engine.py': ('solve_multipliers', 'kohn_sham_correlation_gradients',
                                     'correlation_gradients', 'correlation_gradient'),
    'src/gradients/orbital_coupling.py': ('z_vector_contraction', 'configuration_coupling',
                                          'ov_coupling'),
    'src/gradients/gwbse/derivative_coupling.py': ('ground_state_coupling',),
    'src/gradients/scf_loop_response.py': ('loop_density_response',),
    'src/SingleReference/DensityMatrix/density_matrix.py': ('solve_cphf_relaxation',
                                                            'solve_cphf_relaxation_uhf'),
    'src/SingleReference/GW/density_matrix.py': ('relax_gw_density',),
}


def defaulted_numerics(tree, names):
    """{name: 'missing' | 'defaulted'} of the functions in `names` whose `numerics` parameter is
    absent or carries a default; the ones that require it are left out."""
    out = {name: 'missing' for name in names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        a = node.args
        positional = a.posonlyargs + a.args
        defaults = dict(zip([p.arg for p in positional][len(positional) - len(a.defaults):],
                            a.defaults))
        defaults.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
        if 'numerics' in {p.arg for p in positional + a.kwonlyargs}:
            if 'numerics' in defaults:
                out[node.name] = 'defaulted'
            else:
                out.pop(node.name)
    return out


def test_the_z_vector_solves_require_numerics():
    bad = {}
    for path, names in Z_VECTOR_FUNCTIONS.items():
        for name, why in defaulted_numerics(ast.parse((REPO / path).read_text()),
                                            set(names)).items():
            bad[f'{path}::{name}'] = why
    assert not bad, f'a Z-vector solve takes no Numerics or defaults it: {bad}'


def test_the_z_vector_scan_fails_on_a_default_and_passes_on_a_required_argument():
    planted = ast.parse('def a(x, numerics=None):\n    pass\n'
                        'def b(x, *, numerics=None):\n    pass\n'
                        'def c(x, numerics, flags=None):\n    pass\n'
                        'def d(x, y=1, *, numerics):\n    pass\n')
    assert defaulted_numerics(planted, {'a', 'b', 'c', 'd', 'e'}) == {
        'a': 'defaulted', 'b': 'defaulted', 'e': 'missing'}
