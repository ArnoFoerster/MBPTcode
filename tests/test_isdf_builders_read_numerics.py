"""The ISDF builders read their choices off the one `Numerics`.

The fit's four settings -- the test set's `l_max_second`, the pair
screen `pair_tol`, the Tikhonov `regularization` and the working set
`block_memory_gb` -- are not keywords of the functions in `separable_ri`,
`isdf_jk`, `skeleton_tiles`, `isdf_derivatives` and `space_time` with defaults
of their own, nor of `FrozenFactorization`. The builders take `numerics` and
read the values through `separable_ri.fit_settings` (the defaults in one place,
constants.py),
`FrozenFactorization` takes the object and refuses fit choices its chains'
adjoints do not differentiate, the auxiliary pairing `<basis>-ri` is spelled
once (`paired_auxbasis`), and no ISDF module spells a constant of its own
(`_METRIC_EIG_TOL`, the `metric_rcond` and `0.33` literals, the
`DEFAULT_PAIR_TOL`/`DEFAULT_REGULARIZATION` copies).

Water/cc-pVDZ on the G1 grid; nothing is solved beyond one fit's D and F.
"""
import ast
import pathlib

import pytest
from pyscf import df, gto

from src.Base.numerics import constants
from src.Base.factorization import isdf_jk, separable_ri
from src.Base.numerics.numerics import IsdfNumerics, Numerics
from src.Base.factorization.separable_ri import atomic_grid, build_D_F, fit_settings
from src.Base.factorization.separable_ri import molecular_points_covariant
from src.gradients.rpa.factor_chain import FrozenFactorization

REPO = pathlib.Path(__file__).resolve().parents[1]
QUARTET = ('l_max_second', 'pair_tol', 'regularization', 'block_memory_gb')
#: The modules of the ISDF fit and its adjoints.
FIT_MODULES = ('src/Base/factorization/separable_ri.py', 'src/Base/factorization/isdf_jk.py',
               'src/gradients/skeleton_tiles.py', 'src/Base/factorization/distributed_isdf_jk.py',
               'src/gradients/isdf_derivatives.py',
               'src/gradients/rpa/factor_chain.py')
H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def takers(paths, names):
    """Public functions of `paths` taking one of `names` as a keyword with a
    default: a choice a caller could pass. A kernel handed the resolved value
    (`ao_blocks`) takes it as a required argument from `fit_settings`."""
    out = []
    for path in paths:
        for node in ast.walk(ast.parse((REPO / path).read_text())):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                args = node.args
                with_default = ([a.arg for a in args.args][
                    len(args.args) - len(args.defaults):]
                    + [a.arg for a, d in zip(args.kwonlyargs,
                                             args.kw_defaults) if d])
                taken = set(with_default) & set(names)
                if taken:
                    out.append(f'{path}:{node.name} {sorted(taken)}')
    return out


def test_no_fit_builder_takes_the_quartet():
    assert not takers(FIT_MODULES, QUARTET)
    assert not takers(['src/SingleReference/GW/space_time.py'],
                      ('l_max_second', 'pair_tol', 'regularization'))
    sig = ast.parse((REPO / 'src/SingleReference/GW/space_time.py').read_text())
    factors = [n for n in ast.walk(sig) if isinstance(n, ast.FunctionDef)
               and n.name == 'separable_factors']
    assert 'block_memory_gb' not in {a.arg for a in factors[0].args.args}


def test_the_defaults_are_one():
    isdf = IsdfNumerics()
    assert (isdf.l_max_second, isdf.pair_tol, isdf.regularization) == (
        constants.ISDF_FIT_L_MAX_SECOND, constants.ISDF_FIT_PAIR_TOL,
        constants.ISDF_FIT_REGULARIZATION)
    assert fit_settings(None, *QUARTET) == (
        isdf.l_max_second, isdf.pair_tol, isdf.regularization,
        constants.ISDF_TILE_GB)
    for gone in ('DEFAULT_PAIR_TOL', 'DEFAULT_REGULARIZATION'):
        assert not hasattr(separable_ri, gone), gone


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
    counts = separable_ri.resolve_isdf_grid('G1', 'cc-pvdz')[0]
    radii, origins = {}, {}
    for el in ('H', 'O'):
        radii[el], origins[el] = atomic_grid(el, 'cc-pvdz', 'cc-pvdz-ri',
                                             counts)
    return mol, auxmol, molecular_points_covariant(
        mol, radii, origin_by_element=origins), counts


def test_a_choice_handed_in_reaches_the_fit(water):
    """A coarser pair screen keeps fewer test co-densities."""
    mol, auxmol, coords, _ = water
    D, _ = build_D_F(mol, auxmol, coords)
    coarse, _ = build_D_F(mol, auxmol, coords,
                          Numerics(isdf={'pair_tol': 1e-2}))
    assert coarse.shape[1] < D.shape[1]


def test_the_factorization_takes_the_object(water):
    mol, _, _, counts = water
    numerics = Numerics(isdf={'counts': counts})
    fz = FrozenFactorization(mol, numerics)
    assert fz.counts == counts and fz.fit == 'replicated' and not fz.sliced
    with pytest.raises(NotImplementedError, match='default fit'):
        FrozenFactorization(mol, numerics.derive(
            'a looser screen', isdf={'pair_tol': 1e-4}))
    with pytest.raises(NotImplementedError, match='default fit'):
        fz.require_match(numerics.derive('another shift',
                                         isdf={'regularization': 1e-6}))
    fz.require_match(numerics)


def test_the_pairing_rule_is_spelled_once():
    """`<basis>-ri` is `paired_auxbasis`; the CP2K tier names
    `<name>-ri-<Delta-I>` are names, not the rule."""
    spelled = []
    for path in sorted((REPO / 'src').rglob('*.py')):
        rel = str(path.relative_to(REPO))
        if any(part in rel for part in ('/ADC/', '/CC/', '/DensityMatrix/',
                                        '/Periodic/', '/EpsteinNesbet/',
                                        'cp2k_basis.py')):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            joined = (isinstance(node, ast.JoinedStr) and node.values
                      and isinstance(node.values[-1], ast.Constant)
                      and node.values[-1].value == '-ri')
            added = (isinstance(node, ast.BinOp)
                     and isinstance(node.op, ast.Add)
                     and isinstance(node.right, ast.Constant)
                     and node.right.value == '-ri')
            if joined or added:
                spelled.append(rel)
    assert spelled == ['src/Base/basis/ri_fallback.py'], spelled


def test_no_isdf_duplicate_of_a_constant():
    """The metric's root floor and pseudo-inverse cut and the K kernel's
    memory share are the constants, not literals beside them."""
    assert not hasattr(isdf_jk, '_METRIC_EIG_TOL')
    text = (REPO / 'src/Base/factorization/isdf_jk.py').read_text()
    assert 'metric_rcond' not in text and '0.33' not in text
    assert 'numerics' in isdf_jk.ISDFJK.__init__.__code__.co_varnames
