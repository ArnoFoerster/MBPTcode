"""The ISDF grid a test runs on when the grid is not what it tests: G1.

The library names no default grid (`separable_ri.require_grid`), so a test
asks for one. G1 is the lowest validated accuracy level (< 8 meV on the
three lowest BSE roots at its basis); a basis with no validated level has no
G1 and is refused by `resolve_isdf_grid`.
"""
from src.gradients.rpa.chain_settings import with_setting
from src.Base.numerics.numerics import Numerics
from src.Base.factorization.separable_ri import resolve_isdf_grid

#: The validated level tests run on.
TEST_GRID_LEVEL = 'G1'


def counts_at(obj, level):
    """The shell counts of `level` at the basis of `obj` (a molecule, a mean
    field or a basis name)."""
    basis = obj if isinstance(obj, str) else getattr(obj, 'mol', obj).basis
    return resolve_isdf_grid(level, str(basis))[0]


def g1(obj):
    """The G1 shell counts at the basis of `obj`."""
    return counts_at(obj, TEST_GRID_LEVEL)


def g1_numerics(obj, numerics=None):
    """`numerics` (default `Numerics()`) on the G1 grid, unless it already
    names counts or radii."""
    numerics = Numerics() if numerics is None else numerics
    if numerics.isdf.counts is not None or numerics.isdf.radii is not None:
        return numerics
    return with_setting(numerics, 'isdf.counts', g1(obj))


#: Keywords that already name the grid or hand in a built factorization.
GRID_KEYWORDS = ('counts', 'grid_accuracy', 'radii', 'factors', 'factorization')


def g1_kw(obj, kw):
    """`kw` with the G1 counts added, unless it already names a grid or
    hands in a built factorization (a chain's keywords as well as a
    route's)."""
    if any(kw.get(k) is not None for k in GRID_KEYWORDS):
        return dict(kw)
    return dict(kw, counts=g1(obj))


def g1_numerics_kw(obj, kw):
    """`kw` for a chain: its `numerics` on the G1 grid, unless a built
    factorization is handed in."""
    if kw.get('factorization') is not None:
        return dict(kw)
    return dict(kw, numerics=g1_numerics(obj, kw.get('numerics')))


def g1_route(obj, mode):
    """The grid keyword of a QP or downfolding route: the G1 counts on the
    space-time route, which builds separable factors, nothing elsewhere."""
    return {'counts': g1(obj)} if mode == 'space-time' else {}
