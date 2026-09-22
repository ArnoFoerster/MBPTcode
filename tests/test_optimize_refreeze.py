"""The refreeze loop on BOTH optimizers, and which gradient `grad_max` names.

WHAT BREAKS IF THIS FAILS. A geometry optimization freezes the discrete
conventions -- the quasiparticle set, the frame, the interpolation layout --
at the starting geometry and keeps them, so the minimum it reports belongs to
the surface chosen at the START. `refreeze` rebuilds them at the converged
geometry and optimizes again, and how far the geometry then moves is the only
measurement of that approximation there is. Only the Cartesian optimizer had
the loop, `relax(engine='auto')` routes to geomeTRIC whenever it is installed,
and so every record written through the preferred engine carried
`refreeze_shift: None` -- an unmeasured error bar that reads exactly like a
measured zero. These gates hold the loop on both engines and hold the marker
that tells the two apart.

`grad_max` in these records is the RESIDUAL |dE/dR| at the converged geometry
R*, while the same name elsewhere in the repository is the driving force at the
input geometry R0. `opt_grad_max` names the residual unambiguously; a record
that reports one under the other's meaning compares a converged minimum against
an unrelaxed starting point.

NEGATIVE CONTROL, run and confirmed. Against a build with the outer refreeze
block disabled in each optimizer, `test_refreeze_measures_the_known_drift`
fails for BOTH engines and `test_relax_auto_reaches_the_refreeze_loop` with
them: the shift comes back None instead of the drift the surface was built to
have. The refreeze=0 gates still pass there, which is what makes them a
separate check rather than a restatement.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import GEOM_OPT_CONV
from src.properties.optimize import optimize, optimize_geometric, relax

try:
    # The optional dependency the preferred engine needs, probed exactly as
    # `relax` probes it.
    import geometric  # noqa: F401
    HAVE_GEOMETRIC = True
except ImportError:
    HAVE_GEOMETRIC = False

#: The toy surface, in Bohr and Hartree.
STIFFNESS = 0.4
#: Where the minimum sits while the conventions are frozen, and where refreezing
#: puts it instead. The drift is far larger than the convergence thresholds, so
#: a loop that does not run cannot be mistaken for one that ran and found zero.
R_E = 2.2
REFREEZE_DRIFT = 0.05
#: Both optimizers hold the centroid still -- translations are projected out of
#: the Cartesian step and are zero-gradient coordinates in TRIC -- so a
#: homonuclear pair splits a change in the bond length evenly between its atoms.
KNOWN_SHIFT = 0.5 * REFREEZE_DRIFT

ENGINES = pytest.mark.parametrize('engine', ['internal', 'geometric'])


class DriftingHarmonicSurface:
    """E = k (r - r_e)^2 / 2 in the bond length, whose refreeze MOVES r_e.

    The frozen convention here is r_e itself, and rebuilding it at a new
    geometry shifts the minimum by a known amount. That is the whole point: on
    a real surface the drift of the frozen conventions is unknown, so the loop
    that measures it can only be gated against a surface whose drift is put
    there by hand. `MeanFieldSurface` and the toy surface in
    `test_properties.py` both refreeze to themselves and measure zero, which
    every implementation of the loop reproduces, including no implementation.
    """

    def __init__(self, mol, k=STIFFNESS, r_e=R_E, drift=REFREEZE_DRIFT):
        self.mol0, self.k, self.r_e, self.drift = mol, k, r_e, drift

    def scf_factory(self, mol):
        """No mean field is involved; the surface is a function of the nuclei."""
        return None

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), mf

    def _displacement(self, mol):
        mol = self.mol0 if mol is None else mol
        d = mol.atom_coords()[1] - mol.atom_coords()[0]
        r = float(np.linalg.norm(d))
        return r, d / r

    def total_energy(self, mol=None, mf=None):
        r, _ = self._displacement(mol)
        return 0.5 * self.k * (r - self.r_e) ** 2

    def total_gradient(self, mol=None, mf=None):
        mol = self.mol0 if mol is None else mol
        r, u = self._displacement(mol)
        grad = np.zeros((mol.natm, 3))
        grad[1] = self.k * (r - self.r_e) * u
        grad[0] = -grad[1]
        return grad, self.total_energy(mol), {}

    def refreeze(self, mol):
        """Rebuilt conventions, which here put the minimum `drift` further out."""
        return type(self)(mol, k=self.k, r_e=self.r_e + self.drift,
                          drift=self.drift)

    def label(self):
        return f'drifting harmonic toy, r_e = {self.r_e} Bohr'


@pytest.fixture
def n2_toy():
    """A homonuclear pair 0.2 Bohr inside a minimum that refreezing moves out."""
    mol = gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                verbose=0)
    return DriftingHarmonicSurface(mol), mol


def _run(engine, surface, mol, **kw):
    """One optimizer by name, skipping when its optional dependency is absent."""
    if engine == 'geometric':
        if not HAVE_GEOMETRIC:
            pytest.skip('geomeTRIC is not installed')
        return optimize_geometric(surface, mol, verbose=False, **kw)
    return optimize(surface, mol, verbose=False, **kw)


def _bond_length(mol):
    return float(np.linalg.norm(mol.atom_coords()[1] - mol.atom_coords()[0]))


@ENGINES
def test_the_residual_gradient_has_its_own_name(engine, n2_toy):
    """`opt_grad_max` is the residual at R*, and the record says which engine.

    Two optimizers converge to different residuals on the same minimum, so a
    record that does not name the one that ran cannot be compared with another.
    """
    surface, mol = n2_toy
    _, info = _run(engine, surface, mol)

    assert info['converged']
    assert info['opt_grad_max'] == info['grad_max']
    assert info['opt_grad_max'] < GEOM_OPT_CONV['grad_max']
    assert info['optimizer'] == engine


@ENGINES
def test_refreeze_measures_the_known_drift(engine, n2_toy):
    """refreeze=1 finds the minimum of the REBUILT surface, and says how far.

    The shift is a geometry displacement in Bohr, so it is gated at the
    optimizer's own step threshold: converging to within `step_max` of a
    minimum is all either engine promises.
    """
    surface, mol = n2_toy
    mol_opt, info = _run(engine, surface, mol, refreeze=1)

    assert info['refreeze_shift'] == pytest.approx(
        KNOWN_SHIFT, abs=GEOM_OPT_CONV['step_max'])
    assert info.get('refreeze', 'measured') == 'measured'
    # The returned geometry is the refrozen surface's minimum, not the first
    # one's: the shift is the size of a move that actually happened.
    assert _bond_length(mol_opt) == pytest.approx(
        R_E + REFREEZE_DRIFT, abs=2 * GEOM_OPT_CONV['step_max'])
    assert info['refreeze_denergy'] == pytest.approx(
        info['refreeze_info']['energy'] - info['history'][-1]['e'], abs=1e-12)
    assert info['energy'] == info['refreeze_info']['energy']


@ENGINES
def test_refreeze_zero_says_it_did_not_measure(engine, n2_toy):
    """A null shift is only ever an explicit refreeze=0, and is marked as one.

    Without the marker a record cannot distinguish a drift that was measured
    and came out zero from one that was never looked at.
    """
    surface, mol = n2_toy
    mol_opt, info = _run(engine, surface, mol, refreeze=0)

    assert info['refreeze_shift'] is None
    assert info['refreeze'] == 'not measured'
    # The first surface's own minimum, untouched by any rebuilt convention.
    assert _bond_length(mol_opt) == pytest.approx(
        R_E, abs=GEOM_OPT_CONV['step_max'])


def test_relax_auto_reaches_the_refreeze_loop(n2_toy):
    """`relax` forwards refreeze to whichever engine it picked.

    This is the path every campaign record is written through, and the one that
    reported an unmeasured drift as a null shift for as long as geomeTRIC was
    installed.
    """
    surface, mol = n2_toy
    _, info = relax(surface, mol, engine='auto', refreeze=1, verbose=False)

    assert isinstance(info['refreeze_shift'], float)
    assert info['refreeze_shift'] == pytest.approx(
        KNOWN_SHIFT, abs=GEOM_OPT_CONV['step_max'])
    assert info['optimizer'] == ('geometric' if HAVE_GEOMETRIC else 'internal')
