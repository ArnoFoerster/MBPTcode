"""The refreeze loop of BOTH optimizers, and which gradient `grad_max` names.

WHAT BREAKS IF THIS FAILS. A geometry optimization freezes the discrete
conventions -- the quasiparticle set, the frame, the interpolation layout --
at the starting geometry and keeps them, so the minimum it reports belongs to
the surface chosen at the START. `refreeze` rebuilds them at the converged
geometry R* and relaxes once more on the rebuilt surface. The measurement is
the DRIFT E_new(R*) - E_old(R*), taken at that one geometry: the rebuilt
surface's first evaluation against the old surface's last. Comparing
E_new(R*_new) - E_old(R*_old) instead would put convention drift and walk
descent in one number across two surfaces and two geometries, stop as
converged whenever the walk did not move, and loosen its tolerance by the
walk's own resolution; such passes oscillate (formaldehyde T1: -3.5, -1.8,
+1.4, -0.8 meV). A drift within REFREEZE_TOL_MEV ends the loop; one above it
refreezes at the new minimum and walks again, and a drift still above it after
REFREEZE_MAX_PASSES is warned about and flagged (RefreezeNotConverged,
walk_flags 'refreeze_not_converged'; refuse inputs,
warn numerics), the last pass's geometry kept; nothing stands in between.

`grad_max` in these records is the RESIDUAL |dE/dR| at the converged geometry
R*, while the same name elsewhere in the repository is the driving force at the
input geometry R0. `opt_grad_max` names the residual unambiguously.

The drift, refusal and one-pass tests see a refreeze that records no
`refreeze_drift`, never flags a drift above the tolerance, or accepts refreeze=2.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import importlib
import os
import sys
import warnings
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

from src.Base.constants import (GEOM_OPT_CONV, HARTREE_TO_MEV,
                                REFREEZE_MAX_PASSES, REFREEZE_TOL_MEV)
from src.properties.optimize import (HAVE_GEOMETRIC, RefreezeNotConverged, optimize,
                                     optimize_geometric, relax,
                                     translation_rotation_basis,
                                     walk_resolution)
from src.properties.vibronic import relax_state
from tests.pins import assert_same, assert_same_tree

#: the module itself: `src.properties` exports a function of the same name
optimize_module = importlib.import_module('src.properties.optimize')

#: The toy surface, in Bohr and Hartree.
STIFFNESS = 0.4
#: Where the minimum sits while the conventions are frozen, and how far
#: refreezing moves it: k DRIFT^2 / 2 = 0.54 meV of drift at the old minimum,
#: within REFREEZE_TOL_MEV.
R_E = 2.2
REFREEZE_DRIFT = 0.01
#: Both optimizers hold the centroid still, so a homonuclear pair splits a
#: change in the bond length evenly between its atoms.
KNOWN_SHIFT = 0.5 * REFREEZE_DRIFT
#: Tight thresholds, so a walk ends within 1e-5 Bohr of its minimum.
TIGHT = {'opt_grad_max': 1e-7, 'grad_rms': 1e-7, 'step_max': 1e-6,
         'step_rms': 1e-6}
#: Bohr: how close a tightly converged walk ends to its minimum.
WALK_TOL = 1e-4
#: meV: lifts of the energy a refreeze makes without moving the minimum, one
#: just inside the REFREEZE_TOL_MEV tolerance and one just outside it.
LIFT_CONVERGED = 0.9 * REFREEZE_TOL_MEV
LIFT_REFUSED = 1.1 * REFREEZE_TOL_MEV
#: The settling toy: where its first surface is built, the bond its
#: conventions converge on, how much of the remaining distance each rebuild
#: keeps, and what its conventions are worth per Bohr.
B0 = 2.0
R_TRUE = 2.01
SETTLE = 0.5
LIFT = 1e-4

ENGINES = pytest.mark.parametrize('engine', ['internal', 'geometric'])


class DriftingHarmonicSurface:
    """E = k (r - r_e)^2 / 2 in the bond length, whose refreeze MOVES r_e.

    The frozen convention here is r_e itself, and rebuilding it at a new
    geometry shifts the minimum by a known amount, so the drift at the old
    minimum is known too: k drift^2 / 2.
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


class OffsetHarmonicSurface(DriftingHarmonicSurface):
    """A refreeze that lifts the ENERGY by `lift` meV and leaves the minimum
    where it was."""

    def __init__(self, mol, lift=LIFT_CONVERGED, level=0.0):
        super().__init__(mol, drift=0.0)
        self.lift, self.level = lift, level

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), SimpleNamespace(e_tot=0.0)

    def total_energy(self, mol=None, mf=None):
        return super().total_energy(mol) + self.level

    def total_gradient(self, mol=None, mf=None):
        grad, _, info = super().total_gradient(mol)
        return grad, self.total_energy(mol), info

    def refreeze(self, mol):
        return type(self)(mol, lift=self.lift,
                          level=self.level + self.lift / HARTREE_TO_MEV)


class SettlingLiftSurface(OffsetHarmonicSurface):
    """A refreeze that lifts the energy by `lift` meV, each rebuild lifting
    half as much as the one before: 1.5 x REFREEZE_TOL_MEV, then 0.75 x."""

    def refreeze(self, mol):
        return type(self)(mol, lift=0.5 * self.lift,
                          level=self.level + self.lift / HARTREE_TO_MEV)


class SettlingHarmonicSurface(OffsetHarmonicSurface):
    """Conventions that depend on where they are built: built at bond length
    b, the minimum sits at R_TRUE + SETTLE (b - R_TRUE) and the energy LIFT
    (b - B0) higher, so a refreeze both moves the minimum and lifts it."""

    def __init__(self, mol, built=B0):
        super().__init__(mol, level=LIFT * (built - B0))
        self.r_e = R_TRUE + SETTLE * (built - R_TRUE)

    def refreeze(self, mol):
        return type(self)(mol, built=self._displacement(mol)[0])


class GrowingHarmonicSurface(SettlingHarmonicSurface):
    """The settling toy with a grow-only explicit set, as the adaptive chain
    keeps it (`qp_growth`, which `ExcitedStateChain` exposes)."""

    def __init__(self, mol, built=B0, explicit=(), growth=None):
        super().__init__(mol, built=built)
        self.explicit, self.qp_growth = tuple(explicit), growth

    def refreeze(self, mol):
        added = len(self.explicit)
        return type(self)(mol, built=self._displacement(mol)[0],
                          explicit=self.explicit + (added,),
                          growth={'carried': list(self.explicit),
                                  'added': [added], 'not_explicit': []})


@pytest.fixture
def n2():
    return gto.M(atom='N 0 0 0; N 0 0 2.0', unit='Bohr', basis='sto-3g',
                 verbose=0)


def _run(engine, surface, mol, **kw):
    """One optimizer by name, tightly converged, skipping when its optional
    dependency is absent."""
    if engine == 'geometric':
        if not HAVE_GEOMETRIC:
            pytest.skip('geomeTRIC is not installed')
        return optimize_geometric(surface, mol, verbose=False, nudge=0,
                                  converge='GAU_VERYTIGHT', **kw)
    return optimize(surface, mol, verbose=False, nudge=0, conv=TIGHT, **kw)


def _bond_length(mol):
    return float(np.linalg.norm(mol.atom_coords()[1] - mol.atom_coords()[0]))


@ENGINES
def test_the_residual_gradient_has_its_own_name(engine, n2):
    """`opt_grad_max` is the residual at R*, and the record says which engine."""
    _, info = _run(engine, DriftingHarmonicSurface(n2), n2)
    assert info['converged']
    assert info['opt_grad_max'] == info['grad_max']
    assert info['opt_grad_max'] < GEOM_OPT_CONV['grad_max']
    assert info['optimizer'] == engine


@ENGINES
def test_refreeze_measures_the_known_drift(engine, n2):
    """refreeze=1 finds the minimum of the REBUILT surface, says how far it
    moved, and measures the drift at the old minimum: k drift^2 / 2."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '.*REFREEZE_TOL_MEV', RuntimeWarning)
        mol_opt, info = _run(engine, DriftingHarmonicSurface(n2), n2,
                             refreeze=1)
    assert info['refreeze'] == 'measured'
    assert info['refreeze_shift'] == pytest.approx(KNOWN_SHIFT, abs=WALK_TOL)
    assert _bond_length(mol_opt) == pytest.approx(R_E + REFREEZE_DRIFT,
                                                  abs=2 * WALK_TOL)
    known = 0.5 * STIFFNESS * REFREEZE_DRIFT ** 2
    assert info['refreeze_drift'] == pytest.approx(known, rel=1e-2)
    assert info['refreeze_drift_meV'] == pytest.approx(
        info['refreeze_drift'] * HARTREE_TO_MEV)
    assert 0 < info['refreeze_drift_meV'] <= REFREEZE_TOL_MEV
    assert info['refreeze_converged'] is True
    assert info['refreeze_denergy'] == pytest.approx(
        info['refreeze_info']['energy'] - info['history'][-1]['e'], abs=1e-12)
    assert info['energy'] == info['refreeze_info']['energy']


@ENGINES
def test_the_drift_is_measured_at_one_geometry(engine, n2):
    """The drift is the rebuilt surface minus the old one at the OLD minimum,
    whatever the walk on the rebuilt surface then descends."""
    surface = SettlingHarmonicSurface(n2)
    first, alone = _run(engine, surface, n2)
    rebuilt = surface.refreeze(first)
    expected = rebuilt.total_energy(first) - surface.total_energy(first)
    _, info = _run(engine, SettlingHarmonicSurface(n2), n2, refreeze=1)
    assert info['refreeze_drift'] == pytest.approx(expected, abs=1e-9)
    # the walk's descent is in the mixed number, not in the drift
    assert info['refreeze_denergy'] < info['refreeze_drift'] - 1e-7


@ENGINES
def test_the_one_tolerance_flags(engine, n2, monkeypatch):
    """REFREEZE_TOL_MEV alone decides: the same drift that converges under
    it is flagged once the tolerance is set just below it."""
    monkeypatch.setattr(optimize_module, 'REFREEZE_TOL_MEV',
                        0.95 * LIFT_CONVERGED)
    with pytest.warns(RefreezeNotConverged, match='REFREEZE_TOL_MEV'):
        _, info = _run(engine, OffsetHarmonicSurface(n2), n2, refreeze=1)
    assert info['refreeze'] == 'not converged'
    assert info['refreeze_converged'] is False
    assert info['refreeze_drift_meV'] == pytest.approx(LIFT_CONVERGED,
                                                       abs=1e-9)


@ENGINES
def test_a_drift_within_the_tolerance_converges(engine, n2):
    """A drift just inside REFREEZE_TOL_MEV is converged, with no warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '.*REFREEZE_TOL_MEV', RuntimeWarning)
        _, info = _run(engine, OffsetHarmonicSurface(n2), n2, refreeze=1)
    assert info['refreeze_drift_meV'] == pytest.approx(LIFT_CONVERGED,
                                                       abs=1e-9)
    assert info['refreeze_converged'] is True
    assert info['refreeze_shift'] < WALK_TOL


@ENGINES
def test_a_drift_above_the_tolerance_is_flagged(engine, n2):
    """Above REFREEZE_TOL_MEV the relaxation warns and returns its geometry
    and record, flagged 'refreeze_not_converged'."""
    with pytest.warns(RefreezeNotConverged, match='REFREEZE_TOL_MEV'):
        mol, info = _run(engine, OffsetHarmonicSurface(n2, lift=LIFT_REFUSED),
                         n2, refreeze=1)
    assert info['refreeze'] == 'not converged'
    assert [f['flag'] for f in info['walk_flags']] == ['refreeze_not_converged']
    assert info['refreeze_drift_meV'] == pytest.approx(LIFT_REFUSED, abs=1e-9)
    assert mol.natm == n2.natm


@ENGINES
def test_a_drift_above_the_tolerance_refreezes_again(engine, n2):
    """The outer loop: a first drift of 1.5 x REFREEZE_TOL_MEV refreezes at
    the new minimum and walks again; the second, 0.75 x, ends the loop."""
    lift = 1.5 * REFREEZE_TOL_MEV
    _, info = _run(engine, SettlingLiftSurface(n2, lift=lift), n2, refreeze=1)
    passes = info['refreeze_passes']
    assert [p['pass'] for p in passes] == [1, 2]
    assert passes[0]['drift_meV'] == pytest.approx(lift, abs=1e-9)
    assert passes[1]['drift_meV'] == pytest.approx(0.5 * lift, abs=1e-9)
    assert info['refreeze_drift_meV'] == passes[1]['drift_meV']
    assert info['refreeze_converged'] is True
    assert info['refreeze'] == 'measured'
    assert info['final_surface'].level == pytest.approx(
        1.5 * lift / HARTREE_TO_MEV, abs=1e-12)


@ENGINES
def test_the_loop_stops_after_its_passes(engine, n2):
    """A drift that never falls below REFREEZE_TOL_MEV is flagged after
    REFREEZE_MAX_PASSES passes, each one recorded."""
    with pytest.warns(RefreezeNotConverged, match='REFREEZE_MAX_PASSES'):
        _, info = _run(engine, OffsetHarmonicSurface(n2, lift=LIFT_REFUSED), n2,
                       refreeze=1)
    passes = info['refreeze_passes']
    assert [p['pass'] for p in passes] == list(
        range(1, REFREEZE_MAX_PASSES + 1))
    assert all(p['drift_meV'] == pytest.approx(LIFT_REFUSED, abs=1e-9)
               for p in passes)


def test_a_walk_resumes_in_a_later_pass(n2, monkeypatch):
    """A walk stopped in the loop's second pass resumes there, to the same
    minimum and the same drifts, to a few ulp. The settling toy's drifts fall
    0.048, 0.015, 0.006 meV; a tolerance of 0.01 meV takes three passes."""
    monkeypatch.setattr(optimize_module, 'REFREEZE_TOL_MEV', 0.01)
    states = []
    want_mol, want = _run('internal', SettlingHarmonicSurface(n2), n2,
                          refreeze=1, checkpoint=states.append)
    assert [p['pass'] for p in want['refreeze_passes']] == [1, 2, 3]
    held = next(st for st in states if st['pass'] == 2)
    got_mol, got = _run('internal', SettlingHarmonicSurface(n2), n2,
                        refreeze=1, resume=held)
    assert_same(got_mol.atom_coords(), want_mol.atom_coords(), unit='1',
                quantity='root', label='minimum')
    assert_same_tree(got['refreeze_passes'], want['refreeze_passes'],
                     label='refreeze passes')
    assert_same(got['energy'], want['energy'], label='energy')


@ENGINES
def test_refreeze_is_asked_once(engine, n2):
    """A count of passes is refused: the drift decides how many."""
    with pytest.raises(ValueError, match='refreeze'):
        _run(engine, DriftingHarmonicSurface(n2), n2, refreeze=2)


@ENGINES
def test_refreeze_zero_says_it_did_not_measure(engine, n2):
    """A null drift is only ever an explicit refreeze=0, and is marked."""
    mol_opt, info = _run(engine, DriftingHarmonicSurface(n2), n2, refreeze=0)
    assert info['refreeze_shift'] is None
    assert info['refreeze'] == 'not measured'
    assert _bond_length(mol_opt) == pytest.approx(R_E, abs=WALK_TOL)


def test_relax_auto_reaches_the_refreeze(n2):
    """`relax` forwards refreeze to whichever engine it picked."""
    _, info = relax(OffsetHarmonicSurface(n2), n2, engine='auto', refreeze=1,
                    verbose=False, nudge=0)
    assert info['refreeze'] == 'measured'
    assert info['refreeze_drift_meV'] == pytest.approx(LIFT_CONVERGED,
                                                       abs=1e-9)
    assert info['optimizer'] == ('geometric' if HAVE_GEOMETRIC else 'internal')


def test_a_relaxed_state_reads_its_energy_on_the_last_surface(n2):
    """`relax_state` reports the energy of the surface the walk ended on."""
    record = relax_state(OffsetHarmonicSurface(n2), n2, engine='cartesian',
                         refreeze=1, verbose=False, nudge=0, conv=TIGHT)
    lifted = LIFT_CONVERGED / HARTREE_TO_MEV
    assert record['e_total'] == pytest.approx(lifted, abs=1e-9)
    assert record['surface'].level == lifted
    assert record['info']['energy'] == pytest.approx(record['e_total'],
                                                     abs=1e-12)


@ENGINES
def test_the_record_names_the_states_the_refreeze_added(engine, n2):
    """`refreeze_qp_growth` is what the rebuild carried and added; None on a
    surface without an adaptive set."""
    _, info = _run(engine, GrowingHarmonicSurface(n2), n2, refreeze=1)
    assert info['refreeze_qp_growth'] == {'carried': [], 'added': [0],
                                          'not_explicit': []}
    _, info = _run(engine, SettlingHarmonicSurface(n2), n2, refreeze=1)
    assert info['refreeze_qp_growth'] is None


def test_walk_resolution_is_half_g_hinv_g_in_the_internal_space():
    """On a positive-definite Hessian dE_walk is 1/2 g^T H^-1 g over the
    internal modes; a non-positive mode makes it the safe bound
    1/2 |g|^2 / lambda_min, said so; no positive mode, no estimate."""
    rng = np.random.default_rng(7)
    mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.45',
                unit='Bohr', basis='sto-3g', verbose=0)
    coords = mol.atom_coords()
    tr = translation_rotation_basis(coords)
    w, v = np.linalg.eigh(np.eye(9) - tr.T @ tr)
    q = v[:, w > 0.5]
    lam = np.array([3e-4, 0.2, 0.7])
    basis = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    hess = q @ basis @ np.diag(lam) @ basis.T @ q.T + tr.T @ tr * 0.05
    g = q @ rng.normal(size=3) * 1e-4
    res = walk_resolution(g, hess, coords)
    gi = q.T @ g
    analytic = 0.5 * gi @ np.linalg.solve(q.T @ hess @ q, gi)
    assert res['bound'] == 'quadratic' and res['nonpositive_modes'] == 0
    assert res['de_meV'] == pytest.approx(analytic * HARTREE_TO_MEV,
                                          rel=1e-10)
    bad = q @ basis @ np.diag([-1e-3, 0.2, 0.7]) @ basis.T @ q.T
    res = walk_resolution(g, bad, coords)
    assert res['bound'] == 'safe bound' and res['nonpositive_modes'] == 1
    assert res['de_meV'] == pytest.approx(
        0.5 * gi @ gi / 0.2 * HARTREE_TO_MEV, rel=1e-6)
    assert walk_resolution(g, -np.eye(9), coords)['de_meV'] is None
    assert walk_resolution(g, None, coords) is None
