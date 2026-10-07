"""The refreeze loop on both optimizers, and which gradient `grad_max` names.

A geometry optimization freezes the discrete conventions (the quasiparticle
set, the frame, the interpolation layout) at the starting geometry and keeps
them, so the minimum it reports belongs to the surface chosen at the start.
`refreeze` rebuilds them at the converged geometry and optimizes again, until
the energy has converged in them, and how far the geometry then moves is the
measurement of that approximation. `relax(engine='auto')` routes to
geomeTRIC whenever it is installed, so both engines must run the loop, and a
record that did not measure the drift must say so rather than report a
`refreeze_shift` of None that reads like a measured zero. These gates hold
the loop on both engines, its stopping rule (the walk-resolved tolerance),
and the marker that tells the two cases apart.

`grad_max` in these records is the residual |dE/dR| at the converged geometry
R*, while the same name elsewhere is the driving force at the input geometry
R0. `opt_grad_max` names the residual unambiguously.

Negative control: with the outer refreeze block disabled in each optimizer,
`test_refreeze_measures_the_known_drift` fails for both engines and
`test_relax_auto_reaches_the_refreeze_loop` with them; the refreeze=0 gates
still pass, which makes them a separate check rather than a restatement.

Every check asserts: pytest discards a returned verdict and passes on False.
"""
import importlib
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import (GEOM_OPT_CONV, HARTREE_TO_MEV,
                                REFREEZE_TOL_FLAG_MEV, REFREEZE_TOL_MEV)
from src.properties.optimize import (optimize, optimize_geometric,
                                     refreeze_passes, relax,
                                     translation_rotation_basis,
                                     walk_resolution)
from src.properties.vibronic import relax_state

#: the module itself: `src.properties` exports a function of the same name
optimize_module = importlib.import_module('src.properties.optimize')

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
#: Hartree: how far a refreeze lifts the toy surface without moving it.
OFFSET = 1e-3
#: The settling toy: where its first surface is built (the n2_toy bond),
#: the bond its conventions converge on, how much of the remaining distance
#: each rebuild keeps, and what its conventions are worth per Bohr.
B0 = 2.0
R_TRUE = 2.3
SETTLE = 0.5
LIFT = 0.02

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

    This is the path every relaxation record is written through; with
    geomeTRIC installed it must still measure the drift.
    """
    surface, mol = n2_toy
    _, info = relax(surface, mol, engine='auto', refreeze=1, verbose=False)

    assert isinstance(info['refreeze_shift'], float)
    assert info['refreeze_shift'] == pytest.approx(
        KNOWN_SHIFT, abs=GEOM_OPT_CONV['step_max'])
    assert info['optimizer'] == ('geometric' if HAVE_GEOMETRIC else 'internal')


class OffsetHarmonicSurface(DriftingHarmonicSurface):
    """A refreeze that lifts the energy and leaves the minimum where it was:
    the refrozen walk does not move while its energy rises. Rebuilt where it
    was built it is the same surface."""

    def __init__(self, mol, k=STIFFNESS, r_e=R_E, level=0.0):
        super().__init__(mol, k=k, r_e=r_e, drift=0.0)
        self.level = level

    def mean_field(self, mol=None, mf=None):
        return (self.mol0 if mol is None else mol), SimpleNamespace(e_tot=0.0)

    def total_energy(self, mol=None, mf=None):
        return super().total_energy(mol) + self.level

    def total_gradient(self, mol=None, mf=None):
        grad, _, info = super().total_gradient(mol)
        return grad, self.total_energy(mol), info

    def refreeze(self, mol):
        return type(self)(mol, k=self.k, r_e=self.r_e,
                          level=OFFSET if self.level == 0.0 else self.level)


class SettlingHarmonicSurface(OffsetHarmonicSurface):
    """Conventions that depend on where they are built, as a real surface's
    do: built at bond length b, the minimum sits at R_TRUE + SETTLE (b -
    R_TRUE) and the energy LIFT (b - B0) higher. Refreezing at each minimum
    converges on R_TRUE geometrically, and rebuilt where it was built the
    surface is the same one."""

    def __init__(self, mol, built=B0):
        super().__init__(mol, r_e=R_TRUE + SETTLE * (built - R_TRUE),
                         level=LIFT * (built - B0))

    def refreeze(self, mol):
        return type(self)(mol, built=self._displacement(mol)[0])


@ENGINES
def test_the_refreeze_loop_runs_until_the_energy_has_converged(engine,
                                                               n2_toy):
    """Passes repeat while the energy jumps by more than REFREEZE_TOL_MEV or
    the walk still moves, at most `refreeze` of them; the record lists every
    pass and the energy is the last surface's."""
    _, mol = n2_toy
    tol = REFREEZE_TOL_MEV / HARTREE_TO_MEV
    _, info = _run(engine, SettlingHarmonicSurface(mol), mol, refreeze=40)
    jumps, shifts = info['refreeze_jumps'], info['refreeze_shifts']
    assert info['refreeze_converged'] and 2 < len(jumps) < 40
    assert abs(jumps[-1]) < tol or shifts[-1] < GEOM_OPT_CONV['step_max']
    assert all(abs(j) >= tol for j in jumps[:-1])
    assert info['refreeze_denergy'] == jumps[0]
    assert info['refreeze_shift'] == shifts[0]
    _, info1 = _run(engine, SettlingHarmonicSurface(mol), mol, refreeze=1)
    assert len(info1['refreeze_jumps']) == 1
    assert not info1['refreeze_converged']
    assert info1['refreeze_jumps'][0] == pytest.approx(jumps[0], abs=1e-9)
    _, info = _run(engine, OffsetHarmonicSurface(mol), mol, refreeze=3)
    assert len(info['refreeze_shifts']) == 1 and info['refreeze_converged']
    assert info['energy'] == pytest.approx(OFFSET, abs=1e-7)


def test_a_relaxed_state_reads_its_energy_on_the_last_surface(n2_toy):
    """`relax_state` reports the energy of the surface the walk ended on.

    Read on the surface frozen at the start, a minimum whose conventions moved
    reports the start's: the offset of this toy. Negative control: reading `surface` instead of
    `final_surface` in `relax_state` returns the first level, 0."""
    _, mol = n2_toy
    record = relax_state(OffsetHarmonicSurface(mol), mol, engine='cartesian',
                         refreeze=1, verbose=False)
    assert record['e_total'] == pytest.approx(OFFSET, abs=1e-7)
    assert record['surface'].level == OFFSET
    assert record['info']['energy'] == pytest.approx(record['e_total'],
                                                     abs=1e-12)


class GrowingHarmonicSurface(SettlingHarmonicSurface):
    """The settling toy with a grow-only explicit set, as the adaptive chain
    keeps it: each refreeze carries the set it was built from and adds the
    pass's number (`qp_growth`, which `ExcitedStateChain` exposes)."""

    def __init__(self, mol, built=B0, explicit=(), growth=None):
        super().__init__(mol, built=built)
        self.explicit, self.qp_growth = tuple(explicit), growth

    def refreeze(self, mol):
        added = len(self.explicit)
        return type(self)(mol, built=self._displacement(mol)[0],
                          explicit=self.explicit + (added,),
                          growth={'carried': list(self.explicit),
                                  'added': [added], 'not_explicit': []})


@ENGINES
def test_the_record_names_the_states_each_refreeze_added(engine, n2_toy):
    """`refreeze_qp_growth` lists, per pass, the explicit states that pass
    carried and the ones it added; None on a surface without an adaptive
    set."""
    _, mol = n2_toy
    _, info = _run(engine, GrowingHarmonicSurface(mol), mol, refreeze=40)
    growth = info['refreeze_qp_growth']
    assert len(growth) == len(info['refreeze_jumps']) > 2
    for k, g in enumerate(growth):
        assert g['added'] == [k] and g['carried'] == list(range(k))
    _, info = _run(engine, SettlingHarmonicSurface(mol), mol, refreeze=2)
    assert info['refreeze_qp_growth'] == [None] * len(info['refreeze_jumps'])


# ---------------------------------------------------------------------------
# The walk-resolved tolerance: two passes count as
# converged when |jump| <= max(REFREEZE_TOL_MEV, dE_walk), dE_walk =
# 1/2 g^T H^-1 g on the walk's own last gradient and Hessian.

#: The soft toy: E = SOFT_K (r - r_e)^2 / 2 + level, in Ha/Bohr^2. Each
#: refreeze moves r_e between SOFT_R_E and the other end and the level by
#: SOFT_LEVEL_MEV, so every jump stays above REFREEZE_TOL_MEV; the walks stop
#: at SOFT_CONV's residual, which leaves dE_walk above the jump.
SOFT_K = 1e-3
SOFT_R_E = (2.0, 2.7)
SOFT_LEVEL_MEV = 0.3
SOFT_CONV = {'opt_grad_max': 2e-4, 'grad_rms': 1.0, 'step_max': 10.0,
             'step_rms': 10.0}


class SoftSurface(OffsetHarmonicSurface):
    """A soft bond whose refreeze moves both its minimum and its level."""

    def __init__(self, mol, k=SOFT_K, r_e=SOFT_R_E[0], level=0.0):
        super().__init__(mol, k=k, r_e=r_e, level=level)

    def refreeze(self, mol):
        other = SOFT_R_E[1] if self.r_e == SOFT_R_E[0] else SOFT_R_E[0]
        level = 0.0 if self.level else SOFT_LEVEL_MEV / HARTREE_TO_MEV
        return type(self)(mol, k=self.k, r_e=other, level=level)


def exact_hessian(mol, k, r_e):
    """The Cartesian Hessian of k (r - r_e)^2 / 2 for a pair, Ha/Bohr^2."""
    d = mol.atom_coords()[1] - mol.atom_coords()[0]
    r = float(np.linalg.norm(d))
    u = d / r
    block = k * np.outer(u, u) + k * (r - r_e) / r * (np.eye(3)
                                                       - np.outer(u, u))
    return np.block([[block, -block], [-block, block]])


def soft_pair():
    return gto.M(atom='N 0 0 0; N 0 0 2.3', unit='Bohr', basis='sto-3g',
                 verbose=0)


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


def test_a_soft_mode_converges_on_what_the_walk_resolves():
    """The soft toy's jumps never fall below REFREEZE_TOL_MEV, so the old
    rule runs out of passes; dE_walk, the energy the walk leaves above its own
    minimum, exceeds them, so the walk-resolved rule converges and says so.
    On the exact Hessian dE_walk is the toy's own k (r - r_e)^2 / 2 at the
    walk's end. Negative control: the old rule (no dE_walk) does not
    converge."""
    tol = REFREEZE_TOL_MEV
    mol = soft_pair()
    hess = exact_hessian(mol, SOFT_K, SOFT_R_E[0])
    end, alone = optimize(SoftSurface(mol), mol, verbose=False,
                          conv=SOFT_CONV, hess_init=hess)
    above = 0.5 * SOFT_K * (_bond_length(end) - SOFT_R_E[0]) ** 2
    assert alone['walk_resolution']['bound'] == 'quadratic'
    assert alone['walk_resolution']['de_meV'] == pytest.approx(
        above * HARTREE_TO_MEV, rel=1e-8)
    _, info = optimize(SoftSurface(mol), mol, verbose=False, conv=SOFT_CONV,
                       hess_init=hess, refreeze=4)
    assert info['first_walk_de_meV'] == alone['walk_resolution']['de_meV']
    jumps = [abs(j) * HARTREE_TO_MEV for j in info['refreeze_jumps']]
    assert all(j >= tol for j in jumps)
    assert info['refreeze_converged'] and info['refreeze_walk_decided'][-1]
    assert info['refreeze_tol_meV'][-1] == max(
        info['first_walk_de_meV'] if len(jumps) == 1
        else info['refreeze_walk_de_meV'][-2],
        info['refreeze_walk_de_meV'][-1])
    assert jumps[-1] <= info['refreeze_tol_meV'][-1]
    assert info['refreeze_walk_bound'] == ['quadratic'] * len(jumps)
    assert not info['refreeze_tol_flag']
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(optimize_module, 'walk_resolution', lambda *a: None)
        _, old = optimize(SoftSurface(mol), mol, verbose=False,
                          conv=SOFT_CONV, hess_init=hess, refreeze=4)
    assert not old['refreeze_converged']
    assert len(old['refreeze_jumps']) == 4


@ENGINES
def test_a_stiff_surface_is_decided_on_the_floor(engine, n2_toy):
    """Stiff, the walks resolve far below REFREEZE_TOL_MEV: the tolerance is
    the floor at every pass and the walk-resolved branch never decides."""
    _, mol = n2_toy
    _, info = _run(engine, SettlingHarmonicSurface(mol), mol, refreeze=40)
    assert info['refreeze_converged']
    assert all(d < REFREEZE_TOL_MEV for d in info['refreeze_walk_de_meV'])
    assert info['refreeze_tol_meV'] == ([REFREEZE_TOL_MEV]
                                        * len(info['refreeze_jumps']))
    assert not any(info['refreeze_walk_decided'])
    assert not info['refreeze_tol_flag']


class _Stub:
    """A surface whose refreeze is itself: the walks below are scripted."""

    def refreeze(self, mol):
        return self


def scripted_walk(energies, de_meV):
    """A `walk` that moves the pair 0.1 Bohr and returns the next energy
    and dE_walk of the script."""
    script = iter(zip(energies, de_meV))

    def walk(surface, mol, hessian):
        e, de = next(script)
        moved = mol.copy()
        moved.set_geom_(mol.atom_coords() + np.array([[0, 0, 0],
                                                      [0, 0, 0.1]]),
                        unit='Bohr')
        return moved, {'converged': True, 'energy': e, 'omega': None,
                       'hessian': None, 'status': 'ok',
                       'walk_resolution': {'de_meV': de,
                                           'bound': 'quadratic'}}
    return walk


def test_a_tolerance_above_one_mev_is_flagged():
    """Converged only to a dE_walk above REFREEZE_TOL_FLAG_MEV: the record
    says so and a warning is raised; below it, neither."""
    mol = soft_pair()
    jump = 1.5 / HARTREE_TO_MEV
    first = {'converged': True, 'energy': 0.0, 'omega': None,
             'hessian': None, 'walk_resolution': {'de_meV': 2.0,
                                                  'bound': 'quadratic'}}
    with pytest.warns(RuntimeWarning, match='REFREEZE_TOL_FLAG_MEV'):
        _, info = refreeze_passes(scripted_walk([jump], [1.8]), _Stub(), mol,
                                  dict(first), 4, False)
    assert info['refreeze_converged'] and info['refreeze_tol_flag']
    assert info['refreeze_tol_meV'] == [2.0]
    assert info['refreeze_tol_meV'][0] > REFREEZE_TOL_FLAG_MEV
    small = dict(first, walk_resolution={'de_meV': 0.5,
                                         'bound': 'quadratic'})
    _, info = refreeze_passes(scripted_walk([0.3 / HARTREE_TO_MEV], [0.4]),
                              _Stub(), mol, small, 4, False)
    assert info['refreeze_converged'] and not info['refreeze_tol_flag']
    assert info['refreeze_walk_decided'] == [True]
