"""Surfaces evaluate identically on every rank under the distribution context.

A surface carries no communicator: inside `with distributed(comm):` every rank
builds it and evaluates it whole, the kernels it reaches divide their sweeps
over the ranks, and `surface.evaluate` (the boundary a property routine hands
a geometry across) locks the geometry to rank 0's on the way in and the
energy, force and diagnostics on the way out. Six gates:

  (a) Serially no surface carries a `comm`, and its energy and gradient are
      bitwise the pins (tests/one_fit_pins.json), the recorded baseline
      (`BASELINE_PATH`) within `RECORD_MOVE` of them. A single-machine gate:
      the pins were recorded by a pyscf without OpenMP, which repeats its
      bits; a threaded pyscf adds its K-split partials in thread-arrival
      order and cannot meet them.
  (b) Under `run_simulated` (2 and 3 ranks) the composed and single-chain
      surfaces reproduce the same surface evaluated serially, and every rank
      returns the same bits. Each rank converges its own SCF, so the bar is
      anchored: `COMPOSED_GRAD_K` times what the serial energy and force move
      when evaluated again on one BLAS thread (`one_thread`), at least a
      floor: 1e-12 Ha for the energy, the ISDF gradient reproducibility floor
      for one chain's force, and `COMPOSED_GRAD_FLOOR`
      (tests/test_mpi_routes.py) for RPABSESurface, which combines two
      chains' distributed sweeps.
  (c) `potential_energy_surface` inside the context builds the same surface
      a direct constructor does, bitwise, and refuses a `comm=` keyword by
      name: a communicator handed to a surface would be a second route to
      the ranks, which could disagree with the context.
  (d) The geometry lockstep: rank 1's geometry is moved by one ulp before an
      evaluation, as an optimizer whose arithmetic drifted on one rank would
      hand it. Every rank still returns the unperturbed energy and force
      bitwise, rank 1's Mole ends holding rank 0's coordinates, and an
      audited run counts one repaired lockstep on rank 1 (the geometry, by
      the ulp) and nothing downstream. Run on a factory that converges its
      own SCF and on a build-only one, whose SCF the ranks converge together
      (`converged_factory`).
  (e) The chain's own share of the gradient: the orbital response,
      collocation and fit branch a chain forms from the kernels' adjoints are
      serial code on every rank, so `nuclear_gradient` ends in one lockstep.
      Rank 1's orbital branch is moved by 1e-12 Ha/Bohr and every rank
      returns rank 0's gradient bitwise, within the anchored bar of the
      serial one (`one_thread_scatter`, at least `COMPOSED_GRAD_FLOOR`, since
      the partition reassociates the sums the orbital response amplifies).
  (f) The mean field a chain or surface accepts: one each rank converged
      alone differs in its last bits and its orbital gauge, so every entry
      that takes one locks it to rank 0's (a chain's `mf=`, the one
      `mean_field` is handed at another geometry, the entry point's reference
      and the mean-field surface's own). Rank 1's copy has its lowest orbital
      energy one ulp up and its last orbital's sign flipped, and every rank
      ends holding rank 0's arrays.

One Mole per rank wherever two rank threads differentiate: the nuclear
gradient's Hcore derivative writes into `mol._env` in place
(`mol.with_rinv_at_nucleus`), so two rank threads sharing a Mole corrupt each
other's rinv origin (tests/test_simulated_ranks.py).
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import COMPOSED_GRAD_K, ISDF_GRADIENT_FLOOR
from src.Base.declaration import Excitation, GroundState
from src.Base.utils.mpi_grid import (current_comm, distributed, lockstep_stats,
                                     run_simulated)
from src.gradients import factor_chain
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface
from src.properties.surface import evaluate
from src.properties.surfaces import (potential_energy_surface,
                                     reference_mean_field)
from tests.test_mpi_routes import (COMPOSED_GRAD_FLOOR, one_thread,
                                   one_thread_scatter)

BASELINE_PATH = pathlib.Path(__file__).resolve().parent / 'baseline_3f09ac0.json'
#: (a)'s pinned energies and gradients.
PINS = json.loads((BASELINE_PATH.parent / 'one_fit_pins.json').read_text())[
    'surface_comm']
#: How far (a)'s record may sit from the pins: energy (Ha), gradient (Ha/Bohr).
RECORD_MOVE = (1e-12, 5e-8)

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Where (d) evaluates: one hydrogen 0.03 A along y, so the chain runs its
#: factory, fits and differentiates at a geometry of its own.
H2O_DISPLACED = 'O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468'
SIZES = [2, 3]
#: The least of the energy bar (Ha); E_c^dRPA's frequency-loop reduction is
#: exact to 1e-11 relative (test_simulated_ranks.py).
ENERGY_FLOOR = 1e-12
#: The coordinate (atom, axis) rank 1 moves by one ulp in (d).
PERTURBED = (1, 1)
#: What (e) adds to rank 1's orbital branch, in Ha/Bohr: four orders below
#: the reproducibility floor, far above the last bits ranks differ by.
BRANCH_OFFSET = 1e-12


def chain_scf(mol):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    return mf


def chain_scf_unrun(mol):
    """The same mean field built and not run: the chain converges it over
    the ranks of the context (`converged_factory`)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    return mf


def own_water(atom=H2O):
    """This rank's own Mole; see the module docstring on `mol._env`."""
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def own_ground():
    """A fresh RPAGroundStateChain on this rank's own Mole."""
    mol = own_water()
    return RPAGroundStateChain(mol, chain_scf, mf=chain_scf(mol))


def own_bse():
    """A fresh RPABSESurface on this rank's own Mole, one mean field shared
    by its two halves."""
    mol = own_water()
    return RPABSESurface(mol, chain_scf, spin='singlet', mf=chain_scf(mol))


def own_qp():
    """A fresh RPAQPSurface on this rank's own Mole."""
    mol = own_water()
    return RPAQPSurface(mol, chain_scf, mf=chain_scf(mol))


def at_reference(surface):
    """(E, dE/dR) through the boundary a property routine uses."""
    g, e, _ = evaluate(surface, surface.mol0)
    return e, np.asarray(g)


# --------------------------------------------------------------------- (a)
@pytest.fixture(scope='module')
def baseline():
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope='module')
def baseline_water(baseline):
    spec = baseline['geometries']['water']
    return gto.M(atom=spec['atom'], basis=baseline['basis'], verbose=0)


@pytest.fixture(scope='module')
def baseline_scf(baseline):
    settings = baseline['scf']

    def factory(mol):
        mf = scf.RHF(mol).density_fit(auxbasis=settings['auxbasis'])
        mf.conv_tol = settings['conv_tol']
        mf.conv_tol_grad = settings['conv_tol_grad']
        mf.max_cycle = settings['max_cycle']
        mf.kernel()
        assert mf.converged
        return mf
    return factory


def _r0(baseline, label):
    return baseline['surfaces']['water'][label]['geometries']['R0']


@pytest.mark.parametrize('label,build', [
    ('RPAGroundStateChain', lambda mol, f: RPAGroundStateChain(mol, f)),
    ('RPABSESurface[singlet]',
     lambda mol, f: RPABSESurface(mol, f, spin='singlet')),
    ('RPAQPSurface', lambda mol, f: RPAQPSurface(mol, f)),
    ('ExcitedStateChain[singlet]',
     lambda mol, f: ExcitedStateChain(mol, f, spin='singlet'))])
def test_serially_every_surface_is_the_baseline(baseline, baseline_water,
                                                baseline_scf, label, build):
    recorded, pin = _r0(baseline, label), PINS[label]
    surface = build(baseline_water, baseline_scf)
    assert not hasattr(surface, 'comm')
    # bitwise against the pins: a single-machine gate, see (a)
    energy = surface.total_energy()
    assert energy == pin['total_energy']
    assert abs(energy - recorded['total_energy']) <= RECORD_MOVE[0]
    grad, e_g, _ = evaluate(surface, baseline_water)
    assert e_g == pin['gradient_energy']
    assert abs(e_g - recorded['gradient_energy']) <= RECORD_MOVE[0]
    grad = np.asarray(grad, float)
    assert np.array_equal(grad, np.asarray(pin['gradient'], float))
    moved = np.abs(grad - np.asarray(recorded['gradient'], float)).max()
    assert moved <= RECORD_MOVE[1], moved


# --------------------------------------------------------------------- (b)
@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('own,floor',
                         [(own_ground, ISDF_GRADIENT_FLOOR),
                          (own_bse, COMPOSED_GRAD_FLOOR),
                          (own_qp, ISDF_GRADIENT_FLOOR)],
                         ids=['RPAGroundStateChain', 'RPABSESurface',
                              'RPAQPSurface'])
def test_every_rank_evaluates_the_serial_surface(own, floor, size):
    """Every rank's energy and force the serial surface's within the anchored
    bar, since each converged its own SCF; every rank rank 0's, bitwise."""
    ref_e, ref_g = at_reference(own())
    again = one_thread(lambda: at_reference(own()))
    rep_e, rep_g = ((0.0, 0.0) if again is None else
                    (abs(again[0] - ref_e), np.abs(again[1] - ref_g).max()))
    bar_e = max(ENERGY_FLOOR, COMPOSED_GRAD_K * rep_e)
    bar_g = max(floor, COMPOSED_GRAD_K * rep_g)
    results = run_simulated(lambda comm: at_reference(own()), size)
    d_e = max(abs(e - ref_e) for e, _ in results)
    d_g = max(np.abs(g - ref_g).max() for _, g in results)
    print(f'[info] {size} ranks: force |d| {d_g:.2e} = {d_g / bar_g:.3f} of '
          f'the anchored bar {bar_g:.2e} = max({floor:.1e}, {COMPOSED_GRAD_K} '
          f'x {rep_g:.2e}) Ha/Bohr; energy |d| {d_e:.2e} of {bar_e:.2e} = '
          f'max({ENERGY_FLOOR:.0e}, {COMPOSED_GRAD_K} x {rep_e:.2e}) Ha')
    for e, g in results:
        assert abs(e - ref_e) < bar_e
        assert np.abs(g - ref_g).max() < bar_g
    for e, g in results[1:]:
        assert e == results[0][0]
        assert np.array_equal(g, results[0][1])


# --------------------------------------------------------------------- (c)
def test_the_entry_point_builds_the_direct_surface_inside_the_context():
    """`potential_energy_surface` resolves its own qp_window/scissor/outside
    (the admitted set, 'calibrate', 'scissor'), unlike `RPABSESurface`'s raw
    defaults; the directly built surface reads those settings off the
    dispatched one, so the comparison isolates the dispatch.
    """
    ground = GroundState('rpa', 'hf')
    excitation = Excitation('singlet')

    def one_rank(comm):
        mol = own_water()
        dispatched = potential_energy_surface(
            mol, chain_scf, ground_state=ground, excitation=excitation)
        ex, gr = dispatched.excited, dispatched.ground
        direct = RPABSESurface(mol, chain_scf, spin='singlet', state=0,
                               mf=gr.mf0, factorization=gr.factorization,
                               basis=gr.basis, auxbasis=gr.auxbasis,
                               counts=gr.counts, n_start=gr.n_start,
                               qp_window=ex.qp_window,
                               scissor=ex.scissor, outside=ex.outside,
                               bse_tda=ex.bse_tda, solver=ex.solver,
                               residue_route=ex.residue_route)
        return dispatched.total_energy(), direct.total_energy()

    serial = one_rank(None)
    for e_dispatch, e_direct in run_simulated(one_rank, 2):
        assert e_dispatch == e_direct
        assert abs(e_dispatch - serial[0]) < ENERGY_FLOOR


@pytest.mark.parametrize('chi0,factorization', [('space-time', 'isdf'),
                                                ('dense-qb', 'four-index')])
def test_the_entry_point_refuses_a_comm_keyword(chi0, factorization):
    """A `comm=` is a numeric keyword no realization reads, refused by name
    before any mean field is built, on the cubic row and the dense one."""
    with pytest.raises(TypeError, match='comm'):
        potential_energy_surface(own_water(), chain_scf,
                                 ground_state=GroundState('rpa', 'hf'),
                                 excitation=Excitation('singlet'), chi0=chi0,
                                 factorization=factorization, comm=object())


# --------------------------------------------------------------------- (d)
def self_converged_chain():
    """The BSE@GW chain on a factory that converges its own SCF per rank."""
    mol = own_water()
    return ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol))


def build_only_chain():
    """The BSE@GW chain whose SCF at every geometry the ranks converge
    together: a rank's own molecule would enter that reduction."""
    return ExcitedStateChain(own_water(), chain_scf_unrun)


def perturbed_evaluation(comm, perturb, build=self_converged_chain):
    """One audited evaluation of the BSE@GW chain at the displaced geometry,
    with rank 1's copy of that geometry one ulp off when `perturb`."""
    chain = build()
    here = own_water(H2O_DISPLACED)
    if perturb and comm.Get_rank() == 1:
        coords = here.atom_coords()
        coords[PERTURBED] = np.nextafter(coords[PERTURBED], np.inf)
        here.set_geom_(coords, unit='Bohr')
    with distributed(comm, audit=True):
        lockstep_stats(reset=True)
        g, e, _ = evaluate(chain, here)
        stats = lockstep_stats()
    return np.asarray(g), e, stats, here.atom_coords()


@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('build', [self_converged_chain, build_only_chain],
                         ids=['self-converging', 'build-only'])
def test_the_geometry_lockstep_repairs_a_one_ulp_rank(build, size):
    clean = run_simulated(perturbed_evaluation, size, False, build)
    out = run_simulated(perturbed_evaluation, size, True, build)
    x = clean[0][3][PERTURBED]
    ulp = np.nextafter(x, np.inf) - x
    for (g, e, stats, coords), (g0, e0, _, coords0) in zip(out, clean):
        assert e == e0 and np.array_equal(g, g0)
        assert np.array_equal(coords, coords0)       # rank 0's molecule
        assert stats['audited_calls'] > 1
        assert stats.get('disagreements', 0) == 0
    for rank, (_, _, stats, _) in enumerate(out):
        # the geometry, by the ulp itself, and nothing further down
        assert stats['mismatched_calls'] == (1 if rank == 1 else 0)
        assert stats['max_abs_diff'] == (ulp if rank == 1 else 0.0)
    for _, _, stats, _ in clean:
        assert stats['mismatched_calls'] == 0


# --------------------------------------------------------------------- (e)
def test_the_chains_own_share_of_the_gradient_is_rank_zeros(monkeypatch):
    real = factor_chain.eps_chain_gradient

    def drifted(*args, **kw):
        g_orb, diags = real(*args, **kw)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1:
            g_orb = g_orb + BRANCH_OFFSET
        return g_orb, diags

    monkeypatch.setattr(factor_chain, 'eps_chain_gradient', drifted)

    def one_rank(comm):
        mol = own_water()
        return ExcitedStateChain(mol, chain_scf,
                                 mf=chain_scf(mol)).excitation_gradient()[0]

    serial = ExcitedStateChain(own_water(), chain_scf)
    clean = serial.excitation_gradient()[0]
    scatter = one_thread_scatter(serial.mol0, serial.mf0, clean)
    bar = max(COMPOSED_GRAD_FLOOR, COMPOSED_GRAD_K * scatter)
    out = run_simulated(one_rank, 2)
    for g in out:
        assert np.array_equal(g, out[0])
    d = np.abs(out[0] - clean).max()
    print(f'[info] excitation force |d| {d:.2e} = {d / bar:.3f} of the '
          f'anchored bar {bar:.2e} = max({COMPOSED_GRAD_FLOOR:.1e}, '
          f'{COMPOSED_GRAD_K} x {scatter:.2e}) Ha/Bohr')
    assert d < bar



# --------------------------------------------------------------------- (f)
def perturbed_scf(mol):
    """`chain_scf`, with rank 1's orbital energy one ulp up and its last
    orbital's sign flipped: what another rank's own SCF may hand back."""
    mf = chain_scf(mol)
    comm = current_comm()
    if comm is not None and comm.Get_rank() == 1:
        mf.mo_energy = mf.mo_energy.copy()
        mf.mo_energy[0] = np.nextafter(mf.mo_energy[0], np.inf)
        mf.mo_coeff = mf.mo_coeff.copy()
        mf.mo_coeff[:, -1] *= -1.0
    return mf


def chain_given_mf():
    mol = own_water()
    return ExcitedStateChain(mol, chain_scf, mf=perturbed_scf(mol)).mf0


def chain_mean_field_elsewhere():
    chain = ExcitedStateChain(own_water(), chain_scf)
    here = own_water(H2O_DISPLACED)
    return chain.mean_field(here, perturbed_scf(here))[1]


def entry_point_reference():
    return reference_mean_field(own_water(), perturbed_scf, None)


def mean_field_surface():
    mol = own_water()
    return MeanFieldSurface(mol, perturbed_scf).mean_field(mol)[1]


@pytest.mark.parametrize('accept', [chain_given_mf, chain_mean_field_elsewhere,
                                    entry_point_reference, mean_field_surface],
                         ids=['chain mf=', 'chain.mean_field(mol, mf)',
                              'reference_mean_field',
                              'MeanFieldSurface._mean_field'])
def test_every_accepted_mean_field_is_rank_zeros(accept):
    def one_rank(comm):
        mf = accept()
        return np.array(mf.mo_energy), np.array(mf.mo_coeff)

    out = run_simulated(one_rank, 2)
    for eps, coeff in out[1:]:
        assert np.array_equal(eps, out[0][0])
        assert np.array_equal(coeff, out[0][1])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
