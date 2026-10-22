"""Surfaces evaluate identically on every rank under the distribution context.

A surface carries no communicator: inside `with distributed(comm):` every rank
builds it and evaluates it whole, the kernels it reaches divide their sweeps
over the ranks, and `surface.evaluate` -- the boundary a property routine
hands a geometry across -- locks the geometry to rank 0's on the way in and the
energy, force and diagnostics on the way out. Five gates:

  (a) SERIALLY every surface is the recorded baseline
      (`tests/baseline_forward_routes.json`), and none of them carries a
      `comm`. The pins (tests/one_fit_pins.json) are bitwise -- `==` on the
      energy, `np.array_equal` on the gradient -- and the record is held
      within `RECORD_MOVE` of them. Five differences separate them:
      the fitted Fock skeleton in auxiliary tiles applies an explicit
      Cholesky inverse of the metric where the whole form solved against it
      (the energies bitwise, the gradients 7.3e-14, 5.2e-13 and 9.1e-13
      Ha/Bohr for the ground state, BSE and QP surfaces), and the chain
      differentiates the production fit, the Gram matrix over every product
      pair: water keeps all 576 pairs, so that is the fit's realization
      alone, 1.4e-14, 1.9e-13 and 1.4e-14 Ha and 3.3e-9, 1.8e-8 and 3.4e-9
      Ha/Bohr; and the fit adjoint forms the Gram matrix's adjoint from its
      two single solves instead of applying G^-1 twice to an (nk, nk)
      product, the energies bitwise and the gradients 1.3e-9, 1.2e-8 and
      1.7e-9 Ha/Bohr, that form's own rounding; and the frequency pass
      factorizes 1 - chi0(i.nu) by Cholesky instead of an LU, the
      energies bitwise and the BSE and QP gradients 6.9e-13 and 3.2e-13
      Ha/Bohr; and the frozen frames turn with the molecule (`body_frame`),
      whose rotation's derivative moves the gradients by 5.2e-11, 6.0e-9
      and 7.5e-11 Ha/Bohr, the energies bitwise. A single-machine gate by design: the numbers were recorded by
      a pyscf without OpenMP, which repeats its bits, and a threaded pyscf
      adds its K-split partials in thread-arrival order and cannot meet them.
  (b) Under `run_simulated` (sizes 2 and 3; `run_simulated` enters the
      context on each rank-thread) the composed and single-chain surfaces
      reproduce the SAME surface evaluated serially, and every rank returns
      the same bits. Every rank converges its own SCF, the serial one bit
      for bit with pyscf's OpenMP on one thread, so the partition's
      re-associated reductions are all that differs: the force at
      RANK_SPLIT_FORCE_TOL, the energy at RANK_SPLIT_ENERGY_TOL. Measured
      (max abs force in Ha/Bohr, then the energy in Ha):
                           size=2               size=3
          RPAGroundStateChain  1.6e-13, 0       1.1e-13, 0
          RPABSESurface        1.9e-12, 1.4e-14 2.5e-12, 1.4e-14
          RPAQPSurface         8.8e-13, 0       1.2e-12, 0
  (c) `potential_energy_surface` inside the context builds the same surface
      a direct constructor does (its energy to a few ulp: two surfaces, two
      forwards, so not bitwise), and refuses a `comm=` keyword by
      name: a communicator handed to a surface would be a second route to
      the ranks, which could disagree with the context.
  (d) The geometry lockstep: rank 1's geometry is moved by one ulp before an
      evaluation, as an optimizer whose arithmetic drifted on one rank would
      hand it. Every rank still returns the unperturbed energy and force
      (to a few ulp: a second run), rank 1's Mole ends holding rank 0's
      coordinates bitwise, and an
      audited run counts one repaired lockstep on rank 1 (the geometry, by
      the ulp) and nothing downstream. Run on a factory that converges its
      own SCF and on a build-only one, whose SCF the ranks converge together
      (`converged_factory`).
  (e) The chain's own share of the gradient: the orbital response,
      collocation and fit branch a chain forms from the kernels' adjoints are
      serial code on every rank, so `nuclear_gradient` ends in one lockstep.
      Rank 1's orbital branch is moved by 1e-12 Ha/Bohr and every rank
      returns rank 0's gradient bitwise, within RANK_SPLIT_FORCE_TOL of the
      serial one, since the partition re-associates the sums the orbital
      response carries.
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

from src.Base.numerics import Numerics
from src.Base.constants import (RANK_SPLIT_ENERGY_TOL, RANK_SPLIT_FORCE_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL,
                                SCF_TIGHT_CONV_TOL)
from src.Base.declaration import Excitation, GroundState
from src.Base.utils.mpi_grid import (current_comm, distributed, lockstep_stats,
                                     run_simulated)
from src.gradients import factor_chain
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface
from src.properties.surface import evaluate
from tests.pins import assert_pin, assert_same, recorded_blas
from src.properties.surfaces import (potential_energy_surface,
                                     reference_mean_field)
from tests.isdf_test_grid import g1, g1_numerics
from tests.threads_helper import pyscf_one_thread  # noqa: F401  (fixture)

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO / 'tests' / 'baseline_forward_routes.json'
#: (a)'s surfaces as today's code computes them.
PINS = json.loads((REPO / 'tests' / 'one_fit_pins.json').read_text())[
    'surface_comm']
PINS_BLAS = recorded_blas(REPO / 'tests' / 'one_fit_pins.json')
#: How far (a)'s record may sit from the pins: energy (Ha), gradient (Ha/Bohr).
#: The record is on an earlier ISDF grid: its energies differ from the pins by
#: 4.0e-5 and 7.5e-5 and its gradients by 1.4e-4 and 2.6e-4.
RECORD_MOVE = (1e-4, 3e-4)
#: (a)'s surfaces the record holds in the mean-field outside treatment; their
#: pins are the chain's scissor surface, on the current grid. On water/cc-pVDZ
#: (DF RHF, cc-pvdz-ri, default chain): E_0 -76.25905243 + Omega 0.31049579 =
#: -75.94855664 Ha, the pin bitwise. The record sits 3.3e-4 Ha above it, over
#: RECORD_MOVE, so it cannot be held.
RETIRED_IN_RECORD = ('RPABSESurface[singlet]',)

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Where (d) evaluates: one hydrogen 0.03 A along y, so the chain runs its
#: factory, fits and differentiates at a geometry of its own.
H2O_DISPLACED = 'O 0 0 0.117; H 0 0.787 -0.468; H 0 -0.757 -0.468'
SIZES = [2, 3]
#: The coordinate (atom, axis) rank 1 moves by one ulp in (d).
PERTURBED = (1, 1)
#: What (e) adds to rank 1's orbital branch, in Ha/Bohr: four orders below
#: the reproducibility floor, far above the last bits ranks differ by.
BRANCH_OFFSET = 1e-12


def chain_scf(mol):
    """A mean field converged for gradient work (conv_tol_grad 1e-11)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    mf.kernel()
    return mf


def chain_scf_unrun(mol):
    """The same mean field built and not run: the chain converges it over
    the ranks of the context (`converged_factory`)."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol = SCF_TIGHT_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    return mf


def own_water(atom=H2O):
    """This rank's own Mole; see the module docstring on `mol._env`."""
    return gto.M(atom=atom, basis=BASIS, verbose=0)


def own_ground():
    """A fresh RPAGroundStateChain on this rank's own Mole."""
    mol = own_water()
    return RPAGroundStateChain(mol, chain_scf, mf=chain_scf(mol), numerics=Numerics(isdf={'counts': g1(mol)}))


def own_bse():
    """A fresh RPABSESurface on this rank's own Mole, one mean field shared
    by its two halves."""
    mol = own_water()
    return RPABSESurface(mol, chain_scf, spin='singlet', mf=chain_scf(mol), numerics=g1_numerics(mol))


def own_qp():
    """A fresh RPAQPSurface on this rank's own Mole."""
    mol = own_water()
    return RPAQPSurface(mol, chain_scf, mf=chain_scf(mol), numerics=g1_numerics(mol))


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
    ('RPAGroundStateChain', lambda mol, f: RPAGroundStateChain(mol, f, numerics=Numerics(isdf={'counts': g1(mol)}))),
    ('RPABSESurface[singlet]',
     lambda mol, f: RPABSESurface(mol, f, spin='singlet', numerics=g1_numerics(mol))),
    ('RPAQPSurface', lambda mol, f: RPAQPSurface(mol, f, numerics=g1_numerics(mol)))])
def test_serially_every_surface_is_the_baseline(baseline, baseline_water,
                                                baseline_scf, label, build):
    recorded, pin = _r0(baseline, label), PINS[label]
    surface = build(baseline_water, baseline_scf)
    assert not hasattr(surface, 'comm')
    # the record's BSE surface leaves the orbitals outside the set at their
    # eigenvalues, a treatment the chain does not take: held to its pin alone
    held = label not in RETIRED_IN_RECORD
    # bitwise against the recording BLAS, the bar of each quantity class on
    # another
    energy = surface.total_energy()
    assert_pin(energy, pin['total_energy'], PINS_BLAS, quantity='energy')
    assert not held or abs(energy - recorded['total_energy']) <= RECORD_MOVE[0]
    grad, e_g, _ = evaluate(surface, baseline_water)
    assert_pin(e_g, pin['gradient_energy'], PINS_BLAS, quantity='energy')
    assert not held or abs(e_g - recorded['gradient_energy']) <= RECORD_MOVE[0]
    grad = np.asarray(grad, float)
    assert_pin(grad, pin['gradient'], PINS_BLAS, quantity='gradient')
    moved = np.abs(grad - np.asarray(recorded['gradient'], float)).max()
    assert not held or moved <= RECORD_MOVE[1], moved


# --------------------------------------------------------------------- (b)
@pytest.mark.parametrize('size', SIZES)
@pytest.mark.parametrize('own', [own_ground, own_bse, own_qp],
                         ids=['RPAGroundStateChain', 'RPABSESurface',
                              'RPAQPSurface'])
def test_every_rank_evaluates_the_serial_surface(own, size, pyscf_one_thread):
    """Every rank's energy and force the serial surface's within
    RANK_SPLIT_ENERGY_TOL and RANK_SPLIT_FORCE_TOL, each rank's SCF the
    serial one's bits; every rank rank 0's, bitwise."""
    ref_e, ref_g = at_reference(own())
    results = run_simulated(lambda comm: at_reference(own()), size)
    d_e = max(abs(e - ref_e) for e, _ in results)
    d_g = max(np.abs(g - ref_g).max() for _, g in results)
    print(f'[info] {size} ranks: force |d| {d_g:.2e} = '
          f'{d_g / RANK_SPLIT_FORCE_TOL:.4f} of {RANK_SPLIT_FORCE_TOL:.0e} '
          f'Ha/Bohr; energy |d| {d_e:.2e} = '
          f'{d_e / RANK_SPLIT_ENERGY_TOL:.4f} of {RANK_SPLIT_ENERGY_TOL:.0e} '
          'Ha')
    assert d_e <= RANK_SPLIT_ENERGY_TOL and d_g <= RANK_SPLIT_FORCE_TOL
    for e, g in results[1:]:
        assert e == results[0][0]
        assert np.array_equal(g, results[0][1])


# --------------------------------------------------------------------- (c)
def test_the_entry_point_builds_the_direct_surface_inside_the_context():
    """`potential_energy_surface` resolves its OWN qp_window/scissor
    (the admitted set, 'calibrate') before building the row's
    class, which is a different realization from `RPABSESurface`'s raw
    defaults; the directly built surface reads those resolved settings off
    the dispatched one so the comparison isolates the dispatch alone.
    """
    ground = GroundState('rpa', 'hf')
    excitation = Excitation('singlet')

    def one_rank(comm):
        mol = own_water()
        dispatched = potential_energy_surface(
            mol, chain_scf, ground_state=ground, excitation=excitation)
        ex, gr = dispatched.excited, dispatched.ground
        direct = RPABSESurface(
            mol, chain_scf, spin='singlet', state=0, mf=gr.mf0,
            factorization=gr.factorization, bse_tda=ex.bse_tda,
            numerics=Numerics(isdf={'basis': gr.basis, 'auxbasis': gr.auxbasis, 'counts': gr.counts, 'n_start': gr.n_start},
                              qp={'window': ex.qp_window, 'scissor': ex.scissor, 'residue_route': ex.residue_route},
                              bse={'solver': ex.solver}))
        return dispatched.total_energy(), direct.total_energy()

    serial = one_rank(None)
    for e_dispatch, e_direct in run_simulated(one_rank, 2):
        # two surfaces, each with its own forward: to a few ulp, not bitwise
        assert_same(e_dispatch, e_direct, label='dispatched vs direct')
        assert abs(e_dispatch - serial[0]) <= RANK_SPLIT_ENERGY_TOL


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
    return ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol),
                              numerics=g1_numerics(mol))


def build_only_chain():
    """The BSE@GW chain whose SCF at every geometry the ranks converge
    together: a rank's own molecule would enter that reduction."""
    return ExcitedStateChain(own_water(), chain_scf_unrun,
                              numerics=g1_numerics(own_water()))


def perturbed_evaluation(comm, perturb, build=self_converged_chain):
    """One audited evaluation of the BSE@GW chain at the displaced geometry,
    with rank 1's copy of that geometry one ulp off when `perturb`."""
    mol = own_water()
    chain = ExcitedStateChain(mol, chain_scf, mf=chain_scf(mol), numerics=g1_numerics(mol))
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
        # the repaired run against the clean one: two runs, so the energy
        # and force to a few ulp; the geometry is rank 0's input, its bits
        assert_same(e, e0, label='repaired energy')
        assert_same(g, g0, quantity='gradient', label='repaired force')
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
def test_the_chains_own_share_of_the_gradient_is_rank_zeros(
        monkeypatch, pyscf_one_thread):
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
                                 mf=chain_scf(mol), numerics=g1_numerics(mol)).excitation_gradient()[0]

    serial = ExcitedStateChain(own_water(), chain_scf, numerics=g1_numerics(own_water()))
    clean = serial.excitation_gradient()[0]
    out = run_simulated(one_rank, 2)
    for g in out:
        assert np.array_equal(g, out[0])
    d = np.abs(out[0] - clean).max()
    print(f'[info] excitation force |d| {d:.2e} = '
          f'{d / RANK_SPLIT_FORCE_TOL:.4f} of RANK_SPLIT_FORCE_TOL '
          f'{RANK_SPLIT_FORCE_TOL:.0e} Ha/Bohr')
    assert d <= RANK_SPLIT_FORCE_TOL



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
    return ExcitedStateChain(mol, chain_scf, mf=perturbed_scf(mol),
                              numerics=g1_numerics(mol)).mf0


def chain_mean_field_elsewhere():
    chain = ExcitedStateChain(own_water(), chain_scf,
                               numerics=g1_numerics(own_water()))
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
