"""A TADF rate calculation's two ranked paths at 2 simulated ranks against
serial.

A rate calculation evaluates every geometry's property block inside the
launch's distribution and stripes the S0 Hessian's displaced forces over the
ranks with `mpi_map`. Water/cc-pVDZ:

  * the per-geometry block on the production surface (ISDF-K LRC-wPBEh G1,
    SOP residues, frontier set with the scissor outside, Davidson, grid
    adjoint, sliced factors and the row fit, in the gas phase and in
    PCM(toluene)): one shared evaluation with the
    S1 force and the S1-S2 / T1-T2 numerators, then the couplings off it
    (`analytic_coupling`, `analytic_ground_coupling`), the spin-orbit table
    and the second-order spin-vibronic derivative. Every rank holds rank 0's
    numbers to a few ulp (each rank computes them). Against serial the comparison is split at the pole
    fit (below); the couplings and the derivative up to the sign of the
    Casida vectors they carry, which the Davidson chooses differently at one
    and two ranks (measured: d_0S flips) and which every rate squares.
  * the finite-difference Hessian of the ISDF-K Hartree-Fock force on
    ISDF_HESSIAN_MIN_GRID ('GH', which `numerical_hessian` requires of an
    ISDF-K mean field) with `mpi_map`: each rank converges its own displaced SCFs, cold, so the
    Hessian is the serial one to a few ulp.

THE POLE FIT. A rank split re-associates the row fit's reductions, so the
imaginary-axis data wc each quasiparticle's pole model is fitted to differs
from the serial wc in its last bits (gas phase: 2e-14 absolute, 1e-13 of
|wc|max). Poles the vector fit clips onto the gap can coalesce; unmerged,
its least squares goes rank-deficient and the ranks land on another pole set
(the S0 force 4.2e-8 Ha/Bohr from serial). `SOP_POLE_MERGE_TOL` merges
coalesced poles, and the serial and ranked forces agree to 8e-13, so the gate
is direct, in two parts:

  (i)  the fit's input: every fit on the ranks has a serial partner whose wc
       it equals to FIT_REALIZATION_OBSERVABLE_REL_TOL of |wc|max, the bar of
       what reads the one fit through its products;
  (ii) the ranked block is the serial block to RANK_TOL.

What breaks the gate: rank 1's half of the frequency pass scaled by 1 + 1e-6
(a reduction that is wrong on one rank) breaks (i) and (ii)
(`test_a_wrong_rank_fails_the_gate`).
"""
import functools
import threading

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import (FIT_REALIZATION_OBSERVABLE_REL_TOL,
                                ISDF_HESSIAN_MIN_GRID,
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.isdf_jk import isdf_jk
from src.Base.separable_ri import resolve_isdf_grid
from src.Base.solvent_screening import SolventScreening
from src.Base.utils.mpi_grid import (current_comm, distributed, mpi_map,
                                     run_simulated)
from src.Base.constants import NUCLEAR_FD_STEP, PIN_CROSS_BLAS_GRADIENT_BAR
from src.gradients import qp_space_time
from src.gradients.derivative_coupling import (analytic_coupling,
                                               analytic_ground_coupling)
from src.gradients.state_manifold import StateManifold
from src.properties.excitations import SurfaceSpec, surface_of
from src.properties.hessian import numerical_hessian
from src.properties.spin_orbit import spin_orbit_couplings
from src.properties.spin_vibronic import spin_vibronic_coupling
from src.properties.surfaces import reference_mean_field
from tests.blas_bar import assert_same_within
from tests.pins import assert_same
from tests.threads_helper import pyscf_one_thread  # noqa: F401  (fixture)

BASIS, AUX = 'cc-pvdz', 'cc-pvdz-ri'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
MAX_MEMORY = 4000
S0, S1, T0, T1 = ('singlet', 0), ('singlet', 1), ('triplet', 0), ('triplet', 1)
#: serial against ranked on the row fit and the fixed-order reductions
RANK_TOL = 1e-9
#: the planted defect of the negative control, relative
DEFECT = 1e-6
#: the quantities that carry the sign of a Casida vector
GAUGED = ('d01', 'd_s', 'd_t', 'dv')

PRODUCTION = SurfaceSpec(GroundState('dft', 'lrc-wpbeh'), environment=None,
                         chi0='space-time', residues='sop', solver='davidson',
                         factorization='isdf',
                         qp_states=QPStates(kind='frontier'),
                         numerics={'grid_accuracy': 'G1', 'sliced': True,
                                   'fit': 'rows', 'bse_adjoint': 'grid',
                                   'nroots': 4, 'track': 'overlap'})


def molecule():
    return gto.M(atom=H2O, basis=BASIS, verbose=0, max_memory=MAX_MEMORY)


def grid(level):
    return resolve_isdf_grid(level, BASIS, ['H', 'O'], auxbasis=AUX)


def isdf_lrc(mol):
    """The production mean field; built and not run over ranks."""
    base = dft.RKS(mol, xc='lrc-wpbeh')
    base.max_memory = MAX_MEMORY
    counts, n_start = grid('G1')
    mf = isdf_jk(base, auxbasis=AUX, counts=counts, n_start=n_start)
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.max_cycle = 200
    comm = current_comm()
    if comm is None or comm.Get_size() == 1:
        mf.kernel()
    return mf


def isdf_hf_hessian(mol):
    """A cold Hartree-Fock ISDF-K SCF on the Hessian's grid, always run."""
    counts, n_start = grid(ISDF_HESSIAN_MIN_GRID)
    mf = isdf_jk(scf.RHF(mol), auxbasis=AUX, counts=counts, n_start=n_start)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
    mf.kernel()
    return mf


def declaration(environment):
    """The driver's chain: sliced factors and the row fit, in the gas phase
    and in a continuum alike."""
    if environment is None:
        return PRODUCTION
    return SurfaceSpec(PRODUCTION.ground_state, environment=environment,
                       chi0='space-time', residues='sop', solver='davidson',
                       factorization='isdf',
                       qp_states=QPStates(kind='frontier'),
                       numerics=dict(PRODUCTION.numerics))


def property_block(solvent=None):
    """One geometry's block as the driver computes it."""
    mol = molecule()
    env = None if solvent is None else SolventScreening(mol, solvent=solvent)
    spec = declaration(env)
    mf0 = reference_mean_field(mol, isdf_lrc, env)
    excited = surface_of(spec, Excitation('singlet', root=1), mol,
                         isdf_lrc, mf=mf0)
    man = StateManifold(excited, states=(S0, S1, T0, T1))
    ev = man.evaluate(mol, mf0, gradients=(S0,),
                      couplings=((S0, S1), (T0, T1)))
    nocc = mol.nelectron // 2
    soc = spin_orbit_couplings(ev.mf, ev.mol, nocc, ev.spectrum['singlet'],
                               ev.spectrum['triplet'], nroots=2)
    sv = spin_vibronic_coupling(ev, soc['h_mo'], S0, T0, singlet_paths=(S1,),
                                triplet_paths=(T1,))
    return {'force': ev.gradient[S0], 'omega': np.array([ev.omega[k] for k in
                                                         (S0, S1, T0, T1)]),
            'd01': analytic_ground_coupling(None, S0, evaluation=ev)[0],
            'd_s': analytic_coupling(None, S0, S1, evaluation=ev)[0],
            'd_t': analytic_coupling(None, T0, T1, evaluation=ev)[0],
            'socme': np.abs(soc['socme']), 'dv': sv['dv_cart']}


class PoleFits:
    """Every pole fit a block makes, on any rank: its wc and its poles."""

    def __init__(self):
        self.fits, self.lock = [], threading.Lock()
        self.fit = qp_space_time.sop_from_wc

    def __call__(self, wc, nu_points, eps, nocc, **kw):
        poles, amp = self.fit(wc, nu_points, eps, nocc, **kw)
        with self.lock:
            self.fits.append((np.array(wc), poles.copy()))
        return poles, amp


def partner(wc, fits):
    """The fit in `fits` whose wc is nearest `wc` (same shape)."""
    same = [f for f in fits if f[0].shape == np.shape(wc)]
    return min(same, key=lambda f: float(np.abs(f[0] - wc).max()))


def block_with(monkeypatch, fits, solvent, ranks=None):
    """The property block serially, or on `ranks` simulated ranks, its pole
    fits recorded by `fits`."""
    with monkeypatch.context() as patch:
        patch.setattr(qp_space_time, 'sop_from_wc', fits)
        if ranks is None:
            with distributed(None):
                return property_block(solvent)
        return run_simulated(lambda comm: property_block(solvent), ranks)


def verdicts(serial, serial_fits, ranked, ranked_fits):
    """Parts (i) and (ii) of the module docstring as ratios to their bars
    (passing at <= 1)."""
    assert len(ranked_fits) == len(serial_fits) > 0
    wc_ratio = 0.0
    for wc, _ in serial_fits:
        other = partner(wc, ranked_fits)[0]
        bar = FIT_REALIZATION_OBSERVABLE_REL_TOL * float(np.abs(wc).max())
        wc_ratio = max(wc_ratio, float(np.abs(other - wc).max()) / bar)
    rest = 0.0
    for name, want in serial.items():
        got = ranked[name]
        # a Casida vector's sign is the eigensolver's choice; a coupling and
        # an element's derivative carry it, every rate their squares
        sign = 1.0 if np.sum(got * want) >= 0 else -1.0
        if name in GAUGED:
            got = sign * got
        else:
            assert sign > 0, name
        rest = max(rest, float(np.abs(got - want).max()) / RANK_TOL)
    return wc_ratio, rest


@pytest.mark.parametrize('solvent', [None, 'toluene'])
def test_property_block_over_two_ranks(solvent, pyscf_one_thread,
                                       monkeypatch):
    serial_fits, ranked_fits = PoleFits(), PoleFits()
    serial = block_with(monkeypatch, serial_fits, solvent)
    ranked = block_with(monkeypatch, ranked_fits, solvent, ranks=2)
    for name in serial:
        for r, got in enumerate(ranked):
            assert_same(got[name], ranked[0][name], quantity='root',
                        label=(name, r))
    wc_ratio, rest = verdicts(serial, serial_fits.fits, ranked[0],
                              ranked_fits.fits)
    print(f'\n{solvent}: wc {wc_ratio:.2e}, serial vs ranked {rest:.2e} of '
          f'their bars; the forces differ '
          f'{float(np.abs(ranked[0]["force"] - serial["force"]).max()):.2e}')
    assert wc_ratio <= 1.0
    assert rest <= 1.0


def test_a_wrong_rank_fails_the_gate(pyscf_one_thread, monkeypatch):
    """Rank 1's half of the frequency pass off by DEFECT, relative: the
    ranks still agree with each other (the sum they reduce is the same),
    and parts (i) and (ii) both fail."""
    real = qp_space_time.cd_screening_contraction_multi

    def wrong(*args, **kwargs):
        out = real(*args, **kwargs)
        comm = current_comm()
        if comm is not None and comm.Get_rank() == 1:
            out = [w * (1.0 + DEFECT) for w in out]
        return out

    serial_fits, ranked_fits = PoleFits(), PoleFits()
    serial = block_with(monkeypatch, serial_fits, None)
    monkeypatch.setattr(qp_space_time, 'cd_screening_contraction_multi',
                        wrong)
    ranked = block_with(monkeypatch, ranked_fits, None, ranks=2)
    wc_ratio, rest = verdicts(serial, serial_fits.fits, ranked[0],
                              ranked_fits.fits)
    print(f'\nplanted {DEFECT:g}: wc {wc_ratio:.2e}, serial vs ranked '
          f'{rest:.2e} of their bars')
    assert wc_ratio > 1.0
    assert rest > 1.0


def test_hessian_displacements_over_two_ranks(pyscf_one_thread):
    mol = molecule()
    with distributed(None):
        serial = numerical_hessian(mol, isdf_hf_hessian)

    def rank(comm):
        return numerical_hessian(mol, isdf_hf_hessian,
                                 map_fn=functools.partial(mpi_map, comm=comm))

    # Each rank converges its own displaced SCFs: the Hessian is the serial one
    # to a few ulp on the recording BLAS. On another BLAS every displaced
    # gradient moves by the gradient class bar and the central difference
    # divides that by 2 * step.
    bar = PIN_CROSS_BLAS_GRADIENT_BAR / (2 * NUCLEAR_FD_STEP)
    for r, got in enumerate(run_simulated(rank, 2)):
        assert_same_within(got, serial, bar, label=r)
