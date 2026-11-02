"""Unified solver interface for (downfolded) active-space Hamiltonians.

solve() takes an effective Hamiltonian g_eff (dict body-rank -> tensor, as
produced by src/MultiReference/QDPT/perturbative.py) and dispatches to one of
the many-body solver backends:
  'openfermion' (default) -- explicit determinant-basis matrix + exact
                             diagonalization (ExactDiagonalizationSolver)
  'pyscf_fci'             -- pyscf's FCI at fixed Sz (PyscfFCISolver)
  'dmrg'                  -- DMRG through block2, SU2 spin-adapted
                             (Block2DMRGSolver); reaches active spaces the
                             two exact engines above cannot. block2 must be
                             imported before numpy in the calling process (see
                             docs/installation.md) -- safe from a fresh
                             subprocess, not safe after another backend has
                             already run in the same process.
  'ccsdt', 'ccsdtq'       -- high-order coupled cluster on a Hartree-Fock
                             reference in the active space
                             (CoupledClusterSolver, order 3 or 4), energy
                             only, one state: the lowest singlet when
                             n_alpha = n_beta, else the lowest high-spin
                             state; for active spaces too large for FCI.

Any other name is refused (`SOLVERS`).

Every backend but 'openfermion' goes through the ActiveSpaceSolver
interface of active_space_solver.py.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.Base.numerics.constants import (ACTIVE_SPACE_VERTEX_TOL, DMRG_BOND_DIM,
                                         DMRG_N_SWEEPS)
from src.Solvers.active_space_solver import (Block2DMRGSolver,
                                             CoupledClusterSolver,
                                             ExactDiagonalizationSolver,
                                             PyscfFCISolver)
from src.Solvers.exact_diagonalization import spatial_to_spin_orbitals

#: The coupled-cluster backends and their excitation order.
CC_ORDERS = {'ccsdt': 3, 'ccsdtq': 4}


def _spatial_active_hamiltonian(g_eff, n_spin_orbs, n_elec_active, spin_free,
                                symmetrize, solver):
    """(h1_spatial, eri_spatial, norb, (nocca, noccb)) for a spatial-orbital,
    chemist-notation-eri ActiveSpaceSolver (PyscfFCISolver, Block2DMRGSolver,
    CoupledClusterSolver)
    from a downfolded g_eff (spin-orbital unless spin_free), the Hamiltonian
    unchanged: these engines take any Hermitian one- and two-body operator,
    with no 8-fold integral symmetry required.

    Refused rather than altered: a three-body (or higher) vertex they cannot
    carry; a spin-orbital g_eff that is not spin-adapted, whose same-spin
    blocks would be lost on folding to spatial orbitals; and a non-Hermitian
    g_eff unless symmetrize=True, which takes its Hermitian part, as the
    determinant-basis solver does."""
    for k in sorted(k for k in g_eff if isinstance(k, int) and k >= 3):
        biggest = float(np.max(np.abs(g_eff[k]))) if np.size(g_eff[k]) else 0.0
        if biggest > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} takes a one- and two-body Hamiltonian, but "
                f"g_eff[{k}] has elements up to {biggest:.2e}; build g_eff with "
                f"max_vertex=2, or use solver='openfermion', which carries it.")
    if spin_free:
        h1_spatial = g_eff[1]
        eri_spatial = g_eff[2].transpose(0, 2, 1, 3)
        norb = n_spin_orbs
        nocca = n_elec_active // 2
        noccb = n_elec_active - nocca
    else:
        alpha = np.arange(0, n_spin_orbs, 2)
        beta = np.arange(1, n_spin_orbs, 2)
        norb = n_spin_orbs // 2
        h1_spatial = g_eff[1][np.ix_(alpha, alpha)]
        g2_ab = g_eff[2][np.ix_(alpha, beta, alpha, beta)]          # spatial <pq|rs>
        folded = spatial_to_spin_orbitals({1: h1_spatial, 2: g2_ab}, norb)
        residual = max(float(np.max(np.abs(folded[1] - g_eff[1]))),
                       float(np.max(np.abs(folded[2] - g_eff[2]))))
        if residual > ACTIVE_SPACE_VERTEX_TOL * max(1.0, float(np.max(np.abs(g_eff[2])))):
            raise ValueError(
                f"solver={solver!r} works on spatial orbitals, but this "
                f"spin-orbital g_eff is not spin-adapted (it differs from the "
                f"one its alpha-beta block implies by {residual:.2e}); folding "
                f"it would discard its independent spin blocks. Use "
                f"solver='openfermion'.")
        eri_spatial = g2_ab.transpose(0, 2, 1, 3)
        nocca = sum(1 for i in range(n_elec_active) if i % 2 == 0)
        noccb = sum(1 for i in range(n_elec_active) if i % 2 == 1)

    # (pq|rs) <-> (rs|pq) relabels the same normal-ordered operator
    eri_spatial = 0.5 * (eri_spatial + eri_spatial.transpose(2, 3, 0, 1))
    if symmetrize:
        h1_spatial = 0.5 * (h1_spatial + h1_spatial.T)
        eri_spatial = 0.5 * (eri_spatial + eri_spatial.transpose(1, 0, 3, 2))
    else:
        asym = max(float(np.max(np.abs(h1_spatial - h1_spatial.T))),
                   float(np.max(np.abs(eri_spatial - eri_spatial.transpose(1, 0, 3, 2)))))
        if asym > ACTIVE_SPACE_VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} needs a Hermitian Hamiltonian, and this "
                f"g_eff is not (anti-Hermitian part up to {asym:.2e}); pass "
                f"symmetrize=True for its Hermitian part, or use "
                f"solver='openfermion' with symmetrize=False for the "
                f"non-Hermitian spectrum.")
    return h1_spatial, eri_spatial, norb, (nocca, noccb)


@dataclass(frozen=True)
class _Request:
    """One `solve` call, as every backend reads it."""
    g_eff: dict
    n_spin_orbs: int
    n_elec_active: int
    spin_free: bool
    symmetrize: bool
    nroots: Optional[int]
    dmrg_bond_dim: int
    dmrg_n_sweeps: int

    def spatial(self, solver):
        return _spatial_active_hamiltonian(
            self.g_eff, self.n_spin_orbs, self.n_elec_active, self.spin_free,
            self.symmetrize, solver)


def _pyscf_fci(r):
    h1, eri, norb, nelec = r.spatial('pyscf_fci')
    # spin=None: every state at this Sz, the space direct_spin1 spans here
    return r.g_eff.get(0, 0.0) + PyscfFCISolver().roots(
        h1, eri, norb, nelec, nroots=r.nroots, spin=None)[0]


def _dmrg(r):
    h1, eri, norb, nelec = r.spatial('dmrg')
    nocca, noccb = nelec
    # spin = n_alpha - n_beta: the lowest total S consistent with this Sz
    # (an SU2 target is a total spin, unlike pyscf_fci's spin=None).
    return r.g_eff.get(0, 0.0) + Block2DMRGSolver(
        bond_dim=r.dmrg_bond_dim, n_sweeps=r.dmrg_n_sweeps).roots(
        h1, eri, norb, nelec, nroots=r.nroots, spin=nocca - noccb)[0]


def _coupled_cluster(name):
    def run(r):
        h1, eri, norb, nelec = r.spatial(name)
        return r.g_eff.get(0, 0.0) + CoupledClusterSolver(
            order=CC_ORDERS[name]).roots(h1, eri, norb, nelec,
                                         nroots=r.nroots, spin=None)[0]
    return run


def _exact(r):
    g_eff, n_spin_orbs = r.g_eff, r.n_spin_orbs
    if r.spin_free:
        g_eff = spatial_to_spin_orbitals(g_eff, n_spin_orbs)
        n_spin_orbs = 2 * n_spin_orbs
    # the whole determinant-basis spectrum, a three-body vertex carried and
    # the constant inside the matrix; the lowest nroots of it
    energies = ExactDiagonalizationSolver(symmetrize=r.symmetrize).spectrum(
        g_eff, n_spin_orbs, r.n_elec_active)[0]
    return energies if r.nroots is None else energies[:r.nroots]


#: The backends `solve` dispatches to, by name; any other name is refused.
SOLVERS = {'openfermion': _exact, 'pyscf_fci': _pyscf_fci, 'dmrg': _dmrg,
           'ccsdt': _coupled_cluster('ccsdt'),
           'ccsdtq': _coupled_cluster('ccsdtq')}


def solve(solver, g_eff, n_spin_orbs, n_elec_active, symmetrize=True,
          spin_free=False, nroots=1, dmrg_bond_dim=DMRG_BOND_DIM,
          dmrg_n_sweeps=DMRG_N_SWEEPS):
    """
    The lowest `nroots` energies of a downfolded effective Hamiltonian g_eff,
    ascending, its constant g_eff[0] included: the same contract for every
    backend (`SOLVERS`).

    n_spin_orbs: the active orbitals g_eff is written on (spin orbitals, or
    spatial orbitals when spin_free); n_elec_active: the active electrons.
    symmetrize: solve the Hermitian part of g_eff. With symmetrize=False the
    determinant-basis solver returns the (complex) spectrum of g_eff as it is,
    and 'pyscf_fci' / 'dmrg' / 'ccsdt' / 'ccsdtq' refuse a non-Hermitian g_eff.
    Those take a one- and two-body Hamiltonian only: a non-zero three-body
    vertex is an error, never dropped (build with max_vertex=2, or use
    'openfermion').
    nroots: how many states to return (None: every state the backend can
    reach, the whole determinant spectrum for 'openfermion'). The
    coupled-cluster backends produce one and refuse more: the lowest singlet
    when the active electrons are paired (RHF reference), else the lowest
    state of spin (n_alpha - n_beta) / 2 (UHF reference). 'pyscf_fci' and
    'openfermion' return the lowest states at this Sz of any spin, so where a
    triplet lies lowest they and the coupled-cluster backends differ by
    design.
    dmrg_bond_dim, dmrg_n_sweeps: Block2DMRGSolver knobs, used only when
    solver='dmrg'.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver={solver!r} is not a backend of this driver; "
                         f"choose one of {tuple(SOLVERS)}.")
    return SOLVERS[solver](_Request(g_eff, n_spin_orbs, n_elec_active,
                                    spin_free, symmetrize, nroots,
                                    dmrg_bond_dim, dmrg_n_sweeps))
