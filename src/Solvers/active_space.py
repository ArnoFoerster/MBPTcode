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

'pyhast' (external UCC), 'pdaggerq' (generated CCSD/UCC3/UCC4) and 'ccsdt'
(spin-orbital CCSDT) are recognized but not available through this driver:
each needs a separate, sizeable dependency (an external UCC package, or the
CCSDT subsystem) and none has an end-to-end test here.

The first two adapters, and 'dmrg', are all over the ActiveSpaceSolver
interface of active_space_solver.py.
"""
import numpy as np

from src.Solvers.active_space_solver import (Block2DMRGSolver,
                                             ExactDiagonalizationSolver,
                                             PyscfFCISolver)
from src.Solvers.exact_diagonalization import spatial_to_spin_orbitals

_UNAVAILABLE_SOLVERS = ('pyhast', 'pdaggerq', 'ccsdt', 'CCSDT')
# |element| below which a vertex counts as absent, or a spin/Hermiticity
# residual as round-off
_VERTEX_TOL = 1e-10


def _spatial_active_hamiltonian(g_eff, n_spin_orbs, n_elec_active, spin_free,
                                symmetrize, solver):
    """(h1_spatial, eri_spatial, norb, (nocca, noccb)) for a spatial-orbital,
    chemist-notation-eri ActiveSpaceSolver (PyscfFCISolver, Block2DMRGSolver)
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
        if biggest > _VERTEX_TOL:
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
        if residual > _VERTEX_TOL * max(1.0, float(np.max(np.abs(g_eff[2])))):
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
        if asym > _VERTEX_TOL:
            raise ValueError(
                f"solver={solver!r} needs a Hermitian Hamiltonian, and this "
                f"g_eff is not (anti-Hermitian part up to {asym:.2e}); pass "
                f"symmetrize=True for its Hermitian part, or use "
                f"solver='openfermion' with symmetrize=False for the "
                f"non-Hermitian spectrum.")
    return h1_spatial, eri_spatial, norb, (nocca, noccb)


def solve(solver, g_eff, n_spin_orbs, n_elec_active, act_idx, act_occ_idx,
         act_vir_idx, symmetrize=True, spin_free=False, nroots=1,
         dmrg_bond_dim=500, dmrg_n_sweeps=20):
    """
    Unified solver interface for downfolded effective Hamiltonians.

    symmetrize: solve the Hermitian part of g_eff. With symmetrize=False the
    determinant-basis solver returns the (complex) spectrum of g_eff as it is,
    and 'pyscf_fci' / 'dmrg' refuse a non-Hermitian g_eff. Those two take a
    one- and two-body Hamiltonian only: a non-zero three-body vertex is an
    error, never dropped (build with max_vertex=2, or use 'openfermion').
    nroots: how many states the eigenvalue backends return.
    dmrg_bond_dim, dmrg_n_sweeps: Block2DMRGSolver knobs, used only when
    solver='dmrg'.
    """
    if solver in _UNAVAILABLE_SOLVERS:
        raise NotImplementedError(
            f"solver={solver!r} is not available through this driver (it "
            "needs an external UCC package or the CCSDT subsystem); use "
            "'pyscf_fci', 'dmrg', or the default exact-diagonalization solver.")

    if spin_free and solver != 'pyscf_fci' and solver != 'dmrg':
        # Map spatial to spin-orbitals, then use the standard spin-orbital solvers.
        n_spatial_orbs = n_spin_orbs
        g_eff = spatial_to_spin_orbitals(g_eff, n_spatial_orbs)
        n_spin_orbs = 2 * n_spatial_orbs

        act_idx = np.arange(n_spin_orbs)
        act_occ_idx = np.arange(n_elec_active)
        act_vir_idx = np.arange(n_elec_active, n_spin_orbs)

    if solver == 'pyscf_fci':
        h1_spatial, eri_spatial, norb, nelec = _spatial_active_hamiltonian(
            g_eff, n_spin_orbs, n_elec_active, spin_free, symmetrize, solver)
        # spin=None: every state at this Sz, the space direct_spin1 spans here
        e_act = PyscfFCISolver().roots(h1_spatial, eri_spatial, norb, nelec,
                                       nroots=nroots, spin=None)[0]
        return g_eff.get(0, 0.0) + e_act

    elif solver == 'dmrg':
        h1_spatial, eri_spatial, norb, nelec = _spatial_active_hamiltonian(
            g_eff, n_spin_orbs, n_elec_active, spin_free, symmetrize, solver)
        nocca, noccb = nelec
        # spin = n_alpha - n_beta: the lowest total S consistent with this Sz
        # (an SU2 target is a total spin, unlike pyscf_fci's spin=None).
        e_act, _ = Block2DMRGSolver(bond_dim=dmrg_bond_dim,
                                    n_sweeps=dmrg_n_sweeps).roots(
            h1_spatial, eri_spatial, norb, nelec, nroots=nroots,
            spin=nocca - noccb)
        return g_eff.get(0, 0.0) + e_act

    else:
        return ExactDiagonalizationSolver(symmetrize=symmetrize).spectrum(
            g_eff, n_spin_orbs, n_elec_active)[0]
