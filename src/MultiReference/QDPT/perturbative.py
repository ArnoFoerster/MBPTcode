"""Perturbative (QDPT/Wick's-theorem) downfolding: builds g_eff for an active
space via the generated evaluators in generated_evaluators(_spinfree).py.

Only builds effective Hamiltonians; solving (FCI, exact diagonalization,
DMRG, ...) lives in src/Solvers/active_space.py.
"""
import importlib

import numpy as np

from src.Base.pyscf_interface import get_system_data, get_system_data_spatial
from src.Solvers.active_space import solve


def build_effective_hamiltonian(mol, mf, n_occ_spatial=0, n_act_spatial=2,
                                max_pt_order=2, max_vertex=3,
                                shift=None, spin_free=False, evaluator='auto',
                                active_space=None):
    """Build the QDPT downfolded effective Hamiltonian for an active space. Returns (g_eff, space).

    g_eff: dict body-rank n -> rank-n interaction tensor over active orbitals
    (g_eff[2] is antisymmetrized physicist <pq||rs>), spin-orbital unless
    spin_free=True. space: n_orbs, n_elec_active, act_idx, act_occ_idx,
    act_vir_idx, spin_free, active_space.
    shift: Brillouin-Wigner energy shift in the PT denominators (None = plain).

    evaluator: 'auto' or 'generated', the generated evaluators (equivalent);
    'handwritten', the amplitude-form route with Epstein-Nesbet dressing, is
    not available in this code base and raises.
    active_space: an src.Base.active_space.ActiveSpace in place of the two
    contiguous window counts; its MOs are permuted to match, which is the only
    way to downfold onto a window the energy ordering does not produce.
    """
    if evaluator not in ('auto', 'generated', 'handwritten'):
        raise ValueError(f"evaluator={evaluator!r}; expected 'auto', "
                         "'generated' or 'handwritten'")
    if evaluator == 'handwritten':
        raise NotImplementedError(
            "the hand-written amplitude-form downfolding (with Epstein-Nesbet "
            "dressing) is not available; use evaluator='generated'.")

    if spin_free:
        g_ints, h1, eps, occ_idx, act_idx, virt_idx = get_system_data_spatial(
            mol, mf, n_occ_spatial, n_act_spatial, active_space=active_space)
        n_orbs = len(act_idx)  # spatial orbitals
        n_elec_active = mol.nelectron - 2 * len(occ_idx)

        nocc_total = mol.nelectron // 2
    else:
        g_ints, h1, eps, occ_idx, act_idx, virt_idx = get_system_data(
            mol, mf, n_occ_spatial, n_act_spatial, active_space=active_space)
        n_orbs = len(act_idx)  # spin orbitals
        n_elec_active = mol.nelectron - len(occ_idx)

        nocc_total = 2 * (mol.nelectron // 2)
    act_occ_idx = np.array([i for i in act_idx if i < nocc_total], dtype=int)
    act_vir_idx = np.array([i for i in act_idx if i >= nocc_total], dtype=int)

    eval_fn = _get_evaluator(max_pt_order, max_vertex, spin_free)
    g_eff = eval_fn(g_ints, h1, eps, occ_idx, act_idx, virt_idx,
                    act_occ_idx, act_vir_idx, n_orbs, shift=shift)

    space = {
        'n_orbs': n_orbs,
        'n_elec_active': n_elec_active,
        'act_idx': act_idx,
        'act_occ_idx': act_occ_idx,
        'act_vir_idx': act_vir_idx,
        'spin_free': spin_free,
        'active_space': active_space,
    }
    return g_eff, space


def _get_evaluator(max_pt_order, max_vertex, spin_free):
    """Look up the generated evaluator function for the requested PT order and
    vertex rank, importing the (large) generated module lazily."""
    available = {(2, 2), (2, 3), (3, 2), (3, 3)}
    if (max_pt_order, max_vertex) not in available:
        raise NotImplementedError(
            f"Evaluator for PT order {max_pt_order} and max_vertex {max_vertex} "
            f"(spin_free={spin_free}) is not generated; only "
            f"{sorted(available)} are available.")
    suffix = '_spinfree' if spin_free else ''
    module = importlib.import_module(
        f'src.MultiReference.QDPT.generated_evaluators{suffix}')
    return getattr(module, f'eval_pt{max_pt_order}_v{max_vertex}{suffix}')


def run_pipeline(max_pt_order=2, max_vertex=3, symmetrize=True, mol=None,
                 mf=None, n_occ_spatial=0, n_act_spatial=2,
                 solver='openfermion', shift=None,
                 spin_free=False, evaluator='auto', nroots=1,
                 active_space=None, dmrg_bond_dim=500, dmrg_n_sweeps=20):
    """Downfold and solve in one call (build_effective_hamiltonian +
    src.Solvers.active_space.solve).

    nroots is passed to the solver, so a dynamical (Brillouin-Wigner) solve
    can follow an excited root."""
    g_eff, space = build_effective_hamiltonian(
        mol, mf, n_occ_spatial=n_occ_spatial, n_act_spatial=n_act_spatial,
        max_pt_order=max_pt_order, max_vertex=max_vertex, shift=shift,
        spin_free=spin_free, evaluator=evaluator, active_space=active_space)

    return solve(solver, g_eff, space['n_orbs'], space['n_elec_active'],
                 space['act_idx'], space['act_occ_idx'], space['act_vir_idx'],
                 symmetrize=symmetrize, spin_free=spin_free, nroots=nroots,
                 dmrg_bond_dim=dmrg_bond_dim, dmrg_n_sweeps=dmrg_n_sweeps)
