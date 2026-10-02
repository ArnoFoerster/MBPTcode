"""Determinant-basis matrix of a spin-orbital effective Hamiltonian, and the
spatial-to-spin-orbital embedding that feeds it.

The Hamiltonian is built as a FermionOperator and projected onto the
determinants of one particle number and Sz = 0; every body rank present in
g_eff is carried, including a three-body vertex that a two-body FCI or DMRG
engine cannot take.
"""
import math

import numpy as np

from src.Base.pyscf_interface import (embed_spatial_eri_in_spin_orbitals,
                                      get_antisymmetrized_integrals)


def spatial_to_spin_orbitals(g_eff, n_spatial_orbs):
    """Spin-orbital (interleaved alpha/beta) g_eff from a spatial one.

    g_eff[2] and g_eff[3] come in as physicist tensors and leave antisymmetrized
    over the annihilation indices; the spin selection rules fall out of the
    antisymmetrization, not out of the spatial tensor.
    """
    n_spin = 2 * n_spatial_orbs
    g_spin = {}
    if 0 in g_eff:
        g_spin[0] = g_eff[0]
    if 1 in g_eff:
        h1_spatial = g_eff[1]
        h1_spin = np.zeros((n_spin, n_spin))
        h1_spin[0::2, 0::2] = h1_spatial
        h1_spin[1::2, 1::2] = h1_spatial
        g_spin[1] = h1_spin
    if 2 in g_eff:
        g_spin[2] = get_antisymmetrized_integrals(
            embed_spatial_eri_in_spin_orbitals(g_eff[2]))
    if 3 in g_eff:
        g3_spatial = g_eff[3]
        phys_spin = np.zeros((n_spin, n_spin, n_spin, n_spin, n_spin, n_spin))
        for s1 in (0, 1):
            for s2 in (0, 1):
                for s3 in (0, 1):
                    phys_spin[s1::2, s2::2, s3::2, s1::2, s2::2, s3::2] = g3_spatial
        # Antisymmetrize over the last 3 indices (annihilation operators)
        g_spin[3] = (
            phys_spin
            - phys_spin.transpose(0, 1, 2, 3, 5, 4)
            - phys_spin.transpose(0, 1, 2, 4, 3, 5)
            + phys_spin.transpose(0, 1, 2, 4, 5, 3)
            + phys_spin.transpose(0, 1, 2, 5, 3, 4)
            - phys_spin.transpose(0, 1, 2, 5, 4, 3)
        )
    return g_spin


def build_explicit_hamiltonian(g_eff, n_spin_orbitals, n_electrons):
    """
    Translates the effective Hamiltonian into an explicit matrix in the
    Slater determinant basis for a specific particle number and Sz=0.
    g_eff: dictionary mapping n (body-rank) to an array of shape (n_spin_orbitals,)*(2*n).
           g_eff[0] is the scalar 0-body energy.
    """
    # openfermion is an optional dependency; only this determinant build needs it
    from openfermion import FermionOperator, get_sparse_operator
    g0 = g_eff.get(0, 0.0)
    H_eff = FermionOperator((), g0)

    max_vertex = max(k for k in g_eff.keys() if isinstance(k, int))

    for n in range(1, max_vertex + 1):
        if n not in g_eff: continue
        gn = g_eff[n]
        prefactor = 1.0 / (math.factorial(n)**2)

        nonzero_indices = np.argwhere(np.abs(gn) > 1e-12)
        for idx in nonzero_indices:
            val = gn[tuple(idx)]
            c_str = " ".join([f"{i}^" for i in idx[:n]])
            a_str = " ".join([f"{i}" for i in idx[::-1][:n]])
            term = f"{c_str} {a_str}"
            H_eff += FermionOperator(term, prefactor * val)

    H_fock = get_sparse_operator(H_eff, n_qubits=n_spin_orbitals)

    # Restrict to Sz=0 (equal alpha/beta electrons); even spin-orbitals are alpha, odd are beta.
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha

    valid_det_indices = []
    for idx in range(H_fock.shape[0]):
        alpha_count = bin(idx & 0x55555555).count("1")
        beta_count = bin(idx & 0xaaaaaaaa).count("1")
        if alpha_count == n_alpha and beta_count == n_beta:
            valid_det_indices.append(idx)

    valid_det_indices = np.array(valid_det_indices)
    H_matrix_nelec = H_fock[valid_det_indices, :][:, valid_det_indices]

    return H_matrix_nelec, valid_det_indices
