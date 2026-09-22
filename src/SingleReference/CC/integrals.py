"""Spin-orbital antisymmetrized integrals, in the convention amplitudes.py / lambda*_residual.py / d1_blocks.py expect.

Spin orbitals interleaved (2p=alpha, 2p+1=beta), energy-sorted, so
o = slice(None, nocc), v = slice(nocc, None) is valid. The spatial->spin
expansion uses openfermion/openfermionpyscf rather than a hand-rolled one,
since that pipeline is independently validated against NWChem's CCSDT.
"""
import numpy as np
import pyscf
from pyscf import scf
from src.Base.pyscf_interface import (
    get_effective_one_electron_integrals,
    get_antisymmetrized_spin_block_eri,
    get_orbital_energies,
    get_uhf_spin_orbital_arrays_blockstacked,
    uhf_blockstacked_order,
)


def build_spinorbital_integrals_from_mf(mf):
    """Build the spin-orbital integral dict from a converged pyscf mean-field object.

    Returns dict with fock, g (antisymmetrized <pq||rs>), hf_energy,
    nuclear_repulsion, nocc/nvir (spin-orbital counts), mo_coeff, mo_energy.
    """
    mol = mf.mol

    if isinstance(mf, scf.uhf.UHF):
        eps_spin, g, nocc = get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
        nvir = g.shape[0] - nocc

        # Build blockstacked fock matrix (should be diagonal for canonical UHF,
        # but we build the full matrix to match the RHF branch structure)
        fock_ao = mf.get_fock()
        mo_a, mo_b = mf.mo_coeff[0], mf.mo_coeff[1]
        f_mo_a = mo_a.T @ fock_ao[0] @ mo_a
        f_mo_b = mo_b.T @ fock_ao[1] @ mo_b

        norb_a, norb_b = f_mo_a.shape[0], f_mo_b.shape[0]
        f_all_a_all_b = np.zeros((norb_a + norb_b, norb_a + norb_b))
        f_all_a_all_b[:norb_a, :norb_a] = f_mo_a
        f_all_a_all_b[norb_a:, norb_a:] = f_mo_b

        nocc_a, nocc_b = mf.nelec
        order = uhf_blockstacked_order(nocc_a, nocc_b, norb_a, norb_b)
        fock = f_all_a_all_b[np.ix_(order, order)]

        # For HF energy, we don't strictly need soei if we already have mf.e_tot
        # but let's provide it in blockstacked order too
        h1_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        h1_mo_a = mo_a.T @ h1_ao @ mo_a
        h1_mo_b = mo_b.T @ h1_ao @ mo_b
        h1_all_a_all_b = np.zeros_like(f_all_a_all_b)
        h1_all_a_all_b[:norb_a, :norb_a] = h1_mo_a
        h1_all_a_all_b[norb_a:, norb_a:] = h1_mo_b
        soei = h1_all_a_all_b[np.ix_(order, order)]

        hf_energy = mf.e_tot - mol.energy_nuc()
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy

    else:
        # call-time import: optional (README, Install) and slow to load
        try:
            from openfermion.chem.molecular_data import spinorb_from_spatial
            from openfermionpyscf._run_pyscf import compute_integrals
        except ImportError as exc:
            raise ImportError("the restricted path needs openfermion and "
                              "openfermionpyscf (pip install openfermion "
                              "openfermionpyscf)") from exc

        mo_occ = np.asarray(mf.mo_occ)
        if mo_occ.ndim != 1 or not np.all(np.isin(mo_occ, (0.0, 2.0))):
            raise NotImplementedError(
                "CCSDT integrals need a spin-restricted closed-shell reference "
                "or a UHF reference.")

        oei, tei = compute_integrals(mol, mf)
        nele = int(round(mo_occ.sum()))
        nocc = nele          # spin orbitals
        nvir = 2 * oei.shape[0] - nocc

        soei, stei = spinorb_from_spatial(oei, tei)
        astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
        g = astei.transpose(0, 1, 3, 2)

        o = slice(None, nocc)
        fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
        hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
        mo_coeff = mf.mo_coeff
        mo_energy = mf.mo_energy

    return {
        'fock': fock,
        'g': g,
        'soei': soei,
        'hf_energy': hf_energy,
        'nuclear_repulsion': mol.energy_nuc(),
        'nocc': nocc,
        'nvir': nvir,
        'mo_coeff': mo_coeff,
        'mo_energy': mo_energy,
    }


def build_spinorbital_integrals(geometry, basis, charge=0):
    """Build the molecule, run RHF, and delegate to build_spinorbital_integrals_from_mf.

    geometry: list of (symbol, (x, y, z)) tuples, coordinates in Angstrom.
    """
    atom_str = '; '.join(f'{sym} {x} {y} {z}' for sym, (x, y, z) in geometry)
    mol = pyscf.M(atom=atom_str, basis=basis, charge=charge)
    mf = mol.RHF()
    mf.verbose = 0
    mf.run()
    return build_spinorbital_integrals_from_mf(mf)


def build_restricted_integrals_from_mf(mf):
    """Build the spatial-MO restricted spin-block integral dict from a converged
    spin-restricted closed-shell pyscf mean-field object.

    Thin wrapper -- all the actual integral math already lives in
    src/Base/pyscf_interface.py: get_effective_one_electron_integrals (spatial
    Fock), get_antisymmetrized_spin_block_eri (spatial physicist g_aaaa/
    g_bbbb/g_abab -- g_bbbb == g_aaaa for RHF, not returned here since the
    restricted CCSDT equations never need it as a separate array, see
    the generator), get_orbital_energies (spatial mo_energy).

    Returns dict with f_aa, g_aaaa, g_abab, hf_energy, nuclear_repulsion,
    nocc/nvir (SPATIAL orbital counts, unlike build_spinorbital_integrals_from_mf's
    spin-orbital counts), o/v slices, mo_coeff, mo_energy.
    energy_denominators(f_aa, nocc, nvir) (below) works unchanged for the
    restricted case -- it only depends on the Fock diagonal and occ/vir
    slice sizes, not on whether the space is spin-orbital or spatial.
    """

    mol = mf.mol
    mo_occ = np.asarray(mf.mo_occ)
    if mo_occ.ndim != 1 or not np.all(np.isin(mo_occ, (0.0, 2.0))):
        raise NotImplementedError(
            "restricted CCSDT integrals need a spin-restricted closed-shell "
            "reference (RHF/RKS with doubly occupied orbitals only)")

    nocc = int(round(mo_occ.sum())) // 2   # spatial occupied orbitals
    nmo = mo_occ.shape[0]
    nvir = nmo - nocc

    f_aa = get_effective_one_electron_integrals(mol, mf, representation='spatial')
    g_aaaa, _, g_abab = get_antisymmetrized_spin_block_eri(mol, mf)
    mo_energy = get_orbital_energies(mf, representation='spatial')

    return {
        'f_aa': f_aa,
        'g_aaaa': g_aaaa,
        'g_abab': g_abab,
        'hf_energy': mf.e_tot - mol.energy_nuc(),
        'nuclear_repulsion': mol.energy_nuc(),
        'nocc': nocc,
        'nvir': nvir,
        'o': slice(None, nocc),
        'v': slice(nocc, None),
        'mo_coeff': mf.mo_coeff,
        'mo_energy': mo_energy,
    }


def energy_denominators(fock, nocc, nvir):
    """1/(eps_occ - eps_vir)-type denominators for the CC/Lambda-CC fixed-point updates."""
    eps = np.diagonal(fock).copy()
    n = np.newaxis
    o = slice(None, nocc)
    v = slice(nocc, None)
    e_ai = 1 / (-eps[v, n] + eps[n, o])
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_abcijk = 1 / (-eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    return e_ai, e_abij, e_abcijk
