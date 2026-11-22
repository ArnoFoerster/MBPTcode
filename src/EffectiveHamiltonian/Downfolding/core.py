"""The frozen inactive core of the downfolded Hamiltonian.

The occupied orbitals outside the active space (ActiveSpace.env_occ) are a
frozen core: its mean field enters h_eff as the core Fock operator on the
active orbitals (`core_fock`), and its own energy enters E_0 (`core_energy`).
Both read one exact J/K build of the core (`core_jk`); `core_terms` makes
that build once and returns both, as the `Core` record. The core is the
same for every low-level method. Every quantity is per spin channel
(spin.py).
"""
from dataclasses import dataclass

import numpy as np
from pyscf.scf.hf import get_jk

from .spin import spin_weight


@dataclass(frozen=True)
class Core:
    """The frozen core's part of H_eff: fock, its Fock operator on the active
    orbitals, one (n_act, n_act) array per spin channel; e_core, e_hartree,
    e_exchange, its one-electron, Hartree and exchange energies, the
    environment terms of E_0."""
    fock: tuple
    e_core: float
    e_hartree: float
    e_exchange: float


def core_terms(mol, hcore_ao, mo, spaces):
    """The `Core` of the inactive orbitals of `spaces`, from one J/K build.
    mo = (C,) or (C_a, C_b), spaces likewise; hcore_ao is the AO core
    Hamiltonian."""
    jk = core_jk(mol, mo, spaces)
    fock = core_fock(mol, hcore_ao, mo, spaces, jk=jk)
    return Core(fock, *core_energy(mol, hcore_ao, mo, spaces, jk=jk))


def core_jk(mol, mo, spaces):
    """(dms, j, k): the inactive densities D_I^sigma = C_I C_I^T of the spin
    channels, one electron per spin orbital, as one (nchannel, nao, nao)
    array; the Coulomb matrix J[D_I] of the whole core, D_I = D_I^a + D_I^b
    (2 D_I^sigma of the one restricted channel); and the exchange matrices
    K[D_I^sigma], one per channel.

    J and K are the exact 4-centre ones of `mol`: the bare v of the effective
    Hamiltonian is never density fitted. One build serves both `core_fock`
    and `core_energy`, which take it as `jk`; with no inactive
    orbital nothing is built. mo = (C,) or (C_a, C_b), spaces likewise.
    """
    weight = spin_weight(mo)
    dms = np.array([c[:, p.env_occ] @ c[:, p.env_occ].T
                    for c, p in zip(mo, spaces)])
    if not any(len(p.env_occ) for p in spaces):
        return dms, np.zeros_like(dms[0]), tuple(np.zeros_like(d) for d in dms)
    vj, vk = get_jk(mol, dms)
    return dms, weight * sum(vj), tuple(vk)


def core_fock(mol, hcore_ao, mo, spaces, jk=None):
    """One (n_act, n_act) array per spin channel: the Fock operator of the
    inactive core on the active orbitals.

    f^sigma = h + J[D_I] - K[D_I^sigma] with D_I^sigma = C_I C_I^T the
    inactive density of one spin (one electron per spin orbital) and D_I the
    whole core's, so a restricted reference gives
    h_tu + sum_{i in I} [2(tu|ii) - (ti|iu)]. J and K are the exact ones of
    `core_jk`, built here unless handed in as `jk`. mo = (C,) or
    (C_a, C_b), spaces likewise; hcore_ao is the AO core Hamiltonian.
    """
    _, j, vk = core_jk(mol, mo, spaces) if jk is None else jk
    return tuple(c[:, p.act].T @ (hcore_ao + j - k) @ c[:, p.act]
                 for c, p, k in zip(mo, spaces, vk))


def core_energy(mol, hcore_ao, mo, spaces, jk=None):
    """(e_core, e_hartree, e_exchange): the one-electron, Hartree and exchange
    energies of the inactive core.

    e_core     = sum_s Tr[h D_I^s]
    e_hartree  = 1/2 sum_s Tr[J[D_I] D_I^s]
    e_exchange = -1/2 sum_s Tr[K[D_I^s] D_I^s]
    with D_I the whole core's density, the sums over both spins (twice the
    one restricted channel), on the exact J and K of `core_jk`, built
    here unless handed in as `jk`. Restricted: e_core = 2 sum_i h_ii,
    e_hartree = 2 sum_ij (ii|jj), e_exchange = -sum_ij (ij|ji). With every
    occupied orbital inactive, E_nuc plus the three is the HF energy.
    """
    weight = spin_weight(mo)
    dms, j, vk = core_jk(mol, mo, spaces) if jk is None else jk
    e_core = weight * sum(np.einsum('ij,ji->', hcore_ao, d) for d in dms)
    e_hartree = 0.5 * weight * sum(np.einsum('ij,ji->', j, d) for d in dms)
    e_exchange = -0.5 * weight * sum(np.einsum('ij,ji->', k, d)
                                     for d, k in zip(dms, vk))
    return float(e_core), float(e_hartree), float(e_exchange)
