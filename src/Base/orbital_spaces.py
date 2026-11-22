"""The occupied/virtual split of an orbital spectrum and what follows from it:
the index sets, the particle-hole energies and their range; the occupied
counts of a mean field, checked (`closed_shell_nocc`, `unrestricted_nocc`),
and the frozen-core count of a molecule (`frozen_core`)."""
import numpy as np


def get_occ_virt_indices(eps, nocc):
    """Split an orbital-energy array into occ = [0, nocc), virt = [nocc, norb) index arrays."""
    norb = len(eps)
    return np.arange(nocc), np.arange(nocc, norb)


def particle_hole_window(eps, nocc):
    """(gap, span, mu) of one spectrum: the smallest and the largest
    particle-hole energy eps_a - eps_i, and the mid-gap chemical potential
    (eps_HOMO + eps_LUMO) / 2, HOMO and LUMO the highest occupied and the
    lowest virtual level whatever their order in `eps`."""
    eps = np.asarray(eps, float)
    occ, virt = get_occ_virt_indices(eps, nocc)
    homo, lumo = eps[occ].max(), eps[virt].min()
    return lumo - homo, eps[virt].max() - eps[occ].min(), 0.5 * (homo + lumo)


def transition_range(spectra, noccs):
    """(smallest, largest) particle-hole energy eps_a - eps_i over every spin
    channel: the range a screening built from all of them spans."""
    windows = [particle_hole_window(eps, nocc)
               for eps, nocc in zip(spectra, noccs)]
    return min(w[0] for w in windows), max(w[1] for w in windows)


def response_range(eps, nocc):
    """(smallest, largest) particle-hole energy of a reference: of the one
    spectrum of a restricted one (eps an array, nocc its occupied count), or
    over both spin channels of an unrestricted one (eps and nocc pairs,
    `transition_range`). The range a frequency or time grid of the
    reference's response has to cover."""
    if np.ndim(nocc) == 0:
        return particle_hole_window(eps, nocc)[:2]
    return transition_range(eps, nocc)


def ov_energies(eps, nocc):
    """d_(i,a) = eps_a - eps_i on the flattened pair index of C_ov."""
    occ, virt = get_occ_virt_indices(eps, nocc)
    return (eps[virt][None, :] - eps[occ][:, None]).ravel()


def ov_factor(coeff, occ, virt):
    """C_ov: the occupied-virtual block of a DF factor coeff[Q, p, q],
    (naux, nocc * nvirt) on the pair index of ov_energies."""
    return coeff[:, occ[:, None], virt].reshape(coeff.shape[0], -1)


def occupied_count(mf):
    """The closed-shell occupied count of the mean field `mf`."""
    return int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))


def frozen_core(mol):
    """Doubly occupied core orbitals: 1s past He, 1s2s2p past Ne, 1s..3d past Ar
    (the last only for Ga onwards; K-Zn keep their 3s3p frozen too). Not
    pyscf's `chemcore`, which freezes the shell of the previous noble gas only
    from the third element of a period on (Li, Be none; Na, Mg the 1s; K-Zn
    1s..2p; Ga-Kr 1s..3p)."""
    n = 0
    for a in range(mol.natm):
        z = mol.atom_charge(a)
        n += (z > 2) + 4 * (z > 10) + 4 * (z > 18) + 5 * (z > 30)
    return int(n)


def closed_shell_nocc(mo_occ):
    """Doubly occupied orbital count; raises unless the occupations are the aufbau [2 ... 2 0 ... 0].

    Every view here indexes the occupied block as [0, nocc), so a hole in the
    occupation list would label a virtual orbital as core."""
    mo_occ = np.asarray(mo_occ)
    if mo_occ.ndim != 1:
        raise ValueError('a restricted closed-shell reference has one '
                         'occupation vector; mo_occ is not one')
    nocc = int(np.count_nonzero(mo_occ > 0))
    if np.any(mo_occ[nocc:] > 0):
        raise ValueError('the occupied orbitals are not the lowest ones; a '
                         'non-aufbau reference has no [0, nocc) occupied block')
    return nocc


def unrestricted_nocc(mo_occ):
    """(N_alpha, N_beta), the occupied counts of an unrestricted determinant
    with the occupations mo_occ = (occ_alpha, occ_beta); raises, naming the
    spin, unless each spin's occupations are 1 on its lowest orbitals and 0
    on the rest.

    Fractional occupations (a smearing, pyscf's `frac_occ`) are not one
    determinant: counting each as an occupied orbital would describe a
    system with another number of electrons. A hole below an occupied
    orbital would label a virtual orbital as occupied (`closed_shell_nocc`).
    """
    counts = []
    for spin, occ in zip(('alpha', 'beta'), mo_occ):
        occ = np.asarray(occ, float)
        off = np.minimum(np.abs(occ), np.abs(occ - 1.0)) > 1e-8
        if off.any():
            raise ValueError(
                f'the {spin} orbitals of the mean field: occupations that '
                f'are not 0 or 1 ({occ[off][:4]} on orbitals '
                f'{np.flatnonzero(off)[:4]}); a fractionally occupied '
                f'reference is not one determinant.')
        try:
            counts.append(closed_shell_nocc(occ))
        except ValueError as err:
            raise ValueError(
                f'the {spin} orbitals of the mean field: {err}') from None
    return tuple(counts)
