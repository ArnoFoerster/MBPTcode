"""The occupied/virtual split of an orbital spectrum and what follows from it:
the index sets, the particle-hole energies and their range."""
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


def ov_energies(eps, nocc):
    """d_(i,a) = eps_a - eps_i on the flattened pair index of C_ov."""
    occ, virt = get_occ_virt_indices(eps, nocc)
    return (eps[virt][None, :] - eps[occ][:, None]).ravel()


def ov_factor(coeff, occ, virt):
    """C_ov: the occupied-virtual block of a DF factor coeff[Q, p, q],
    (naux, nocc * nvirt) on the pair index of ov_energies."""
    return coeff[:, occ[:, None], virt].reshape(coeff.shape[0], -1)
