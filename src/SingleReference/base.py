import numpy as np


def get_occ_virt_indices(eps, nocc):
    """Split an orbital-energy array into occ = [0, nocc), virt = [nocc, norb) index arrays."""
    norb = len(eps)
    return np.arange(nocc), np.arange(nocc, norb)


def transition_range(spectra, noccs):
    """(smallest, largest) particle-hole energy eps_a - eps_i over every spin
    channel: the range a screening built from all of them spans."""
    lo, hi = [], []
    for eps, nocc in zip(spectra, noccs):
        occ, virt = get_occ_virt_indices(eps, nocc)
        lo.append(eps[virt].min() - eps[occ].max())
        hi.append(eps[virt].max() - eps[occ].min())
    return min(lo), max(hi)
