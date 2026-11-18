"""HF as the low-level method: no screening and no correlation, so the
downfolded Hamiltonian is the CASCI Hamiltonian of the reference orbitals.

The bare v is the whole interaction (vtilde = 0), the double counting is the
full one of a zero vtilde (so nothing), and both correlation energies are
zero, which leaves E_0 = E_nuc + E_HF,core + E_HF,ee: the CASCI core energy.
With a KS reference this is still the HF expression on the KS orbitals, as in
pyscf's CASCI.
"""
from dataclasses import dataclass

import numpy as np

from ..terms import DownfoldingTerms

METHOD = 'hf'


@dataclass
class HFLowLevel:
    """The HF low-level result: the mean field and nothing else."""
    mf: object
    method: str = METHOD


def _no_options(opts):
    if opts:
        raise TypeError(f"method={METHOD!r} takes no options; got "
                        + ', '.join(sorted(opts)))


def run(mf, **opts):
    """The HFLowLevel of `mf`; there is nothing to compute before the
    active space is known. HF takes no options."""
    _no_options(opts)
    return HFLowLevel(mf=mf)


def terms(low, part, **opts):
    """The DownfoldingTerms of HF: vtilde = 0 on the n_act active orbitals of
    `part`, no static self-energy, both correlation energies zero, and the
    full double counting (of a zero vtilde). HF takes no options."""
    _no_options(opts)
    shape = (part.n_act,) * 4
    return DownfoldingTerms(
        vtilde=tuple(np.zeros(shape) for _ in range(3)), sigma_bar=None,
        e_c_full=0.0, e_c_active=0.0, dc='full')
