"""HF as the low-level method: no screening and no correlation, so the
downfolded Hamiltonian is the CASCI Hamiltonian of the reference orbitals
(of the UHF/UKS determinant in its own alpha and beta orbitals for an
unrestricted reference, pyscf's UCASCI).

The bare v is the whole interaction (vtilde = 0), the double counting is the
full one of a zero vtilde (so nothing), and both correlation energies are
zero, which leaves E_0 = E_nuc + E_HF,core + E_HF,ee: the CASCI core energy.
With a KS reference this is still the HF expression on the KS orbitals, as in
pyscf's CASCI.
"""
from dataclasses import dataclass, field

import numpy as np

from ..masks import channels_of
from ..terms import DownfoldingTerms

METHOD = 'hf'


@dataclass
class HFLowLevel:
    """The HF low-level result: the mean field and nothing else.
    numerical_flags is filled by `run_low_level`."""
    mf: object
    method: str = METHOD
    numerical_flags: list = field(default_factory=list)


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
    `part`, one block for the ActiveSpace of a restricted reference and the
    three (aa, ab, bb) for the pair of an unrestricted one; no static
    self-energy, both correlation energies zero, and the full double counting
    (of a zero vtilde). HF takes no options."""
    _no_options(opts)
    parts = channels_of(part)
    n_act = parts[0].n_act
    blocks = 1 if len(parts) == 1 else 3
    return DownfoldingTerms(
        vtilde=tuple(np.zeros((n_act,) * 4) for _ in range(blocks)),
        sigma_bar=None, e_c_full=0.0, e_c_active=0.0, dc='full')
