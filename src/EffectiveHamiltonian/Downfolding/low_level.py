"""What a low-level method hands the downfolding's general layer: the seam
between the two.

A low-level method (methods/hf.py, methods/rpa.py) returns a
`DownfoldingTerms`; the general layer turns it into the effective
Hamiltonian and never looks at how it was made.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DownfoldingTerms:
    """What a low-level method hands the general layer.

    vtilde: the correction to the two-body interaction in chemist order
        (tu|vw) on the active orbitals, each block (n_act,)*4: (v,) from a
        restricted method, (aa, ab, bb) from an unrestricted one (spin.py).
    sigma_bar: None, or the static one-body self-energy, each block
        (n_act, n_act), one per spin channel: (s,) or (a, b).
    e_c_full: the correlation energy of the full space.
    e_c_active: the correlation energy of the active space alone.
    dc: the double-counting convention, 'full' or 'hartree'.
    diagnostics: free-form record of the method's own checks and settings.
    """
    vtilde: tuple
    sigma_bar: Optional[tuple]
    e_c_full: float
    e_c_active: float
    dc: str
    diagnostics: dict = field(default_factory=dict)
