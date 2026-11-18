"""The seam between the downfolding's general layer and its low-level methods.

A low-level method (methods/hf.py, methods/rpa.py) returns a
`DownfoldingTerms`; the general layer turns it into the effective
Hamiltonian and never looks at how it was made.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DownfoldingTerms:
    """What a low-level method hands the general layer.

    vtilde: three arrays (aa, ab, bb), each (n_act,)*4, the effective
        two-body interaction in chemist order (tu|vw) on the active
        orbitals, occupied first. The opposite-spin block is oriented
        ab[t,u,v,w] = (t_alpha u_alpha | v_beta w_beta).
    sigma_bar: None, or (a, b), each (n_act, n_act), the static one-body
        self-energy of the alpha and beta spin channels.
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


def choice(value, choices, keyword):
    """`value` normalized as a keyword choice (lower case, '_' -> '-',
    'imag-frequency' -> 'imagfrequency'), refused unless one of `choices`."""
    name = str(value).lower().replace('_', '-')
    name = {'imag-frequency': 'imagfrequency'}.get(name, name)
    if name not in choices:
        raise ValueError(f"Unknown {keyword}={value!r}; choose one of "
                         + ', '.join(repr(c) for c in choices) + '.')
    return name
