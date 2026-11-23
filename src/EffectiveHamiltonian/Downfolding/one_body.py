"""The one-body term of the downfolded Hamiltonian, per spin channel.

h_eff^sigma = f^sigma - t^DC,sigma (+ Sigma_bar^sigma): the Fock operator of
the frozen inactive core on the active orbitals (core.py), minus the double
counting of the effective interaction vtilde with the active reference
density (double_counting.py), plus the method's static self-energy when it
hands one back. Every quantity is per spin channel (spin.py).
"""
from src.Base.utils.choices import require_choice

from .double_counting import DC_MODES


def effective_one_body(f, hartree, exchange, dc, sigma_bar=None):
    """The one-body part of H_eff, one array per spin channel.

    dc='full' subtracts the whole double counting, f - hartree + exchange;
    dc='hartree' subtracts the Hartree part only, f - hartree, the convention
    that goes with the static self-energy Sigma_bar (which then carries the
    exchange). sigma_bar, one array per channel when given, is added.
    """
    dc = require_choice(dc, DC_MODES, 'dc')
    h = []
    for s in range(len(f)):
        h_s = f[s] - hartree[s]
        if dc == 'full':
            h_s = h_s + exchange[s]
        if sigma_bar is not None:
            h_s = h_s + sigma_bar[s]
        h.append(h_s)
    return tuple(h)
