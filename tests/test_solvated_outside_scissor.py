"""The frozen scissor outside the quasiparticle set carries the reaction field once.

In a continuum the quasiparticle equation of an explicit orbital q is

    w_q = eps_q + <q|Sigma_x - v_xc|q> + Sigma_c,qq(w_q) + Sigma^env_qq,

Sigma^env the static reaction-field term of Duchemin, Jacquemin and Blase,
J. Chem. Phys. 144, 164106 (2016), Eq. (18). An orbital p outside the set has
no equation: `outside='scissor'` lends it the GW correction of its probe q and
`_env_static_outside` adds its own Sigma^env_pp, so

    eps^QP_p = eps_p + (w_q - eps_q - Sigma^env_qq) + Sigma^env_pp.

Both tests fail if the scissor lends w_q - eps_q, which puts the probe's
reaction field on top of the orbital's own (about 2 eV for water/cc-pVDZ,
RHF, PCM(toluene)). The gas-phase scissor is checked by
tests/test_outside_scissor.py, and the force in a continuum with the scissor
outside against a central difference by tests/test_isdf_pcm_force.py.

Run: python tests/test_solvated_outside_scissor.py   (or pytest)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import HARTREE_TO_EV  # noqa: E402
from src.Base.solvent_screening import SolventScreening  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Exact bookkeeping: the scissor is a sum of the same floats.
EXACT_HA = 1e-12
#: How far a scissored orbital's solvation shift may sit from its own
#: Sigma^env: the probe's GW correction changes with its root, by tenths of an
#: eV at most, while a doubled reaction field is off by the whole 1.8-2.2 eV.
ONCE_EV = 0.3


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 200
    mf.kernel()
    assert mf.converged
    return mf


def chain(mol, mf, environment=None):
    """The frontier set, the frozen scissor outside it, SOP residues."""
    return ExcitedStateChain(mol, scf_factory, mf=mf, qp_window=2,
                             outside='scissor', residue_route='sop',
                             environment=environment)


def forward(ch, mol, mf):
    """eps, eps^QP and the Eq. (18) term of the chain's forward at `mol`,
    read off `kernel_pieces`."""
    pieces = ch.kernel_pieces(mol, mf)
    shift = pieces[14]
    return SimpleNamespace(
        eps=np.asarray(pieces[6]), eps_qp=np.asarray(pieces[7]),
        shift=None if shift is None else np.asarray(shift))


@pytest.fixture(scope='module')
def water():
    """(chain, its forward) in the gas phase and in toluene."""
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    mf = scf_factory(mol)
    out = {}
    for name, env in (('gas', None),
                      ('toluene', SolventScreening(mol, solvent='toluene'))):
        ch = chain(mol, mf, env)
        ch.excitation(mol, mf)
        out[name] = (ch, forward(ch, mol, mf))
    return mol, out


def outside_of(ch, norb):
    inside = {int(p) for p in ch.qp_set}
    return [p for p in range(norb) if p not in inside]


def test_the_scissor_lends_the_probe_gw_correction_without_its_reaction_field(
        water):
    """eps^QP_p - eps_p - Sigma^env_pp is the nearest probe's
    w_q - eps_q - Sigma^env_qq, on every orbital outside the set."""
    _, out = water
    ch, sh = out['toluene']
    eps, env, eps_qp = sh.eps, sh.shift, sh.eps_qp
    probes = [int(q) for q in ch.qp_set]
    outside = outside_of(ch, len(eps))
    assert outside, 'the window must leave orbitals outside'
    worst = 0.0
    for p in outside:
        q = min(probes, key=lambda r: abs(eps[r] - eps[p]))
        lent = eps_qp[q] - eps[q] - env[q]
        worst = max(worst, abs(eps_qp[p] - eps[p] - env[p] - lent))
    assert worst < EXACT_HA, (
        f'a scissored orbital is {worst * HARTREE_TO_EV:.3f} eV off the '
        f"probe's GW correction plus its own reaction field")


def test_a_scissored_orbital_feels_the_reaction_field_once(water):
    """Solvation moves a scissored orbital by its own Sigma^env_pp, up to the
    change of the probe's GW correction -- not by twice the reaction field."""
    _, out = water
    ch, sol = out['toluene']
    _, gas = out['gas']
    env = sol.shift
    outside = outside_of(ch, len(sol.eps))
    moved = (sol.eps_qp - gas.eps_qp - env)[outside] * HARTREE_TO_EV
    assert np.abs(env[outside]).min() * HARTREE_TO_EV > 2 * ONCE_EV, (
        'the reaction field must dwarf the tolerance for the gate to mean '
        'anything')
    assert np.abs(moved).max() < ONCE_EV, (
        f'scissored orbitals moved {moved.min():+.3f} to {moved.max():+.3f} '
        f'eV beyond their own reaction field')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
