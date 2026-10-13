"""The calibrated scissor inside the set carries the static term once.

With `scissor='calibrate'`, a state of the explicit set whose pole model
Eq. (27) refuses is solved on the quadrature at the reference geometry R0 and
takes the 'scissor' route afterwards,

    eps^QP_p = eps_p + xc_p + s_p,

with xc_p = <p|Sigma_x - v_xc|p> + Sigma^env_pp (Sigma^env the reaction field
of Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016),
Eq. (18), zero in the gas phase). The root at R0 is
w_p = eps_p + xc_p + Sigma_c,pp(w_p), so the frozen shift is
s_p = w_p - eps_p - xc_p: at R0 the state sits on its own root, and elsewhere
it moves with eps_p + xc_p.

The first test fails if s_p also carries xc_p (s_p = w_p - eps_p); the second
checks what the frozen shift lets move. Formaldehyde/cc-pVDZ, PBE0 in the gas
phase and HF in toluene.

Run: python tests/test_inside_scissor_static_term.py   (or pytest)
"""
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import dft, gto, scf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

from src.Base.constants import (HARTREE_TO_EV, QP_CD_NEWTON_TOL,  # noqa: E402
                                SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.solvent_screening import SolventScreening  # noqa: E402
from src.gradients.excited_state import ExcitedStateChain  # noqa: E402

H2CO = 'C 0 0 -0.5296; O 0 0 0.6763; H 0 0.9339 -1.1088; H 0 -0.9339 -1.1088'
BASIS = 'cc-pvdz'
#: The explicit set: every orbital up to LUMO+3, so the deep ones and the
#: highest are inside it and outside the pole model's reach.
WINDOW = tuple(range(12))
#: Exact bookkeeping: eps + xc + (w - (eps + xc)) is w to the last bits.
EXACT_HA = 1e-12
#: The displacement of the second geometry, Bohr, on the carbon along z.
STEP = 2e-2
CASES = (('pbe0', 'gas'), ('hf', 'toluene'))


def factory_for(xc):
    def factory(mol):
        mf = (scf.RHF(mol) if xc == 'hf' else dft.RKS(mol, xc=xc))
        mf = mf.density_fit(auxbasis=BASIS + '-jkfit')
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged
        return mf
    return factory


def build(mol, factory, mf, solvent, scissor):
    env = None if solvent == 'gas' else SolventScreening(mol, solvent=solvent)
    return ExcitedStateChain(mol, factory, mf=mf, qp_window=list(WINDOW),
                             residue_route='sop', scissor=scissor,
                             outside='scissor', environment=env,
                             solver='dense')


def forward(chain, mol, mf):
    """eps, eps^QP and the per-state static term of the chain's forward at
    `mol`, read off `kernel_pieces`."""
    pieces = chain.kernel_pieces(mol, mf)
    shift = pieces[14]
    return SimpleNamespace(
        eps=np.asarray(pieces[6]), eps_qp=np.asarray(pieces[7]),
        xc_correction=np.asarray(chain._xc_correction(mf, chain.qp_set,
                                                      shift)))


def displaced(mol):
    out = mol.copy()
    coords = out.atom_coords()
    coords[0, 2] += STEP
    out.set_geom_(coords, unit='Bohr')
    out.build(False, False)
    return out


@pytest.fixture(scope='module', params=CASES, ids=['-'.join(c) for c in CASES])
def case(request):
    """(chain, its forward at R0, the all-explicit chain's roots at R0)."""
    warnings.simplefilter('ignore')
    xc, solvent = request.param
    factory = factory_for(xc)
    mol = gto.M(atom=H2CO, basis=BASIS, verbose=0)
    mf = factory(mol)
    chain = build(mol, factory, mf, solvent, 'calibrate')
    chain.excitation()
    at_r0 = forward(chain, chain.mol0, chain.mf0)
    # The same set with no tier: the refused states stay on the quadrature,
    # which is the route the calibration solved them by.
    plain = build(mol, factory, mf, solvent, None)
    plain.excitation()
    explicit = forward(plain, plain.mol0, plain.mf0).eps_qp
    return chain, at_r0, explicit


def test_an_inside_scissor_state_sits_on_its_explicit_root(case):
    chain, at_r0, explicit = case
    tiered = sorted(chain.scissor_map)
    assert tiered, 'the case must put states on the calibrated scissor'
    for p in tiered:
        off = at_r0.eps_qp[p] - chain.qp_seeds[p]
        assert abs(off) < EXACT_HA, (
            f'orbital {p} sits {off * HARTREE_TO_EV:+.3f} eV off the root '
            f'it was calibrated on')
        off = at_r0.eps_qp[p] - explicit[p]
        assert abs(off) < QP_CD_NEWTON_TOL, (
            f'orbital {p} sits {off * HARTREE_TO_EV:+.3f} eV off its root '
            f'with every state solved')


def test_a_displaced_geometry_moves_it_by_eps_and_the_static_term(case):
    chain, at_r0, _ = case
    mol = displaced(chain.mol0)
    there = forward(chain, mol, chain.mean_field(mol)[1])
    states = [int(p) for p in chain.qp_set]
    for p in sorted(chain.scissor_map):
        i = states.index(p)
        static = ((there.eps[p] + there.xc_correction[i])
                  - (at_r0.eps[p] + at_r0.xc_correction[i]))
        moved = there.eps_qp[p] - at_r0.eps_qp[p]
        assert abs(static) > 1e-5, 'the step must move the orbital'
        assert abs(moved - static) < EXACT_HA, (p, moved, static)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
