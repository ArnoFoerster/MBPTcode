"""Eq 71 of Monino and Loos 2023 on the signed fold (dBSE2@GW doubles, level 'dbse2').

Checks, water (RHF, DF factors), basis per check. The GW energies are a deterministic
stand-in (gw_like: occupied +0.03 Ha, virtual -0.05 Ha on the HF energies), so a swapped
energy set is visible; the W lines are screened at the HF energies throughout.

  1. build_pieces_dbse2 (6-31G): M per channel equals the Casida A of
     LinearResponseSolver at the GW energies, lBSE=False for singles='eq62' and
     lBSE=True with the static W_aux for 'eq62+eq59', to 1e-10 Ha; D is
     e_c + e_d - e_k - e_l, shape (no, no, nv, nv); V(y) has the eight signed blocks
     '+aaaa' ... '-baba' of that shape; Vt is the eta-transpose of V (V dense by unit
     singles, Vt on random doubles, eta = +1 on the '+' blocks, -1 on the '-' blocks) to
     1e-12; dnorm2 is |Y+|^2 - |Y-|^2, positive on a '+'-only Y and negative on a
     '-'-only Y; the HF energies pass as eps_gw, a closed GW gap raises ValueError, a
     flip parity raises NotImplementedError; matvec and solve_folded leave u, D and B
     untouched.
  2. fold against unfolded: the dense signed supermatrix [[M, V+^T, -V-^T], [V+, D, 0],
     [V-, 0, D]] from the pieces by unit vectors, eig not eigh; STO-3G the whole matrix,
     6-31G per channel on the eigenspaces of the alpha<->beta flip; both kernels; every
     root of solver 'newton' and 'davidson' (singlet, triplet, and spin=None on STO-3G)
     is a real eigenvalue to 3e-10 Ha (1e-8 eV) and its T1 the eigenvector's
     |y|^2 / (|y|^2 + |Z+|^2 - |Z-|^2) to 1e-8; every real eigenvalue below min D is a
     root (completeness); the two solvers agree to 1e-8 Ha and 1e-8 on T1, their matvec
     counts printed side by side; the count of complex eigenvalues is printed.
  3. eq 74 as printed (kernel 'printed', I.J + K.L - I.K - K.I) and Yamada's kernel
     ('eq71', I.J + K.L - I.L - L.I) transcribed on spin orbitals at omega = 0.3 Ha
     against A_eff(omega) - M from the pieces, singlet and triplet, 6-31G, to 1e-11 Ha;
     eq 71's triplet is zero to 1e-12, the printed triplet carries the loop terms.
  4. the static limit of the pieces' terms (static_limit_pieces) against
     static_second_order_kernel_df(..., kernel=, channel=) at the GW energies, both
     kernels, both channels, 6-31G, to 1e-10 Ha.
  5. boundaries: 'dbse2' is a fold level; the dressed W lines reproduce
     sum_PQ B_P,pq W_PQ B_Q,rs to 1e-12; coupling_pairs('eq71', 'triplet') is empty and
     'printed' keeps its pair; an unknown kernel, channel, sets or singles raises
     ValueError; a B of the wrong shape raises ValueError; pieces at level 'dbse2'
     without dnorm2 raise ValueError in folded_operator.

Run: python tests/test_ee_dbse2_fold.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.pyscf_interface import DFIntegrals, get_orbital_energies
from src.SingleReference.ADC.eeADC import ee_dbse2_pieces, ee_fold
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_screened_coulomb_aux)

WATER = 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587'


def check(ok, label, detail=''):
    """Print one verdict line and return `ok` as a bool."""
    tail = f'   ({detail})' if detail else ''
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + tail)
    return bool(ok)


def water(basis='6-31g'):
    """(eps, B, no): RHF water with DF factors B[Q, p, q] in the MO basis."""
    mol = gto.M(atom=WATER, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    B = DFIntegrals.from_scf(mol, mf).B_aa
    return eps, B, mol.nelectron // 2


def gw_like(eps, no):
    """A deterministic stand-in for the G0W0 energies: the HF gap closed by 0.08 Ha."""
    e = np.array(eps, float)
    e[:no] += 0.03
    e[no:] -= 0.05
    return e


def check_boundaries(eps, B, no):
    """The level name, the dressed W lines, the pair lists and the refusals."""
    ok = True
    ok &= check('dbse2' in ee_fold.FOLD_LEVELS, "'dbse2' is a fold level")
    L, Bt = ee_dbse2_pieces.dressed_factors(eps, B, no)
    W_aux = static_screened_coulomb_aux(eps, B, no)
    rng = np.random.default_rng(11)
    n = len(eps)
    d = 0.0
    for p, q, r, s in rng.integers(0, n, size=(4, 4)):
        # w(pq|rs) = sum_P Bt_P,pq Bt_P,rs against sum_PQ B_P,pq W_PQ B_Q,rs
        d = max(d, abs(float(Bt[:, p, q] @ Bt[:, r, s])
                       - float(B[:, p, q] @ W_aux @ B[:, r, s])))
    ok &= check(d < 1e-12 and np.allclose(L @ L.T, W_aux, atol=1e-12),
                'the dressed W lines reproduce the static W_aux', f'{d:.1e}')
    ok &= check(ee_dbse2_pieces.coupling_pairs('eq71', 'triplet') == [],
                "coupling_pairs('eq71', 'triplet') is empty")
    pr = ee_dbse2_pieces.coupling_pairs('printed', 'triplet')
    ok &= check(len(pr) == 1 and len(pr[0]) == 2 and pr[0][0].shape == (4,),
                "coupling_pairs('printed', 'triplet') keeps its pair")
    for kw, word in ((dict(kernel='eq74', channel='singlet'), 'kernel'),
                     (dict(kernel='eq71', channel='doublet'), 'channel')):
        try:
            ee_dbse2_pieces.coupling_pairs(**kw)
            ok &= check(False, f'an unknown {word} raises ValueError')
        except ValueError as exc:
            ok &= check(word in str(exc), f'an unknown {word} raises ValueError',
                        str(exc)[:60])
    try:
        ee_dbse2_pieces.dressed_factors(eps, B[:, :, :3], no)
        ok &= check(False, 'a B of the wrong shape raises ValueError')
    except ValueError as exc:
        ok &= check('shape' in str(exc), 'a B of the wrong shape raises ValueError',
                    str(exc)[:60])
    try:
        ee_fold.folded_operator({'level': 'dbse2'}, 0.3)
        ok &= check(False, "dbse2 pieces without dnorm2 raise ValueError")
    except ValueError as exc:
        ok &= check('dnorm2' in str(exc),
                    'dbse2 pieces without dnorm2 raise ValueError', str(exc)[:60])
    return ok


def main():
    """Run every check; print ALL PASSED or FAILURES DETECTED; exit 0 or 1."""
    all_ok = True
    eps, B, no = water('6-31g')
    all_ok &= check_boundaries(eps, B, no)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
