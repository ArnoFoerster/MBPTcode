"""Static second-order GW kernel (BSE2@GW) on the DF Casida route.

`static_second_order_kernel_df` returns the Theta^GW blocks of Monino and Loos,
J. Chem. Phys. 159, 034105 (2023), eq 72, to add to the BSE@GW Casida matrices.
H2O/cc-pVDZ at QuAcK's geometry, HF start, exact three-index factors (no with_df,
so the eigh/Cholesky factorisation of the full ERI tensor). Four checks:

1. The DF contraction equals a dense transcription of the six terms on the norb^4
   W tensor built from the same factors, at the default block size and at one that
   leaves a partial last block over c.
2. With QuAcK's G0W0@HF energies on the diagonal (graphical roots, full-RPA W,
   eta = 0) the lowest singlet and triplet of BSE@GW and BSE2@GW, full and TDA,
   reproduce QuAcK master 2236bfc, whose RGW_phBSE2_static_kernel_A/B is added to
   both manifolds with the HF energies in the denominators: run
   2026-09-25_quack-xcheck-h2o on WSL, files full_bse, tda_fullw_bse, full_bse2,
   tda_fullw_bse2 (same-input match to 1e-6 eV in
   2026-09-25_h2o-bse2-vs-quack-sameqp).
3. Shifting the HF energies by -0.3 Ha puts the LUMO below zero, so a
   particle-particle denominator eps_c + eps_d changes sign inside its sum: the
   routine warns once, and stays silent on the unshifted set.
4. H2/STO-3G, nocc = nvirt = 1: the size-one block shape against the dense form,
   for both kernels.
5. kernel='eq71', the crossed class of eq 71 in place of the loop class of the
   printed eq 72, on H2O/6-31G: the DF blocks against the crossed class in the
   spin-orbital vertex form of the second-order block (the twelve-class split of
   `developments/ee_gw_fold/dbse2_derivation_2026-10-13/eq74_block_classes.py` in
   mbpt-tooling) with the static denominator, plus the dense ladders, at the default
   block size and at one with a partial last block, the B block by the letter swap
   j <-> b that takes the printed A blocks to their B blocks; the loop class in
   the vertex form against the printed particle-hole term, which ties the vertex
   normalisation to the QuAcK-checked kernel; the crossed class's triplet vanishes
   in both forms and the routine's triplet blocks are zero.
6. device='gpu' against the CPU path on the same factors, Theta^A and Theta^B to
   1e-12 relative, for both kernels and both channels at eta 0 and 1e-3, on
   H2/STO-3G, H2O/cc-pVDZ (exact factors) and cyclazine/6-31G (density fitted,
   Monino and Loos's geometry); command-line names select a subset; on water
   with a device budget that caps every block at one virtual; and on water with a
   device too small for the two pair-sized arrays, which raises ValueError
   before any upload. Without CuPy, or
   with CuPy and no visible GPU, these rows print a skip line; without CuPy
   device='gpu' raises ImportError.
"""
import os
import sys
import types
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_orbital_energies)
from src.SingleReference.LinearResponse import linear_response
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, static_second_order_kernel_df)

try:
    import cupy
except ImportError:              # optional: only the device='gpu' rows need it
    cupy = None

# QuAcK mol/H2O.xyz, Angstrom
GEOM = 'O 0.0000 0.0000 0.0000; H 0.7571 0.0000 0.5861; H -0.7571 0.0000 0.5861'
# QuAcK G0W0@RHF e_QP, eV, all 24 orbitals (full_bse.log of the run above)
QUACK_QP_EV = np.array([
    -547.096903359, -33.376695376, -18.558315280, -14.436803319, -12.158825764,
    4.708293937, 6.656989854, 20.360279319, 21.783506682, 30.406097434,
    31.317121452, 32.403102704, 38.226762239, 39.069729540, 44.458840280,
    49.498548274, 50.876928601, 65.959485567, 67.738625861, 88.744413957,
    90.503059309, 95.157864951, 103.217832913, 112.841177294])
# (S1, T1) in eV per (kernel, tda), the static rows of the QuAcK summaries
QUACK_EV = {('bse', False): (8.450045, 7.664144), ('bse', True): (8.484507, 7.698575),
            ('bse2', False): (8.736268, 7.802672), ('bse2', True): (8.759333, 8.010908)}
TOL_EV = 1e-5
# cycl[3,3,3]azine S0, B3LYP/cc-pVDZ, Monino and Loos, JCP 159, 034105 (2023), SI
# page S3, Angstrom
CYCLAZINE = """N 0.000000 0.009080 0.000000; C 1.224802 0.716230 0.000000
C -1.224802 0.716230 0.000000; C 2.437473 -0.010534 0.000000
C -2.437473 -0.010534 0.000000; H 3.363630 0.562441 0.000000
H -3.363630 0.562441 0.000000; C 1.201735 2.129808 0.000000
C -1.201735 2.129808 0.000000; H 2.161023 2.645400 0.000000
H -2.161023 2.645400 0.000000; C 0.000000 2.821547 0.000000
H 0.000000 3.914218 0.000000; C 0.000000 -1.405179 0.000000
C 2.435666 -1.397141 0.000000; C -2.435666 -1.397141 0.000000
H 3.381943 -1.943483 0.000000; H -3.381943 -1.943483 0.000000
C 1.235729 -2.091996 0.000000; C -1.235729 -2.091996 0.000000
H 1.202592 -3.180558 0.000000; H -1.202592 -3.180558 0.000000"""
# (geometry, basis, density fitting) of the device='gpu' rows
GPU_SYSTEMS = {'h2': ('H 0 0 0; H 0 0 0.74', 'sto-3g', False),
               'water': (GEOM, 'cc-pvdz', False),
               'cyclazine': (CYCLAZINE, '6-31g', True)}


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}"
          + (f'   ({detail})' if detail else ''))
    return bool(ok)


def dense_theta(eps, coeff, W_aux, nocc, ladders=True):
    """The six terms of the Notes of static_second_order_kernel_df on the dense
    W_pq,rs = sum_PQ B_P,pq W_PQ B_Q,rs, as (n_pair, n_pair) blocks in (ia, jb);
    the particle-hole terms alone (the triplet channel) with ladders=False."""
    W = np.einsum('Ppq,PQ,Qrs->pqrs', coeff, W_aux, coeff)
    o, v = slice(0, nocc), slice(nocc, None)
    e = eps
    d_kc = 1.0 / (e[v][None, :] - e[o][:, None])
    s_kl = 1.0 / (e[o][:, None] + e[o][None, :])
    s_cd = 1.0 / (e[v][:, None] + e[v][None, :])
    A = 4.0 * np.einsum('ijkc,kc,abkc->iajb', W[o, o, o, v], d_kc, W[v, v, o, v])
    B = 4.0 * np.einsum('ibkc,kc,ajkc->iajb', W[o, v, o, v], d_kc, W[v, o, o, v])
    if ladders:
        A = A + 2.0 * np.einsum('akjl,kl,kilb->iajb',
                                W[v, o, o, o], s_kl, W[o, o, o, v])
        A = A - 2.0 * np.einsum('acjd,cd,cidb->iajb',
                                W[v, v, o, v], s_cd, W[v, o, v, v])
        B = B + 2.0 * np.einsum('akbl,kl,kilj->iajb',
                                W[v, o, v, o], s_kl, W[o, o, o, o])
        B = B - 2.0 * np.einsum('acbd,cd,cidj->iajb',
                                W[v, v, v, v], s_cd, W[v, o, v, o])
    n_pair = A.shape[0] * A.shape[1]
    return A.reshape(n_pair, n_pair), B.reshape(n_pair, n_pair)


def spin_orbital_w(coeff, W_aux):
    """W_pq,rs = sum_PQ B_P,pq W_PQ B_Q,rs over spin orbitals, spins interleaved
    (spin orbital 2p, 2p + 1 = spatial p alpha, beta), zero across spin in a pair."""
    w_sp = np.einsum('Ppq,PQ,Qrs->pqrs', coeff, W_aux, coeff)
    idx = np.arange(2 * w_sp.shape[0])
    sp, sg = idx // 2, idx % 2
    same = sg[:, None] == sg[None, :]
    return w_sp[np.ix_(sp, sp, sp, sp)] * same[:, :, None, None] \
        * same[None, None, :, :]


def singlet_triplet(X):
    """S = X_aa,aa + X_aa,bb and T = X_aa,aa - X_aa,bb of a spin-orbital block
    X[i, a, j, b] with interleaved spins, as (n_pair, n_pair) blocks in (ia, jb)."""
    S = X[::2, ::2, ::2, ::2] + X[::2, ::2, 1::2, 1::2]
    T = X[::2, ::2, ::2, ::2] - X[::2, ::2, 1::2, 1::2]
    n_pair = S.shape[0] * S.shape[1]
    return S.reshape(n_pair, n_pair), T.reshape(n_pair, n_pair)


def second_order_classes_static(eps, coeff, W_aux, nocc):
    """The crossed class -(I.L + L.I) and the loop class -(I.K + K.I) of the
    second-order block A R A^T / 4 over a 2p2h state, in spin orbitals, with the
    static denominator of eq 72's recipe (drop omega and every external orbital
    energy, keep the internal pair's): R -> -1/(e_d - e_k) in the crossed class,
    whose internal pair is (k, d), and R -> -1/(e_d - e_l) in the loop class,
    internal pair (l, d). Vertices over ordered doubles (k, l, c, d) from the
    pair (i, a), chemist W: I = delta_ac W_ik,ld, K = delta_ik W_ac,ld,
    L = delta_il W_ad,kc. Returns the singlet and triplet of each class as
    (n_pair, n_pair) blocks in (ia, jb) over spatial pairs:
    {'crossed': (S, T), 'loop': (S, T)}."""
    w = spin_orbital_w(coeff, W_aux)
    n = w.shape[0]
    e = eps[np.arange(n) // 2]
    no = 2 * nocc
    nv = n - no
    o, v = slice(0, no), slice(no, n)
    # vertices (i, a; k, l, c, d)
    I = np.einsum('ikld,ac->iaklcd', w[o, o, o, v], np.eye(nv))
    K = np.einsum('acld,ik->iaklcd', w[v, v, o, v], np.eye(no))
    L = np.einsum('adkc,il->iaklcd', w[v, v, o, v], np.eye(no))
    e_o, e_v = e[o], e[v]
    R_crossed = np.broadcast_to(
        -1.0 / (e_v[None, None, None, :] - e_o[:, None, None, None]), (no, no, nv, nv))
    R_loop = np.broadcast_to(
        -1.0 / (e_v[None, None, None, :] - e_o[None, :, None, None]), (no, no, nv, nv))

    def prod(X, R, Z):
        ns = no * nv
        return ((X.reshape(ns, -1) * R.ravel()) @ Z.reshape(ns, -1).T) \
            .reshape(no, nv, no, nv)

    crossed = -(prod(I, R_crossed, L) + prod(L, R_crossed, I))
    loop = -(prod(I, R_loop, K) + prod(K, R_loop, I))
    return {'crossed': singlet_triplet(crossed), 'loop': singlet_triplet(loop)}


def crossed_class_B_static(eps, coeff, W_aux, nocc):
    """The B block of the crossed class by the letter swap j <-> b that takes the
    routine's printed A blocks to its B blocks (an assumption, see its Notes), in
    spin orbitals: sum_kc (W_ik,bc W_ka,jc + W_jk,ac W_kb,ic) / (e_c - e_k), as
    singlet and triplet (n_pair, n_pair) blocks in (ia, jb)."""
    w = spin_orbital_w(coeff, W_aux)
    n = w.shape[0]
    e = eps[np.arange(n) // 2]
    no = 2 * nocc
    o, v = slice(0, no), slice(no, n)
    d_kc = 1.0 / (e[v][None, :] - e[o][:, None])
    N = np.einsum('ikbc,kc,kajc->iajb', w[o, o, v, v], d_kc, w[o, v, o, v])
    return singlet_triplet(N + N.transpose(2, 3, 0, 1))


def eq71_reference(eps, coeff, W_aux, nocc):
    """The kernel='eq71' singlet blocks: the crossed class's singlet in the vertex
    form (A) and by the letter swap (B), each plus the dense ladders."""
    S_cA, _ = second_order_classes_static(eps, coeff, W_aux, nocc)['crossed']
    S_cB, _ = crossed_class_B_static(eps, coeff, W_aux, nocc)
    A_full, B_full = dense_theta(eps, coeff, W_aux, nocc)
    A_ph, B_ph = dense_theta(eps, coeff, W_aux, nocc, ladders=False)
    return S_cA + (A_full - A_ph), S_cB + (B_full - B_ph)


def hf_factors(atom, basis, density_fit):
    """(eps, coeff, W_aux, nocc) at RHF: orbital energies, the three-index factor
    (with_df's when density_fit, else the exact one) and the static screening."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).density_fit() if density_fit else scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = mol.nelectron // 2
    eps = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    w_aux = LinearResponseSolver(eps, coeff_df=coeff,
                                 spin_mode='restricted').static_screening_aux(nocc)
    return eps, coeff, w_aux, nocc


if __name__ == '__main__':
    mol = gto.M(atom=GEOM, basis='cc-pvdz', unit='Angstrom', verbose=0)
    nocc = mol.nelectron // 2
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    eps_hf = np.asarray(get_orbital_energies(mf, representation='spatial'), float)
    coeff = get_density_fitting_coefficients(mol, mf, representation='spatial')
    w_aux = LinearResponseSolver(eps_hf, coeff_df=coeff,
                                 spin_mode='restricted').static_screening_aux(nocc)
    all_ok = True

    # 1. DF contraction against the dense transcription, two block sizes
    A_ref, B_ref = dense_theta(eps_hf, coeff, w_aux, nocc)
    scale = max(np.abs(A_ref).max(), np.abs(B_ref).max())
    for blksize in (8, 4):
        A_df, B_df = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc,
                                                   blksize=blksize)
        dev = max(np.abs(A_df - A_ref).max(), np.abs(B_df - B_ref).max()) / scale
        all_ok &= check(dev < 1e-10, f'DF Theta^A/Theta^B vs dense, blksize={blksize}',
                        f'max rel dev {dev:.1e}')
    # 1b. the triplet channel is the particle-hole part of the dense transcription
    A_ph, B_ph = dense_theta(eps_hf, coeff, w_aux, nocc, ladders=False)
    A_t, B_t = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc,
                                             channel='triplet')
    dev = max(np.abs(A_t - A_ph).max(), np.abs(B_t - B_ph).max()) / scale
    all_ok &= check(dev < 1e-10, "channel='triplet' vs dense particle-hole terms",
                    f'max rel dev {dev:.1e}')
    try:
        static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc, channel='doublet')
        all_ok &= check(False, 'unknown channel raises ValueError')
    except ValueError:
        all_ok &= check(True, 'unknown channel raises ValueError')

    # 2. BSE@GW and BSE2@GW energies on QuAcK's QP energies vs QuAcK
    eps_qp = QUACK_QP_EV / HARTREE_TO_EV
    theta = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc)
    lr = LinearResponseSolver(eps_qp, coeff_df=coeff, spin_mode='restricted')
    for kernel in ('bse', 'bse2'):
        for tda in (False, True):
            got = []
            for triplet in (False, True):
                A, B = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_aux,
                                                triplet=triplet)
                if kernel == 'bse2':
                    A, B = A + theta[0], B + theta[1]
                got.append(float(CasidaSolver(A, B).solve(tda=tda)[0].min())
                           * HARTREE_TO_EV)
            ref = QUACK_EV[(kernel, tda)]
            dev = max(abs(got[0] - ref[0]), abs(got[1] - ref[1]))
            all_ok &= check(dev < TOL_EV,
                            f"{kernel}@G0W0 {'TDA' if tda else 'full'} S1/T1 vs QuAcK",
                            f'{got[0]:.6f}/{got[1]:.6f} eV, dev {dev * 1e6:.1f} ueV')

    # 3. Energies not measured from the chemical potential: a ladder denominator
    #    changes sign inside its sum and the routine says so; the HF set is silent.
    for shift, expect in ((0.0, 0), (-0.3, 1)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            static_second_order_kernel_df(eps_hf + shift, coeff, w_aux, nocc)
        hits = [w for w in caught if 'changes sign' in str(w.message)]
        all_ok &= check(len(hits) == expect,
                        f'sign-change warning with eps shifted by {shift} Ha',
                        f'{len(hits)} warning(s), expected {expect}')

    # 4. The degenerate block shape, nocc = nvirt = 1 (H2/STO-3G): every reshape
    #    and transpose of the routine on size-one axes, against the dense form.
    mol1 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='Angstrom',
                 verbose=0)
    mf1 = scf.RHF(mol1)
    mf1.conv_tol = 1e-10
    mf1.kernel()
    eps1 = np.asarray(get_orbital_energies(mf1, representation='spatial'),
                      float)
    coeff1 = get_density_fitting_coefficients(mol1, mf1, representation='spatial')
    w1 = LinearResponseSolver(eps1, coeff_df=coeff1,
                              spin_mode='restricted').static_screening_aux(1)
    A1, B1 = static_second_order_kernel_df(eps1, coeff1, w1, 1)
    A1_ref, B1_ref = dense_theta(eps1, coeff1, w1, 1)
    dev = max(abs(A1 - A1_ref).max(), abs(B1 - B1_ref).max())
    all_ok &= check(A1.shape == (1, 1) and dev < 1e-12 * max(abs(A1_ref).max(), 1.0),
                    'nocc = nvirt = 1 block shape vs dense', f'abs dev {dev:.1e}')
    A1, B1 = static_second_order_kernel_df(eps1, coeff1, w1, 1, kernel='eq71')
    A1_ref, B1_ref = eq71_reference(eps1, coeff1, w1, 1)
    dev = max(abs(A1 - A1_ref).max(), abs(B1 - B1_ref).max())
    all_ok &= check(A1.shape == (1, 1) and dev < 1e-12 * max(abs(A1_ref).max(), 1.0),
                    "nocc = nvirt = 1 block shape vs dense, kernel='eq71'",
                    f'abs dev {dev:.1e}')

    # 5. kernel='eq71' on H2O/6-31G: the crossed class in the spin-orbital vertex
    #    form plus the dense ladders; the loop class in the same form against the
    #    printed particle-hole term; the triplet zero.
    mol5 = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    mf5 = scf.RHF(mol5)
    mf5.conv_tol = 1e-10
    mf5.kernel()
    eps5 = np.asarray(get_orbital_energies(mf5, representation='spatial'), float)
    coeff5 = get_density_fitting_coefficients(mol5, mf5, representation='spatial')
    w5 = LinearResponseSolver(eps5, coeff_df=coeff5,
                              spin_mode='restricted').static_screening_aux(nocc)
    nv5 = len(eps5) - nocc
    classes = second_order_classes_static(eps5, coeff5, w5, nocc)
    A5_ph, _ = dense_theta(eps5, coeff5, w5, nocc, ladders=False)
    A5_ref, B5_ref = eq71_reference(eps5, coeff5, w5, nocc)
    scale5 = max(np.abs(A5_ref).max(), np.abs(B5_ref).max())
    dev = np.abs(classes['loop'][0] - A5_ph).max() / scale5
    all_ok &= check(dev < 1e-10, 'loop class (vertex form, static) vs printed '
                    'particle-hole term', f'max rel dev {dev:.1e}')
    for blksize in (8, 3):
        A5, B5 = static_second_order_kernel_df(eps5, coeff5, w5, nocc, blksize=blksize,
                                               kernel='eq71')
        dev = max(np.abs(A5 - A5_ref).max(), np.abs(B5 - B5_ref).max()) / scale5
        all_ok &= check(dev < 1e-10, f"kernel='eq71' vs crossed class + ladders, "
                        f'blksize={blksize} (nvirt={nv5})', f'max rel dev {dev:.1e}')
    dev = max(np.abs(classes['crossed'][1]).max(),
              np.abs(crossed_class_B_static(eps5, coeff5, w5, nocc)[1]).max()) / scale5
    all_ok &= check(dev < 1e-12, 'crossed class triplet vanishes, A (vertex form) '
                    'and B (letter swap)', f'max rel {dev:.1e}')
    A5_t, B5_t = static_second_order_kernel_df(eps5, coeff5, w5, nocc, kernel='eq71',
                                               channel='triplet')
    all_ok &= check(A5_t.shape == (nocc * nv5,) * 2 and not A5_t.any()
                    and not B5_t.any(), "kernel='eq71', channel='triplet' is zero")
    try:
        static_second_order_kernel_df(eps5, coeff5, w5, nocc, kernel='eq74')
        all_ok &= check(False, 'unknown kernel raises ValueError')
    except ValueError:
        all_ok &= check(True, 'unknown kernel raises ValueError')

    # 6. device='gpu' against the CPU path on the same factors; the sign-change
    #    warning concerns the physics of an input, not the comparison
    try:
        static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc, device='tpu')
        all_ok &= check(False, 'unknown device raises ValueError')
    except ValueError:
        all_ok &= check(True, 'unknown device raises ValueError')
    if cupy is None:
        try:
            static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc, device='gpu')
            all_ok &= check(False, "device='gpu' without CuPy raises ImportError")
        except ImportError:
            all_ok &= check(True, "device='gpu' without CuPy raises ImportError")
        print("  [skip] device='gpu' against the CPU path: CuPy is not importable")
    elif not cupy.is_available():
        print("  [skip] device='gpu' against the CPU path: CuPy sees no GPU")
    else:
        name = cupy.cuda.runtime.getDeviceProperties(0)['name']
        print(f"  device: {name.decode() if isinstance(name, bytes) else name}, "
              f'CuPy {cupy.__version__}')
        names = [a for a in sys.argv[1:] if not a.startswith('--')] \
            or list(GPU_SYSTEMS)
        for sname in names:
            eps_s, coeff_s, w_s, nocc_s = hf_factors(*GPU_SYSTEMS[sname])
            for kernel in ('printed', 'eq71'):
                for channel in ('singlet', 'triplet'):
                    devs, host = [], True
                    for eta in (0.0, 1e-3):
                        kw = dict(eta=eta, channel=channel, kernel=kernel)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore',
                                                    message='a ladder denominator')
                            ref = static_second_order_kernel_df(
                                eps_s, coeff_s, w_s, nocc_s, **kw)
                            got = static_second_order_kernel_df(
                                eps_s, coeff_s, w_s, nocc_s, device='gpu', **kw)
                        host &= all(type(x) is np.ndarray for x in got)
                        scale = max(np.abs(ref[0]).max(), np.abs(ref[1]).max(),
                                    1e-300)
                        devs.append(max(np.abs(got[0] - ref[0]).max(),
                                        np.abs(got[1] - ref[1]).max()) / scale)
                    all_ok &= check(host and max(devs) < 1e-12,
                                    f"{sname}: device='gpu' vs cpu, "
                                    f'kernel={kernel!r}, {channel}, host arrays',
                                    f'max rel dev {devs[0]:.1e} at eta 0, '
                                    f'{devs[1]:.1e} at eta 1e-3')
        # a device budget that caps every block at one virtual
        keep, devs = linear_response.BSE2_GPU_BLOCK_GB, []
        linear_response.BSE2_GPU_BLOCK_GB = 1e-12
        try:
            for kernel in ('printed', 'eq71'):
                ref = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc,
                                                    kernel=kernel)
                got = static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc,
                                                    kernel=kernel, device='gpu')
                scale = max(np.abs(ref[0]).max(), np.abs(ref[1]).max())
                devs.append(max(np.abs(got[0] - ref[0]).max(),
                                np.abs(got[1] - ref[1]).max()) / scale)
        finally:
            linear_response.BSE2_GPU_BLOCK_GB = keep
        all_ok &= check(max(devs) < 1e-12, "water: device='gpu' vs cpu, singlet, "
                        'BSE2_GPU_BLOCK_GB at 1e-12 (blocks of one virtual)',
                        f'max rel dev {devs[0]:.1e} printed, {devs[1]:.1e} eq71')
        # a device whose free memory, pool included, cannot hold the two
        # pair-sized arrays: refused before anything else of cp is touched
        need = 16 * (nocc * (len(eps_hf) - nocc)) ** 2 / 1024 ** 3
        real = linear_response.cp
        linear_response.cp = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                Device=lambda: types.SimpleNamespace(mem_info=(1024, 1024))),
            get_default_memory_pool=lambda: types.SimpleNamespace(
                free_bytes=lambda: 1024))
        try:
            static_second_order_kernel_df(eps_hf, coeff, w_aux, nocc, device='gpu')
            msg = 'no error'
        except ValueError as e:
            msg = str(e)
        finally:
            linear_response.cp = real
        all_ok &= check(f'{need:.3g} GiB' in msg
                        and f'{2048 / 1024 ** 3:.3g} GiB' in msg,
                        "water: device='gpu' past the free memory raises ValueError",
                        msg)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
