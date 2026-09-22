"""Gates for the spin-free (spatial-orbital) and density-fitted EE-ADC routes.

The chain of arbiters:

  1. sb_einsum reproduces a spin-orbital contraction exactly -- the
     contraction layer itself, before any ADC content.
  2. The spin-free sigma equals the (already validated) spin-orbital sigma
     on random vectors at every level. Because both call the SAME equations
     in ee_equations, this is really a test of the spin blocking; because the
     spin-orbital side is pinned to pyscf and to the lambda oracle, it
     inherits all of that.
  3. The operator commutes with the alpha<->beta involution, and the two
     eigenspaces reproduce pyscf's singlet EE-ADC and the spin-orbital
     triplets.
  4. The DF route equals the dense-integral spin-free route to machine
     precision when B is exact, term by term for the three vvvv kernels and
     then for the whole operator. That is what checks the Q-factorized spin
     rules in ee_r_sigma_df, which are the one place a hand derivation
     survives.
"""
import numpy as np
import pytest

from pyscf import gto, scf, adc, tdscf

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, DFIntegrals)
from src.SingleReference.ADC.eeADC import (ee_utils, ee_equations as pe,
                                     ee_spin_blocks as psb,
                                     ee_u_sigma_full as pus,
                                     ee_r_sigma as prs, ee_r_sigma_df as pdf)

ATOMS = ['H 0 0 0; F 0 0 0.917', 'Li 0 0 0; H 0 0 1.6',
         'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
         'Be 0 0 0; H 0 0 1.34; H 0 0 -1.34']
LEVELS = ['adc1', 'adc2', 'adc2x', 'adc3']


def _system(atom, basis='sto-3g'):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run()
    eps_sp = get_orbital_energies(mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    return mol, mf, eps_sp, eri


def _spin_orbital(eps_sp, eri):
    return np.repeat(eps_sp, 2), get_antisymmetrized_spin_eri(eri)


# ------------------------------------------------- 1. contraction layer
@pytest.mark.parametrize('atom', ATOMS[:2])
def test_sb_einsum_matches_spin_orbital(atom):
    mol, _, eps_sp, eri = _system(atom)
    eps, g = _spin_orbital(eps_sp, eri)
    nocc, norb = mol.nelectron, len(eps)
    no, nv = nocc // 2, (norb - nocc) // 2
    gb = ee_utils.g_blocks(g, nocc, norb)
    gb_sb = psb.g_blocks_sb(eri.transpose(0, 2, 1, 3), no, no + nv)
    for key, fam in [('oooo', 'oooo'), ('ooov', 'ooov'), ('oovv', 'oovv'),
                     ('ovov', 'ovov'), ('ovvv', 'ovvv'), ('vvvv', 'vvvv')]:
        got = psb.to_spin_orbital(gb_sb[key], fam, nocc, norb)
        assert np.abs(got - gb[key]).max() < 1e-12
    t1 = ee_utils.mp_amplitudes(eps, g, nocc, norb, 1)['t2_1']
    t1_sb = psb.from_spin_orbital(t1, 'oovv', nocc, norb)
    ref = np.einsum('ikac,jkbc->iajb', gb['oovv'], t1, optimize=True)
    got = psb.to_spin_orbital(
        psb.sb_einsum('ikac,jkbc->iajb', gb_sb['oovv'], t1_sb), 'ovov',
        nocc, norb)
    assert np.abs(ref - got).max() < 1e-12


# --------------------------------------- 2. spin-free == spin-orbital sigma
@pytest.mark.parametrize('atom', ATOMS[:3])
@pytest.mark.parametrize('level', LEVELS)
def test_spinfree_sigma_matches_spin_orbital(atom, level):
    mol, _, eps_sp, eri = _system(atom)
    eps, g = _spin_orbital(eps_sp, eri)
    nocc, norb = mol.nelectron, len(eps)
    no, nv = nocc // 2, (norb - nocc) // 2
    V = eri.transpose(0, 2, 1, 3)
    aop_r, _, dr = prs.build_operator(eps_sp, V, no, level=level)
    aop_u, _, _ = pus.build_operator(eps, g, nocc, level=level)

    v = np.random.default_rng(3).normal(size=dr['nH'])
    y1, Y = prs.to_blocks(v, no, nv, level)
    pieces = [psb.to_spin_orbital(y1, 'ov', nocc, norb).ravel()]
    if level != 'adc1':
        pieces.append(prs.SCALE * ee_utils.fold_doubles(
            psb.to_spin_orbital(Y, 'oovv', nocc, norb), nocc, norb))
    w_u = aop_u(np.concatenate(pieces))
    n_s_u = nocc * (norb - nocc)
    w1u = psb.from_spin_orbital(w_u[:n_s_u].reshape(nocc, norb - nocc), 'ov',
                                nocc, norb)
    Wu = psb.SB() if level == 'adc1' else psb.from_spin_orbital(
        ee_utils.unfold_doubles(w_u[n_s_u:], nocc, norb) / prs.SCALE,
        'oovv', nocc, norb)
    ref = prs.from_blocks(w1u, Wu, no, nv, level)
    assert np.abs(aop_r(v) - ref).max() < 1e-11


# --------------------------------------------- 3. structure and spectrum
def _dense(aop, n):
    return np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level', LEVELS)
def test_symmetric_with_exact_diagonal(atom, level):
    mol, _, eps_sp, eri = _system(atom)
    no = mol.nelectron // 2
    aop, diag, d = prs.build_operator(eps_sp, eri.transpose(0, 2, 1, 3), no,
                                      level=level)
    H = _dense(aop, d['nH'])
    assert np.abs(H - H.T).max() < 1e-11
    assert np.abs(np.diag(H) - diag).max() < 1e-11


def _channels(aop, n, no, nv, level):
    H = _dense(aop, n)
    F = np.column_stack([prs.spin_flip_vector(np.eye(n)[:, k], no, nv, level)
                         for k in range(n)])
    assert np.abs(F @ F - np.eye(n)).max() < 1e-11
    assert np.abs(F @ H - H @ F).max() < 1e-10
    out = {}
    for name, sgn in (('singlet', 1), ('triplet', -1)):
        w, U = np.linalg.eigh(0.5 * (np.eye(n) + sgn * F))
        B = U[:, w > 0.5]
        out[name] = np.sort(np.linalg.eigvalsh(B.T @ H @ B))
    return out


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level,method', [('adc2', 'adc(2)'),
                                          ('adc2x', 'adc(2)-x'),
                                          ('adc3', 'adc(3)')])
def test_singlet_channel_matches_pyscf(atom, level, method):
    mol, mf, eps_sp, eri = _system(atom)
    no = mol.nelectron // 2
    nv = len(eps_sp) - no
    aop, _, d = prs.build_operator(eps_sp, eri.transpose(0, 2, 1, 3), no,
                                   level=level)
    e = _channels(aop, d['nH'], no, nv, level)
    a = adc.ADC(mf); a.method = method; a.method_type = 'ee'; a.verbose = 0
    for r in np.array(a.kernel(nroots=3)[0]):
        assert np.abs(e['singlet'] - r).min() < 1e-6


@pytest.mark.parametrize('atom', ATOMS)
def test_adc1_channels_are_cis(atom):
    mol, mf, eps_sp, eri = _system(atom)
    no = mol.nelectron // 2
    nv = len(eps_sp) - no
    aop, _, d = prs.build_operator(eps_sp, eri.transpose(0, 2, 1, 3), no,
                                   level='adc1')
    e = _channels(aop, d['nH'], no, nv, 'adc1')
    for name, singlet in (('singlet', True), ('triplet', False)):
        td = tdscf.TDA(mf); td.nstates = 3; td.singlet = singlet; td.kernel()
        for r in td.e:
            assert np.abs(e[name] - r).min() < 1e-9


# ------------------------------------------------------------- 4. DF route
def _exact_B(eri, norb):
    M = eri.reshape(norb * norb, norb * norb)
    w, U = np.linalg.eigh(0.5 * (M + M.T))
    keep = np.abs(w) > 1e-12
    return (U[:, keep] * np.sqrt(w[keep])).T.reshape(-1, norb, norb)


@pytest.mark.parametrize('atom', ATOMS[:3])
def test_df_vvvv_kernels(atom):
    """Each Q-factorized kernel against the same contraction on an explicit
    <ab||cd> block -- this is what checks the DF spin rules in isolation."""
    mol, _, eps_sp, eri = _system(atom)
    norb = len(eps_sp)
    no, nv = mol.nelectron // 2, norb - mol.nelectron // 2
    V = eri.transpose(0, 2, 1, 3)
    B = _exact_B(eri, norb)
    gb = psb.g_blocks_sb(V, no, norb)
    ref = pe.VvvvKernels(pe.SPIN_BLOCKED, gb['vvvv'])
    got = pdf.DFVvvvKernels(B, no, norb)
    eo, ev = eps_sp[:no], eps_sp[no:]
    d_ijab = (ev[None, None, :, None] + ev[None, None, None, :]
              - eo[:, None, None, None] - eo[None, :, None, None])
    amps = pe.amplitudes(pe.SPIN_BLOCKED, gb, d_ijab,
                         2.0 * (eo[:, None] - ev[None, :]), 3)
    zint = pe.z_intermediates(pe.SPIN_BLOCKED, gb, amps)
    rho = pe.density2(pe.SPIN_BLOCKED, amps)
    for name, arg in (('ladder', amps['t2_1']), ('vv_density', rho['vv']),
                      ('z10', zint['Z10'])):
        a, b = getattr(ref, name)(arg), getattr(got, name)(arg)
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            xa, xb = a.get(k), b.get(k)
            xa = np.zeros_like(xb) if xa is None else xa
            xb = np.zeros_like(xa) if xb is None else xb
            assert np.abs(xa - xb).max() < 1e-10, (name, k)


@pytest.mark.parametrize('atom', ATOMS)
@pytest.mark.parametrize('level', LEVELS)
def test_df_matches_dense_integrals(atom, level):
    mol, _, eps_sp, eri = _system(atom)
    norb = len(eps_sp)
    no = mol.nelectron // 2
    B = _exact_B(eri, norb)
    a_r, d_r, d = prs.build_operator(eps_sp, eri.transpose(0, 2, 1, 3), no,
                                     level=level)
    a_d, d_d, _ = pdf.build_operator(eps_sp, B, no, level=level)
    v = np.random.default_rng(7).normal(size=d['nH'])
    assert np.abs(a_r(v) - a_d(v)).max() < 1e-10
    assert np.abs(d_r - d_d).max() < 1e-10


@pytest.mark.parametrize('atom', ATOMS[:2])
def test_df_with_real_auxbasis_is_close(atom):
    """A genuine RI factor must land within DF error, not exactly."""
    mol, mf, eps_sp, eri = _system(atom)
    no = mol.nelectron // 2
    nv = len(eps_sp) - no
    B = DFIntegrals.from_scf(
        mol, scf.RHF(mol).density_fit(auxbasis='weigend').run()).B_aa
    a_r, _, d = prs.build_operator(eps_sp, eri.transpose(0, 2, 1, 3), no,
                                   level='adc3')
    a_d, _, _ = pdf.build_operator(eps_sp, B, no, level='adc3')
    e_r = _channels(a_r, d['nH'], no, nv, 'adc3')['singlet']
    e_d = _channels(a_d, d['nH'], no, nv, 'adc3')['singlet']
    assert np.abs(e_r - e_d).max() < 5e-2
