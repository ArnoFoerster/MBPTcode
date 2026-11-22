"""The one-body term of the downfolded Hamiltonian: core Fock, split double counting.

`core_fock`, `double_counting` and `effective_one_body` are written per spin
channel. The restricted tests pin the spatial formulas on water;
the per-spin test gives the two channels different orbitals and different
active sets and compares with the spin-orbital oracle
(tests/downfolding_oracle.py), the only place the per-spin paths see two
genuinely different channels. Measured agreement: at most 1.2e-14 against both
(tolerance 1e-10).
"""
import numpy as np
import pytest
from src.EffectiveHamiltonian.Downfolding.one_body import (
    core_fock, double_counting, effective_one_body, inactive_jk)
from src.EffectiveHamiltonian.Downfolding.spin import spin_weight
from src.EffectiveHamiltonian.Downfolding.masks import ph_element_mask
from src.EffectiveHamiltonian.active_space import ActiveSpace
from tests.downfolding_oracle import (
    mo_integrals, spin_orbital_integrals, spin_orbital_one_body, water)

TOL = 1e-10


def _water_part(mf, occ_active, virt_active):
    nocc = mf.mol.nelectron // 2
    space = ActiveSpace.from_indices(len(mf.mo_energy), nocc, occ_active,
                                     virt_active)
    return space


def _random_vtilde(rng, n):
    """(aa, ab, bb) with the symmetries of a real chemist interaction: aa and
    bb 8-fold, ab only within each pair (t<->u, v<->w), so the alpha-beta
    orientation is visible."""
    def pair_sym(x):
        x = x + x.transpose(1, 0, 2, 3)
        return x + x.transpose(0, 1, 3, 2)
    aa, ab, bb = (pair_sym(rng.standard_normal((n,) * 4)) for _ in range(3))
    aa = aa + aa.transpose(2, 3, 0, 1)
    bb = bb + bb.transpose(2, 3, 0, 1)
    return aa, ab, bb


def _orthogonal(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def test_core_fock_restricted_matches_the_spatial_formula():
    mf = water()
    part = _water_part(mf, [3, 4], [5, 6])
    f_a, f_b = core_fock(mf.mol, mf.get_hcore(), (mf.mo_coeff, mf.mo_coeff),
                         (part, part))
    h, eri = mo_integrals(mf)
    a, i = part.act, part.env_occ
    expected = (h[np.ix_(a, a)]
                + 2 * np.einsum('tujj->tu', eri[np.ix_(a, a, i, i)])
                - np.einsum('tjju->tu', eri[np.ix_(a, i, i, a)]))
    assert np.abs(f_a - expected).max() < TOL
    assert np.abs(f_b - expected).max() < TOL


def test_double_counting_restricted_matches_the_spatial_formula():
    # t^DC_tu = sum_{k in A_o} [2 vt_tu,kk - vt_tk,ku]; with Sigma^E only the
    # Hartree part, sum_{k in A_o} 2 vt_tu,kk.
    rng = np.random.default_rng(0)
    n, n_occ = 4, 2
    vt = _random_vtilde(rng, n)[0]
    f = rng.standard_normal((n, n))
    f = (f + f.T, f + f.T)
    hartree, exchange = double_counting((vt, vt, vt), (n_occ, n_occ))
    k = slice(0, n_occ)
    t_h = 2 * np.einsum('tukk->tu', vt[:, :, k, k])
    t_x = np.einsum('tkku->tu', vt[:, k, k, :])
    for sigma in (0, 1):
        assert np.abs(hartree[sigma] - t_h).max() < TOL
        assert np.abs(exchange[sigma] - t_x).max() < TOL
    for dc, t_dc in (('full', t_h - t_x), ('hartree', t_h)):
        for h_eff in effective_one_body(f, hartree, exchange, dc):
            assert np.abs(h_eff - (f[0] - t_dc)).max() < TOL


def test_per_spin_channels_differ():
    """Alpha and beta with different orbitals (beta rotated within the
    occupied and within the virtual orbitals) and different active sets of
    the same size, against the spin-orbital oracle."""
    mf = water()
    rng = np.random.default_rng(0)
    nmo, nocc = mf.mo_coeff.shape[1], mf.mol.nelectron // 2
    rot = np.zeros((nmo, nmo))
    rot[:nocc, :nocc] = _orthogonal(rng, nocc)
    rot[nocc:, nocc:] = _orthogonal(rng, nmo - nocc)
    c_a = mf.mo_coeff
    c_b = c_a @ rot
    part_a = _water_part(mf, [3, 4], [5, 6])
    part_b = _water_part(mf, [2, 4], [6, 8])
    n, n_occ = part_a.n_act, part_a.n_act_occ
    aa, ab, bb = _random_vtilde(rng, n)
    sigma_bar = tuple(rng.standard_normal((n, n)) for _ in range(2))

    f = core_fock(mf.mol, mf.get_hcore(), (c_a, c_b), (part_a, part_b))
    hartree, exchange = double_counting((aa, ab, bb), (n_occ, n_occ))

    h_so, eri_so = spin_orbital_integrals(mf.mol, c_a, c_b)
    occ_env = list(part_a.env_occ) + list(nmo + part_b.env_occ)
    act = list(part_a.act) + list(nmo + part_b.act)
    occ_act = list(range(n_occ)) + list(n + np.arange(n_occ))
    vt_so = np.zeros((2 * n,) * 4)
    al, be = slice(0, n), slice(n, 2 * n)
    vt_so[al, al, al, al] = aa
    vt_so[be, be, be, be] = bb
    vt_so[al, al, be, be] = ab                       # (t_a u_a | v_b w_b)
    vt_so[be, be, al, al] = ab.transpose(2, 3, 0, 1)

    for dc in ('full', 'hartree'):
        f_so, t_so = spin_orbital_one_body(h_so, eri_so, occ_env, act, vt_so,
                                           occ_act, dc)
        # nothing couples the two spins in a one-body term
        assert np.abs(f_so[al, be]).max() < TOL
        assert np.abs(t_so[al, be]).max() < TOL
        h_eff = effective_one_body(f, hartree, exchange, dc)
        h_sig = effective_one_body(f, hartree, exchange, dc, sigma_bar)
        for s, b in enumerate((al, be)):
            assert np.abs(f[s] - f_so[b, b]).max() < TOL
            assert np.abs(h_eff[s] - (f_so[b, b] - t_so[b, b])).max() < TOL
            assert np.abs(h_sig[s] - h_eff[s] - sigma_bar[s]).max() < TOL
    # the channels really are different
    assert np.abs(f[0] - f[1]).max() > 1e-2
    assert np.abs(hartree[0] - hartree[1]).max() > 1e-2


def test_ph_only_vtilde_has_no_hartree_dc():
    eps = np.array([-1.0, -0.6, -0.4, 0.3, 0.7, 1.1])
    part = ActiveSpace.from_indices(6, 3, [1, 2], [3, 5])
    mask = ph_element_mask(part)
    rng = np.random.default_rng(0)
    vtilde = tuple(v * mask for v in _random_vtilde(rng, part.n_act))
    n_occ = part.n_act_occ
    hartree, exchange = double_counting(vtilde, (n_occ, n_occ))
    virt = np.arange(part.n_act) >= n_occ
    for s in (0, 1):
        assert np.all(hartree[s] == 0.0)
        assert np.all(exchange[s][~np.outer(virt, virt)] == 0.0)
        assert np.abs(exchange[s][np.outer(virt, virt)]).min() > 0


def test_one_restricted_channel_is_two_identical_channels():
    """A restricted reference goes in as ONE spin channel, computed once: its
    core Fock operator, double counting and one-body term are those of each
    of two identical channels (measured 0.0 to 4e-15, tolerance 1e-12), and a
    J/K build handed in as `jk` is used instead of a second one."""
    mf = water()
    part = _water_part(mf, [2, 4], [5, 6])
    c, hcore = mf.mo_coeff, mf.get_hcore()
    n, k = part.n_act, part.n_act_occ
    rng = np.random.default_rng(4)
    vt = _random_vtilde(rng, n)[0]
    sig = rng.standard_normal((n, n))

    f_one = core_fock(mf.mol, hcore, (c,), (part,))
    f_two = core_fock(mf.mol, hcore, (c, c), (part, part))
    h_one, x_one = double_counting((vt,), (k,))
    h_two, x_two = double_counting((vt, vt, vt), (k, k))
    assert len(f_one) == len(h_one) == len(x_one) == 1
    for one, two in ((f_one, f_two), (h_one, h_two), (x_one, x_two)):
        assert np.abs(one[0] - two[0]).max() < 1e-12
        assert np.abs(one[0] - two[1]).max() < 1e-12
    for dc in ('full', 'hartree'):
        one = effective_one_body(f_one, h_one, x_one, dc, (sig,))
        two = effective_one_body(f_two, h_two, x_two, dc, (sig, sig))
        assert len(one) == 1 and np.abs(one[0] - two[0]).max() < 1e-12

    jk = inactive_jk(mf.mol, (c,), (part,))
    assert jk[0].shape[0] == 1 and len(jk[2]) == 1
    marked = (jk[0], jk[1] + 1.0, jk[2])          # a J no fresh build would give
    shifted = core_fock(mf.mol, hcore, (c,), (part,), jk=marked)[0]
    s_act = c[:, part.act].T @ np.ones_like(hcore) @ c[:, part.act]
    assert np.abs(shifted - f_one[0] - s_act).max() < 1e-12


def test_three_spin_channels_are_refused():
    with pytest.raises(ValueError, match='spin channels'):
        spin_weight((0, 1, 2))
    assert (spin_weight((0,)), spin_weight((0, 1))) == (2, 1)
