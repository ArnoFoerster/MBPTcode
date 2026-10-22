"""One build per forward quantity: every route reaches the same builder.

Each gate makes a second build impossible to hide: the shared builder is
replaced by a recording (or perturbed) wrapper and the route under test must
see the change. A route that re-spelled the quantity keeps its own numbers and
fails here.
"""
import inspect

import numpy as np
from pyscf import ao2mo, gto, scf

import src.Base.utils.time_frequency as time_frequency
import src.SingleReference.GW.imaginary_time as imaginary_time
import src.SingleReference.GW.reaction_field as reaction_field
import src.SingleReference.GW.space_time as gw_space_time
import src.SingleReference.LinearResponse.space_time as lr_space_time
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver)
from src.SingleReference.LinearResponse.quasi_boson_bse import BSEqb
from tests.pins import assert_same

NORB, NOCC = 9, 4
#: Ha. Two associations of one (A + B)^-1 contraction on water/STO-3G; an
#: eta-broadened chi0 built from the four-index direct term sits 3.0e-8 away.
FOUR_INDEX_ROUNDOFF = 1e-12
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


def spectrum(seed=3):
    """A small gapped orbital spectrum and its midgap mu."""
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.normal(size=NORB))
    eps[NOCC:] += 1.5
    return eps, 0.5 * (eps[NOCC - 1] + eps[NOCC])


# ------------------------------------------------- fit ranges and grid size
def test_the_molecular_fit_ranges_are_the_padded_window(monkeypatch):
    """`time_frequency.sigma_fit_ranges` reads the one padded window:
    a perturbed padding in `self_energy_fit_ranges_from_window` moves it."""
    eps, mu = spectrum()
    before = time_frequency.sigma_fit_ranges(eps, NOCC, mu=mu)
    one = time_frequency.self_energy_fit_ranges_from_window

    def padded(eps, mu, w_lo, w_hi, pad=time_frequency.SELF_ENERGY_PAD):
        return one(eps, mu, w_lo, w_hi, pad=(0.31, 3.0))

    monkeypatch.setattr(time_frequency, 'self_energy_fit_ranges_from_window',
                        padded)
    after = time_frequency.sigma_fit_ranges(eps, NOCC, mu=mu)
    w_lo = eps[NOCC] - eps[NOCC - 1]
    assert_same(before[0][0], 0.3 * w_lo, label='default pad')
    assert_same(after[0][0], 0.31 * w_lo, label='perturbed pad')


def test_the_gw_grid_size_is_the_ranges_rule(monkeypatch):
    """`minimax_points_for_gw` is `minimax_points_for_ranges` on its three
    ratios: the resolver it calls is the shared one."""
    eps, mu = spectrum()
    seen = []
    one = time_frequency.minimax_points_for_ranges

    def recording(ratios, **kw):
        seen.append(tuple(ratios))
        return one(ratios, **kw)

    monkeypatch.setattr(imaginary_time, 'minimax_points_for_ranges', recording)
    got = imaginary_time.minimax_points_for_gw(eps, NOCC, mu=mu)
    assert len(seen) == 1 and len(seen[0]) == 3
    assert got == one(seen[0])


# ------------------------------------------------------- the chi0(i tau) sweep
def test_chi0_has_one_sweep():
    """chi0(i.omega) is only the tiled projected sweep: no untiled M x M
    Pi(i.tau) builder and no switch that reaches one."""
    assert not hasattr(lr_space_time, 'polarizability_imaginary_time')
    params = inspect.signature(lr_space_time.chi0_imaginary_frequency).parameters
    assert 'stream' not in params


# ------------------------------------------------------ the Dyson step for W
DYSON_ROUTES = (gw_space_time._dyson_in_place, gw_space_time._dyson_owned,
                imaginary_time.screened_interaction_tau_blocked,
                imaginary_time.screened_interaction_tau,
                lr_space_time.screening_space_time)


def test_every_dyson_route_calls_the_one_kernel():
    """No imaginary-axis screening route inverts I - chi0 itself; each calls
    `dyson_screening`, the one place the bare-gauge transform is applied."""
    for module in (imaginary_time, gw_space_time):
        assert module.dyson_screening is lr_space_time.dyson_screening
    for route in DYSON_ROUTES:
        source = inspect.getsource(route)
        assert 'np.linalg.inv' not in source, route.__name__
        assert 'dyson_screening(' in source, route.__name__


def test_the_dyson_kernel_in_the_bare_gauge():
    """W - I = [I - T^T chi0 T]^-1 - I, the passenger form without T."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(6, 6))
    chi0 = -(a @ a.T) / 6
    T = np.eye(6) + 0.1 * rng.normal(size=(6, 6))
    W = lr_space_time.dyson_screening(chi0, T, correlation=True)
    ref = np.linalg.inv(np.eye(6) - T.T @ chi0 @ T) - np.eye(6)
    assert_same(W, ref, unit='1', quantity='root', label='bare gauge')
    assert_same(lr_space_time.dyson_screening(chi0),
                np.linalg.inv(np.eye(6) - chi0), unit='1', quantity='root',
                label='without T')


# ------------------------------------------------- the Eq. (18) reaction field
def test_every_shift_is_the_congruent_build(monkeypatch):
    """Both Eq. (18) arrays, the cached DF one and the separable one, are
    `congruent_quasiparticle_shift`; no contraction of their own and no
    second copy of Delta W in the module."""
    for name in ('screened_interaction_difference', 'quasiparticle_shift'):
        assert not hasattr(reaction_field, name)
    source = inspect.getsource(reaction_field.environment_quasiparticle_shift)
    assert 'einsum' not in source
    assert 'congruent_quasiparticle_shift(' in source
    rng = np.random.default_rng(11)
    x, d = rng.normal(size=(15, NORB)), rng.normal(size=(15, 5))
    a = rng.normal(size=(5, 5))
    w = np.linalg.inv(np.eye(5) + a @ a.T / 5)
    t = np.eye(5) + 0.1 * rng.normal(size=(5, 5))
    seen = []
    one = reaction_field.congruent_quasiparticle_shift

    def recording(*args):
        seen.append(args)
        return one(*args)

    monkeypatch.setattr(reaction_field, 'congruent_quasiparticle_shift',
                        recording)
    got = reaction_field.separable_quasiparticle_shift(x, d, w, t, NOCC)
    assert len(seen) == 1
    assert_same(got, one(*seen[0]), quantity='root', label='separable shift')


def test_the_bare_partner_is_the_dyson_kernel():
    """[1 - T^T chi~ T]^-1 with chi~ = 1 - W_dressed^-1, through `dyson_screening`."""
    rng = np.random.default_rng(13)
    a = rng.normal(size=(5, 5))
    w = np.linalg.inv(np.eye(5) + a @ a.T / 5)
    t = np.eye(5) + 0.1 * rng.normal(size=(5, 5))
    chi = np.eye(5) - np.linalg.inv(w)
    assert_same(reaction_field.bare_screening(w, t),
                np.linalg.inv(np.eye(5) - t.T @ chi @ t), unit='1',
                quantity='root', label='bare partner')
    assert 'dyson_screening(' in inspect.getsource(reaction_field.bare_screening)


# ----------------------------------------------- the four-index static W(0)
def test_the_four_index_screenings_are_one_response():
    """The full-ERI BSE direct term, `construct_4d_w_rpa` and the quasi-boson
    kernel screen with one eta-free (A + B)^-1: their (ij|W|ab) agree."""
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    nocc, nmo = mol.nelectron // 2, mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    eps = np.asarray(mf.mo_energy, float)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), nmo)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    W4 = lr.construct_4d_w_rpa(nocc)[0]
    direct = W4[:nocc, :nocc, nocc:, nocc:].transpose(0, 2, 1, 3).reshape(
        nocc * nvir, -1)
    w_pairs = lr.static_screening_aux(nocc)
    A, _ = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w_pairs)
    d = (eps[nocc:][None, :] - eps[:nocc][:, None]).ravel()
    v = eri[:nocc, nocc:, :nocc, nocc:].reshape(nocc * nvir, -1)
    assert np.abs((np.diag(d) + 2.0 * v - A) - direct).max() \
        <= FOUR_INDEX_ROUNDOFF
    qb = BSEqb(mf, eri, nocc)
    d_qp = (qb.eps_qp[nocc:][None, :] - qb.eps_qp[:nocc][:, None]).ravel()
    assert np.abs((np.diag(d_qp) + qb.kappa * v - qb.A_bse) - direct).max() \
        <= FOUR_INDEX_ROUNDOFF
