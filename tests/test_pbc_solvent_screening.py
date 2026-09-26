"""Planar-interface screening for periodic GW-BSE: v -> v + vtilde on a slab.

src/SingleReference/Periodic/pbc_solvent_screening.py -- the periodic analogue
of Base/solvent_screening.py's reaction term, with the closed-cavity PCM
replaced by an image-charge boundary condition at a planar interface, which is
what a slab with electrolyte above (and optionally a metal electrode below)
actually is.

Checks:
  1. Kernel algebra, exact: the rank-2 factorization reproduces the closed-form
     three-layer solution; one dielectric side collapses to a single image
     charge; the thin-sheet limit gives v + vtilde = (2pi/k) 2/(eps1 + eps2);
     eps = 1 gives identically zero.
  2. A metal electrode is eps = inf, i.e. reflection coefficient exactly 1;
     two facing conductors are refused (the factorization loses all precision
     there, see the class docstring).
  3. Measure and factorization together, against an INDEPENDENT evaluation in
     the mixed (G_par, z) representation with numerical z-quadrature -- the
     one place a silent prefactor error could hide.
  4. The reaction-field metric is Hermitian and negative semi-definite, and
     eps = 1 leaves the three-center factor L bit-identical.
  5. IN-PLANE SUPERCELL FOLDING: a 2x2 in-plane supercell at Gamma must
     reproduce the primitive cell on a 2x2 k-mesh, per unit cell, for the
     SCREENED kernel as exactly as for the bare one. This is the decisive
     normalization test -- a 2D cell introduces a second
     family of normalization conventions (in-plane BZ area against cell
     volume) and this is what would catch an error there.
  6. Sigma^solv has the image-charge sign structure (occupied up, virtual
     down) and a metal substrate screens harder than water.
  7. An over-tight cavity is refused rather than silently over-screening.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.df import ft_ao

from src.SingleReference.Periodic.pbc_rpa import make_auxcell, ri_rpa_ecorr_from_dfints
from src.SingleReference.Periodic.pbc_rpa_damping import (nyquist_params,
                                                          make_coulG_damped,
                                                          check_low_dim_support)
from src.SingleReference.Periodic.pbc_damped_integrals import build_dfintegrals_coulG
from src.SingleReference.Periodic.pbc_solvent_screening import (
    SlabDielectricEnvironment, build_dfintegrals_screened, solvent_cohsex_kpts,
    reaction_kernel_mixed, reaction_field_form, _planar_grid)

from src.Base.constants import HARTREE_TO_EV


def check(ok, label, detail=''):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def make_slab(nrep, kmesh, mesh):
    """A layer of H2 units, vacuum along z. Small enough to fold in-plane."""
    cell = pgto.Cell()
    atoms = []
    for i in range(nrep):
        for j in range(nrep):
            atoms += [f'H {4.0 * i} {4.0 * j} -0.37', f'H {4.0 * i} {4.0 * j} 0.37']
    cell.atom = '; '.join(atoms)
    cell.a = np.diag([4.0 * nrep, 4.0 * nrep, 24.0])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.dimension = 2
    cell.mesh = mesh
    cell.verbose = 0
    cell.build()
    mf = pscf.KRHF(cell, cell.make_kpts(kmesh), exxdiv=None).density_fit()
    mf.kernel()
    return cell, mf


# --- fixtures -------------------------------------------------------------
#
# Runs both ways: `pytest` collects these, and `python tests/..._screening.py`
# runs them in order with a summary. Without the fixtures pytest reads `cell`,
# `mf` and `coulG` as missing and ERRORS on five of the six tests -- a silent
# hole in an unattended run rather than a failure anyone would notice.
#
# Module-scoped: `make_slab` is an SCF, built once and shared.

KMESH = [2, 2, 1]
MESH = [13, 13, 72]


@pytest.fixture(scope='module')
def _slab():
    return make_slab(1, KMESH, MESH)


@pytest.fixture(scope='module')
def cell(_slab):
    return _slab[0]


@pytest.fixture(scope='module')
def mf(_slab):
    return _slab[1]


@pytest.fixture(scope='module')
def coulG(cell):
    r0, beta, _ = nyquist_params(cell, KMESH)
    check_low_dim_support(cell, KMESH, r0, beta)
    return make_coulG_damped(r0, beta)


def test_kernel_algebra(cell):
    k = np.array([0.05, 0.3, 1.0, 3.0])
    z0, zc = 3.0, 0.6
    z = np.linspace(zc - z0, zc + z0, 7)
    env = SlabDielectricEnvironment(cell, solvent='water', z_center=zc, z_half_width=z0)

    M = env.rank2_matrix(k, 0.0)
    u = np.stack([np.exp(k[:, None] * ((z - zc)[None, :] - z0)),
                  np.exp(-k[:, None] * ((z - zc)[None, :] + z0))], axis=1)
    rank2 = np.einsum('kmn,kmz,knw->kzw', M, u, u)
    exact = reaction_kernel_mixed(env, k, z, z)
    ok = check(np.abs(rank2 - exact).max() / np.abs(exact).max() < 1e-13,
               'rank-2 factorization == closed-form three-layer kernel',
               f'{np.abs(rank2 - exact).max() / np.abs(exact).max():.1e}')

    one = SlabDielectricEnvironment(cell, eps_top=4.0, eps_bot=1.0,
                                    z_center=0.0, z_half_width=z0)
    zz = z - zc
    got = reaction_kernel_mixed(one, k, zz, zz)
    beta = (4.0 - 1) / (4.0 + 1)
    img = -(2 * np.pi / k)[:, None, None] * beta * np.exp(
        -k[:, None, None] * (2 * z0 - zz[None, :, None] - zz[None, None, :]))
    ok &= check(np.abs(got - img).max() / np.abs(img).max() < 1e-13,
                'one dielectric side == a single image charge',
                f'{np.abs(got - img).max() / np.abs(img).max():.1e}')

    for ea, eb in ((1.7764, 1.7764), (2.5, 1.0), (4.0, 9.0)):
        sheet = SlabDielectricEnvironment(cell, eps_top=ea, eps_bot=eb, z_center=0.0,
                                          z_half_width=1e-7, allow_static_eps=True)
        total = 2 * np.pi / k + reaction_kernel_mixed(sheet, k, np.zeros(1), np.zeros(1))[:, 0, 0]
        ref = (2 * np.pi / k) * 2.0 / (ea + eb)
        err = np.abs(total / ref - 1).max()
        ok &= check(err < 1e-5, f'thin-sheet limit eps=({ea}, {eb}) -> 2/(eps1+eps2)',
                    f'rel {err:.1e}')

    vac = SlabDielectricEnvironment(cell, eps=1.0, z_center=0.0, z_half_width=z0)
    ok &= check(np.abs(reaction_kernel_mixed(vac, k, z, z)).max() == 0.0,
                'eps = 1 gives an identically zero kernel')
    assert ok


def test_metal_substrate(cell):
    env = SlabDielectricEnvironment(cell, solvent='water', eps_bot=np.inf,
                                    z_center=0.0, z_half_width=3.0)
    b_top, b_bot = env.beta
    ok = check(b_bot == 1.0 and abs(b_top - (1.7764 - 1) / (1.7764 + 1)) < 1e-3,
               'a metal electrode is eps = inf, reflection coefficient 1',
               f'beta = ({b_top:.4f}, {b_bot:.4f})')
    try:
        SlabDielectricEnvironment(cell, eps_top=np.inf, eps_bot=np.inf, z_half_width=3.0)
    except ValueError:
        ok &= check(True, 'two facing conductors are refused')
    else:
        ok &= check(False, 'two facing conductors are refused')
    assert ok


def test_measure(cell):
    """reaction_field_form against a mixed-representation numerical evaluation."""
    aux = make_auxcell(cell, 'weigend')
    Gv, Gvbase, kws = cell.get_Gv_weights(cell.mesh)
    gpar, gz, wz, shape = _planar_grid(cell, cell.mesh)
    area = cell.vol / np.linalg.norm(cell.a[2])
    env = SlabDielectricEnvironment(cell, solvent='water', z_center=0.0, z_half_width=3.5)

    q = np.zeros(3)
    auxG = ft_ao.ft_ao(aux, Gv, kpt=q)
    fast = reaction_field_form(env, auxG, auxG, q, gpar, gz, wz, area, 0.0)

    n_par, n_z, naux = len(gpar), len(gz), auxG.shape[1]
    wq = wz * 2 * np.pi * area                      # the G_z quadrature weight itself
    zgrid = np.linspace(-env.z_half_width, env.z_half_width, 601)
    simp = np.ones(len(zgrid))
    simp[1:-1:2], simp[2:-1:2] = 4, 2
    simp *= (zgrid[1] - zgrid[0]) / 3
    chi3 = auxG.reshape(n_par, n_z, naux)
    kpar = np.linalg.norm(q[None, :2] + gpar[:, :2], axis=1)
    slow = np.zeros((naux, naux), dtype=complex)
    for g in range(n_par):
        if kpar[g] < 1e-8:
            continue                                # head dropped in both routes
        chi_z = np.einsum('z,zP,zg->gP', wq / (2 * np.pi), chi3[g],
                          np.exp(1j * np.outer(gz, zgrid)))
        vt = reaction_kernel_mixed(env, kpar[g:g + 1], zgrid, zgrid)[0]
        slow += (1 / area) * np.einsum('gP,g,gh,h,hQ->PQ',
                                       chi_z.conj(), simp, vt, simp, chi_z)
    scale = np.abs(fast).max()
    ok = check(np.abs(fast - slow).max() / scale < 1e-7,
               'measure + factorization == independent mixed-representation route',
               f'{np.abs(fast - slow).max() / scale:.1e}')
    ok &= check(np.abs(fast - fast.conj().T).max() / scale < 1e-12,
                'the reaction-field metric is Hermitian')
    w = np.linalg.eigvalsh(0.5 * (fast + fast.conj().T))
    ok &= check(w.max() < 1e-10 * scale,
                'the reaction-field metric is negative semi-definite',
                f'largest eigenvalue {w.max():.1e} vs |dJ|max {scale:.1f}')
    assert ok


def _cohsex_pattern(dfints):
    """The (sum_virt - sum_occ) contraction of ONE integral set, unsymmetrized
    -- the reference for how non-Hermitian the underlying G-space build already
    is at this mesh (see solvent_cohsex_kpts)."""
    nk, nmo = dfints.nkpts, dfints.nmo
    out = np.zeros((nk, nmo, nmo), dtype=complex)
    for kn in range(nk):
        for kp in range(nk):
            L1, L2 = dfints.Lblock(kn, kp), dfints.Lblock(kp, kn)
            o, v = slice(0, dfints.nocc[kp]), slice(dfints.nocc[kp], nmo)
            out[kn] += 0.5 * (np.einsum('Ppa,Paq->pq', L1[:, :, v], L2[:, v, :])
                              - np.einsum('Ppi,Piq->pq', L1[:, :, o], L2[:, o, :])) / nk
    return out


def test_vacuum_and_signs(cell, mf, coulG):
    bare = build_dfintegrals_coulG(mf, coulG_fn=coulG)
    ref = _cohsex_pattern(bare)
    bare_asym = max(np.abs(ref[k] - ref[k].conj().T).max()
                    for k in range(bare.nkpts)) / np.abs(ref).max()
    vac_env = SlabDielectricEnvironment(cell, eps=1.0, z_center=0.0, z_half_width=4.0)
    vac = build_dfintegrals_screened(mf, vac_env, coulG_fn=coulG)
    d = max(np.abs(vac.L[q] - bare.L[q]).max() for q in range(bare.nkpts))
    ok = check(d == 0.0, 'eps = 1 leaves L bit-identical to the bare build', f'{d:.1e}')

    shifts = {}
    for tag, kw in (('water', dict(solvent='water')),
                    ('water|metal', dict(solvent='water', eps_bot=np.inf))):
        env = SlabDielectricEnvironment(cell, z_center=0.0, z_half_width=4.0, **kw)
        scr = build_dfintegrals_screened(mf, env, coulG_fn=coulG)
        sigma = solvent_cohsex_kpts(bare, scr)
        herm = max(np.abs(sigma[k] - sigma[k].conj().T).max() for k in range(bare.nkpts))
        ok &= check(herm < 1e-14 * np.abs(sigma).max(),
                    f'{tag}: Sigma^solv is Hermitian', f'{herm:.1e}')
        # ... and the pre-symmetrization asymmetry must stay at the level the
        # bare G-space build already has at this mesh, not exceed it by an
        # order of magnitude -- that would mean the screening introduced its own.
        raw = _cohsex_pattern(scr)
        raw_asym = max(np.abs(raw[k] - raw[k].conj().T).max()
                       for k in range(bare.nkpts)) / np.abs(raw).max()
        ok &= check(raw_asym < 10 * bare_asym,
                    f'{tag}: screening adds no non-Hermiticity of its own',
                    f'{raw_asym:.1e} vs bare {bare_asym:.1e}')
        nocc = bare.nocc[0]
        diag = np.array([np.diag(sigma[k]).real for k in range(bare.nkpts)])
        ok &= check((diag[:, :nocc] > 0).all() and (diag[:, nocc:] < 0).all(),
                    f'{tag}: occupied up, virtual down (image-charge signs)',
                    f'occ {diag[:, :nocc].mean() * HARTREE_TO_EV:+.3f}, '
                    f'virt {diag[:, nocc:].mean() * HARTREE_TO_EV:+.3f} eV')
        shifts[tag] = abs(diag[:, :nocc].mean())
    ok &= check(shifts['water|metal'] > 1.5 * shifts['water'],
                'a metal electrode screens harder than water',
                f"{shifts['water|metal'] / shifts['water']:.2f}x")
    assert ok


def test_inplane_folding():
    """The in-plane-BZ-area vs cell-volume normalization test."""
    prim, mf_p = make_slab(1, [2, 2, 1], [13, 13, 72])
    sup, mf_s = make_slab(2, [1, 1, 1], [26, 26, 72])
    r0p, bp, rc_p = nyquist_params(prim, [2, 2, 1])
    r0s, bs, rc_s = nyquist_params(sup, [1, 1, 1])
    ok = check(abs(rc_p - rc_s) < 1e-9, 'folded pair shares one damping radius',
               f'Rc {rc_p:.4f} vs {rc_s:.4f}')
    cg_p, cg_s = make_coulG_damped(r0p, bp), make_coulG_damped(r0s, bs)
    ok &= check(abs(mf_p.e_tot - mf_s.e_tot / 4) < 1e-6,
                'mean field folds', f'{abs(mf_p.e_tot - mf_s.e_tot / 4):.1e}')

    rel = {}
    for tag, kw in (('bare', None),
                    ('screened', dict(solvent='water', eps_bot=np.inf))):
        ec = []
        for cell, mf, cg, n in ((prim, mf_p, cg_p, 1), (sup, mf_s, cg_s, 4)):
            if kw is None:
                dfints = build_dfintegrals_coulG(mf, coulG_fn=cg)
            else:
                env = SlabDielectricEnvironment(cell, z_center=0.0,
                                                z_half_width=4.0, **kw)
                dfints = build_dfintegrals_screened(mf, env, coulG_fn=cg)
            ec.append(ri_rpa_ecorr_from_dfints(dfints, nw=24) / n)
        rel[tag] = abs(ec[0] - ec[1]) / abs(ec[0])
        print(f'       {tag:9s}: Ec/cell primitive {ec[0]:.10f}  supercell {ec[1]:.10f}')
    # The screened kernel must fold as well as the bare one does; the bare
    # number is the SCF/mesh floor, not a property of the screening.
    assert ok & check(rel['screened'] < 3 * max(rel['bare'], 1e-12),
                      'the SCREENED kernel folds as exactly as the bare one',
                      f"screened {rel['screened']:.1e} vs bare {rel['bare']:.1e}")


def test_overtight_cavity_is_refused(cell, mf, coulG):
    env = SlabDielectricEnvironment(cell, solvent='water', eps_bot=np.inf,
                                    z_center=0.0, z_half_width=0.9)
    leak = env.leaked_density_fraction(mf)
    try:
        build_dfintegrals_screened(mf, env, coulG_fn=coulG)
    except ValueError as err:
        ok = check('not positive definite' in str(err),
                   'an over-tight cavity is refused, not silently over-screened',
                   f'leaked density {leak:.1%}')
    else:
        ok = check(False,
                   'an over-tight cavity is refused, not silently over-screened',
                   f'no error raised (leaked density {leak:.1%})')
    assert ok


def _run(label, fn, *args):
    """Run one check, report, and KEEP GOING -- see the note in
    tests/test_pbc_occupations.py: which checks fail TOGETHER is the diagnosis,
    so one failure must not hide the rest."""
    print(f'\n-- {label}')
    try:
        fn(*args)
        return True
    except AssertionError as err:
        print(f'   FAILED: {err}' if str(err) else '   FAILED')
        return False


if __name__ == '__main__':
    cell_, mf_ = make_slab(1, KMESH, MESH)
    r0, beta, _ = nyquist_params(cell_, KMESH)
    check_low_dim_support(cell_, KMESH, r0, beta)
    coulG_ = make_coulG_damped(r0, beta)

    all_ok = True
    all_ok &= _run('1. kernel algebra', test_kernel_algebra, cell_)
    all_ok &= _run('2. metal electrode', test_metal_substrate, cell_)
    all_ok &= _run('3. measure and factorization', test_measure, cell_)
    all_ok &= _run('4/6. vacuum invariant and Sigma^solv signs',
                   test_vacuum_and_signs, cell_, mf_, coulG_)
    all_ok &= _run('7. over-tight cavity',
                   test_overtight_cavity_is_refused, cell_, mf_, coulG_)
    all_ok &= _run('5. in-plane supercell folding', test_inplane_folding)

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
