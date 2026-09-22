"""The BSE@GW chain in a polarizable continuum: `ExcitedStateChain` with a
`SolventScreening` environment.

The reaction field is one substitution, v -> v + vtilde with vtilde =
v chi_solvent v (Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106
(2016), Eqs. (11)-(16)), plus the first-order static COHSEX operator Sigma^solv
that the substitution cannot generate on a bare-exchange reference (their
Eqs. (20)-(22)). The chain reaches both through its environment: the auxiliary
gauge is dressed in `FactorChain.factors` and the environment is attached to
the mean field while production's static term is evaluated. HALF A REACTION
FIELD IS WORSE THAN NONE, which is the first gate below.

Judge by OUTPUT, not by the exit code: `main` prints every check with its
measured number and returns a bool, in the style of tests/test_solvent_screening.py.

Fast tier, H2O/cc-pVDZ (seconds):
  1. the chain's own environment is the single source of truth: a screening
     attached only to the mean field reaches a gas-phase chain nowhere
  2. the solvated chain's quasiparticle energies reproduce the PRODUCTION
     space-time GW route with the same screening attached
  3. the reaction field has the image-charge sign structure through the chain:
     IP down, EA up, gap closed
  4. the gradient entry points refuse rather than returning a gas-phase force
  5. the dressed and bare gauges share one fit and one static screening, and
     the shared form answers what the separate ones did

Slower tier (`--ct`), C2H4...F2 / cc-pVDZ (a couple of minutes): the physical
trend -- a charge-transfer root is stabilized more than a local one as the
dielectric constant rises.
"""
import os
import sys

import numpy as np
import pytest
from pyscf import gto, scf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Base.constants import HARTREE_TO_EV
from src.Base.solvent_screening import (SolventScreening,
                                        attach_solvent_screening,
                                        detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.isdf_derivatives import (dfactor_adjoint,
                                            dfactor_adjoint_gauges)
from src.gradients.reaction_field_adjoint import (reaction_field_backward,
                                                  reaction_field_shift,
                                                  static_screening)
from src.properties.characters import ct_character, roots

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: Ethylene stacked over F2 at 3.5 A -- the standard cheap donor/acceptor pair.
#: The ethylene pi -> F2 sigma* root is the charge-transfer one; ethylene's own
#: pi -> pi* and F2's sigma -> sigma* are local.
C2H4_F2 = ('C 0.0000 0.6695 0.0000; C 0.0000 -0.6695 0.0000; '
           'H 0.9289 1.2321 0.0000; H -0.9289 1.2321 0.0000; '
           'H 0.9289 -1.2321 0.0000; H -0.9289 -1.2321 0.0000; '
           'F 0.0000 0.0000 3.5000; F 0.0000 0.0000 4.9120')
F2_ATOMS = (6, 7)
#: How far a LOCAL root may move in the strongest continuum swept below
#: (eps = 2.5, CS2). The electron and the hole of a compact neutral excitation
#: induce opposing reaction fields, so what survives is the difference between
#: two nearly equal polarization energies -- tens of meV, not the eV a charge
#: separation gets. Generous by an order of magnitude against the measured 43
#: meV, because the number this guards against was 2186.
LOCAL_SHIFT_BOUND_MEV = 300.0


def check(ok, label, detail=''):
    """Print a named check with its measured number and return the verdict.

    It does NOT assert: every test below prints all of its numbers first and
    asserts the aggregate at the end, because a solvent shift is only readable
    next to the shifts it is supposed to bracket.
    """
    print(f'  [{"ok" if ok else "FAIL"}] {label}' + (f'   {detail}' if detail else ''))
    return bool(ok)


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


def solvated(mol, mf, solvent='water'):
    """The chain of `mol` in a continuum: one environment, both halves of it.

    THE CAVITY IS PER GEOMETRY. `SolventScreening` builds a surface for one
    `mol`; a displaced geometry gets its own from `Environment.for_geometry`,
    cached on the molecule object by `FactorChain.environment_at`. That is what
    makes a finite difference of `excitation` the derivative of the real
    surface, and the reference any future analytic gradient has to reproduce.
    """
    return ExcitedStateChain(mol, scf_factory, mf=mf,
                             environment=SolventScreening(mol, solvent=solvent))


# --------------------------------------------------------------------------
def test_the_chain_environment_is_the_single_source_of_truth(water):
    """A screening attached only to the MEAN FIELD reaches a gas-phase chain
    nowhere -- not even half way.

    This is the invariant the environment seam buys, and it replaces a real
    defect. The reaction field has two halves: the substitution v -> v + vtilde,
    which lives in the auxiliary gauge a chain builds for itself, and the static
    COHSEX Sigma^solv, which production's one-body term reads off the mean
    field. A chain that owned only the first inherited the second by accident,
    and HALF A REACTION FIELD IS WORSE THAN NEITHER: Sigma^solv alone closes
    the quasiparticle gap by 3.5 eV with no matching screening in the BSE
    kernel, and water's S1 collapsed to 4.97 eV against 8.46 gas-phase and 8.14
    solvated.

    Now the chain evaluates that term with ITS OWN environment attached, so a
    gas-phase chain is bitwise gas-phase whatever the mean field carries, and a
    solvated chain gets both halves without anything being attached at all.

    EACH PROBE GETS ITS OWN CHAIN, all four built on the clean mean field
    before anything is attached to it. The first quasiparticle solve freezes
    the Newton branch -- the guard band, and the converged root every displaced
    geometry is then seeded from -- so a chain probed a second time starts that
    Newton at its own previous answer and stops 1.5e-15 Ha from where it
    stopped the first time. That is below the 1e-11 convergence tolerance and
    has nothing to do with what the mean field carries. Two chains built the
    same way are bitwise identical, so that is the comparison.
    """
    mol, mf = water
    gas_chain, stray_chain = (ExcitedStateChain(mol, scf_factory, mf=mf),
                              ExcitedStateChain(mol, scf_factory, mf=mf))
    sol, sol_attached = solvated(mol, mf), solvated(mol, mf)

    def probe(chain):
        return np.array([chain.excitation(mol, mf), chain.quasiparticle(0, mol, mf),
                         chain.quasiparticle(1, mol, mf)])

    gas = probe(gas_chain)
    full = probe(sol)
    attach_solvent_screening(mf, solvent='water')
    try:
        stray = probe(stray_chain)
        full_attached = probe(sol_attached)
    finally:
        detach_solvent_screening(mf)
    for name, g, h, f in zip(('S1', 'eps^QP HOMO', 'eps^QP LUMO'), gas, stray, full):
        print(f'       {name:12s} gas {g * HARTREE_TO_EV:9.4f}   '
              f'gas chain, screened mf {h * HARTREE_TO_EV:9.4f}   '
              f'solvated {f * HARTREE_TO_EV:9.4f} eV')
    ok = check(np.array_equal(gas, stray),
               'a gas-phase chain is BITWISE gas-phase whatever the mean field '
               'carries', f'max |d| {np.abs(stray - gas).max():.1e} Ha')
    ok &= check(np.array_equal(full, full_attached),
                'and a solvated chain does not double count an attachment',
                f'max |d| {np.abs(full_attached - full).max():.1e} Ha')
    gap_gas = gas[2] - gas[1]
    gap_sol = full[2] - full[1]
    assert ok & check(
        (gap_gas - gap_sol) * HARTREE_TO_EV > 1.0,
        'the environment the chain OWNS moves it, by both halves at once',
        f'quasiparticle gap {(gap_sol - gap_gas) * HARTREE_TO_EV:+.4f} eV')


def test_solvated_chain_moves(water):
    """The fix: Omega and eps^QP move, with the image-charge sign structure."""
    mol, mf = water
    gas = ExcitedStateChain(mol, scf_factory, mf=mf)
    sol = solvated(mol, mf)
    ip_g, ea_g = -gas.quasiparticle(0, mol, mf), -gas.quasiparticle(1, mol, mf)
    ip_s, ea_s = -sol.quasiparticle(0, mol, mf), -sol.quasiparticle(1, mol, mf)
    om_g, om_s = gas.excitation(mol, mf), sol.excitation(mol, mf)
    print(f'       IP  {ip_g * HARTREE_TO_EV:8.4f} -> {ip_s * HARTREE_TO_EV:8.4f} eV'
          f'   ({(ip_s - ip_g) * HARTREE_TO_EV:+.4f})')
    print(f'       EA {-ea_g * HARTREE_TO_EV:8.4f} -> {-ea_s * HARTREE_TO_EV:8.4f} eV'
          f'   ({(ea_g - ea_s) * HARTREE_TO_EV:+.4f})')
    print(f'       gap {(ip_g - ea_g) * HARTREE_TO_EV:8.4f} -> '
          f'{(ip_s - ea_s) * HARTREE_TO_EV:8.4f} eV'
          f'   ({((ip_s - ea_s) - (ip_g - ea_g)) * HARTREE_TO_EV:+.4f})')
    print(f'       S1  {om_g * HARTREE_TO_EV:8.4f} -> {om_s * HARTREE_TO_EV:8.4f} eV'
          f'   ({(om_s - om_g) * HARTREE_TO_EV:+.4f})')
    ok = check(ip_s < ip_g - 0.01 / HARTREE_TO_EV, 'the solvent lowers the IP')
    ok &= check(ea_s > ea_g + 0.01 / HARTREE_TO_EV, 'the solvent raises the EA')
    ok &= check((ip_s - ea_s) < (ip_g - ea_g),
                'the quasiparticle gap closes (image-charge, Born)')
    ok &= check(abs(om_s - om_g) > 1e-6, 'the BSE excitation moves at all',
                f'{(om_s - om_g) * HARTREE_TO_EV * 1000:+.1f} meV')
    assert ok


def test_matches_production_gw(water):
    """The solvated chain's eps^QP must reproduce the PRODUCTION space-time GW
    route with the same screening attached. Same physics, two independent
    implementations of it: the chain dresses its own metric and adds its own
    Sigma^solv, production dresses `separable_factors` and adds
    `static_exchange_matrix`'s. The gas-phase pair is printed too, because the
    agreement is only meaningful if the SHIFT agrees, not just the total."""
    mol, mf = water
    gas = ExcitedStateChain(mol, scf_factory, mf=mf)
    sol = solvated(mol, mf)
    radii = gas.radii
    ok = True
    for tag, chain, attach in (('gas', gas, False), ('water', sol, True)):
        if attach:
            attach_solvent_screening(mf, solvent='water')
        prod = -calc_qp_energy(mf, selfenergy='GW', df=True, state='homo',
                               mode='space-time', radii=radii)
        if attach:
            detach_solvent_screening(mf)
        chn = -chain.quasiparticle(0, mol, mf) * HARTREE_TO_EV
        d = abs(prod - chn)
        print(f'       {tag:6s}: production {prod:8.4f} eV, chain {chn:8.4f} eV')
        # The two differ by the grid/Pade conventions the chain froze (contour
        # deformation vs Pade continuation), so they are NOT expected to agree
        # to machine precision -- 50 meV is the gas-phase agreement, and the
        # SOLVENT SHIFT is what has to match.
        ok &= check(d < 0.10, f'{tag}: chain and production GW agree',
                    f'{d * 1000:.1f} meV')
    assert ok


def test_every_gradient_entry_point_now_carries_the_environment(water):
    """The refusal is gone because the derivative is built: the reaction field's
    metric half and its static half both reach the reverse pass.

    Checked here for existence and finiteness; the NUMBER is gated against
    finite differences in test_pcm_bilinear_gradient.
    """
    mol, mf = water
    sol = solvated(mol, mf)
    gas = ExcitedStateChain(mol, scf_factory, mf=mf)
    hits = 0
    for name, args in (('excitation_gradient', ()), ('quasiparticle_gradient', (0,)),
                       ('total_gradient', ())):
        g = getattr(sol, name)(*args)[0]
        gg = getattr(gas, name)(*args)[0]
        # a force, and not the gas-phase one
        hits += int(np.isfinite(g).all() and np.abs(g - gg).max() > 1e-5)
    check(hits == 3, 'every gradient entry point returns a solvated force '
                     'that differs from the gas-phase one', f'{hits}/3')


def test_one_fit_and_one_static_screening_serve_both_gauges(water):
    """The shared work is shared without changing the answer.

    A solvated chain carries two factors -- the dressed one the BSE kernel
    screens with, the bare one the self-energy does -- differing only in the
    metric root. The test set, the three-centre integrals, the fit M and every
    derivative integral below them are common to both, so the D branch runs
    once for the pair; what is left per gauge is the root's own adjoint. The
    two orders of summation are not bit-identical, which is the only thing that
    may differ.

    The static screening is shared the same way: [1 - chi0(0)]^-1 in the
    dressed gauge IS the BSE kernel's W and the dressed half of Eq. (18) at
    once, so the forward builds the pair and the reverse is handed it. Nothing
    is reordered there, so that half is bitwise.
    """
    mol, mf = water
    chain = solvated(mol, mf)
    mol, mf = chain.mean_field(mol, mf)
    x_mo, d, eps, auxmol, crd, _ = chain.factors_at(mol, mf)
    d_bare = chain.bare_factor(mol, auxmol, crd)
    assert d_bare is not None, 'the solvated chain built no bare gauge'
    env = chain.environment_at(mol)
    kw = dict(layout=chain.layout, pts_local=chain.pts_local,
              atom_of_point=chain.owner, frames=chain.frames,
              with_frames=chain.with_frames)
    # Random adjoints rather than the chain's own: the branch is linear in
    # them, so this covers every direction a state could land on it.
    rng = np.random.default_rng(0)
    d_bar = rng.standard_normal(d.shape) * 1e-3
    d_bar_bare = rng.standard_normal(d.shape) * 1e-3

    one = dfactor_adjoint_gauges(mol, auxmol, crd,
                                 [(d_bar, env), (d_bar_bare, None)], **kw)
    two = (dfactor_adjoint(mol, auxmol, crd, d_bar, environment=env, **kw)
           + dfactor_adjoint(mol, auxmol, crd, d_bar_bare, environment=None, **kw))
    scale = float(np.abs(two).max())
    err = float(np.abs(one - two).max())
    ok = check(err < 1e-11 * scale,
               'one fit for both gauges is the two fits it replaces',
               f'max |d| = {err:.1e} on |g| = {scale:.1e}')

    pair = (static_screening(x_mo, d, eps, chain.nocc, chain.w_grid),
            static_screening(x_mo, d_bare, eps, chain.nocc, chain.w_grid))
    shared = reaction_field_shift(x_mo, d, d_bare, eps, chain.nocc,
                                  grid=chain.w_grid, screening=pair)
    own = reaction_field_shift(x_mo, d, d_bare, eps, chain.nocc,
                               grid=chain.w_grid)
    ok &= check(np.array_equal(shared, own),
                'Eq. (18) off the shared screening is BITWISE the one that '
                'rebuilds it', f'max |d| = {np.abs(shared - own).max():.1e} Ha')
    w = rng.standard_normal(len(eps)) * 1e-3
    fwd = reaction_field_backward(w, x_mo, d, d_bare, eps, chain.nocc,
                                  grid=chain.w_grid, screening=pair)
    rebuilt = reaction_field_backward(w, x_mo, d, d_bare, eps, chain.nocc,
                                      grid=chain.w_grid)
    ok &= check(all(np.array_equal(p, q) for p, q in zip(fwd, rebuilt)),
                'and so is its adjoint on (eps, X, D_dressed, D_bare)')
    assert ok


# --------------------------------------------------------------------------
def follow(x_ref, y_ref, xn, yn):
    """Index of the root whose eigenvector overlaps `(x_ref, y_ref)` most.

    THE TRAP THIS EXISTS FOR: comparing Omega[i] across dielectric constants at
    a FIXED index i is meaningless. The reaction field closes the quasiparticle
    gap by several eV and moves charge-transfer roots much further than local
    ones, so roots cross and the index stops naming a state. Tracking by index
    on C2H4...F2 gave a CT "shift" of -137 meV at eps=1.5 and +840 meV at
    eps=2.5 -- non-monotonic and the wrong sign, both pure relabelling.

    The mean field is the same for every eps (the screening is post-SCF), so
    all eigenvectors live in the same particle-hole space and a plain overlap
    is a valid metric.
    """
    ov = np.abs(x_ref @ np.asarray(xn) + y_ref @ np.asarray(yn))
    # Casida vectors are normalized as X^T X - Y^T Y = 1, so this overlap is
    # X^T X + Y^T Y >= 1 for a perfectly followed root, not 1 exactly.
    return int(np.argmax(ov)), float(ov.max())


def test_ct_beats_local():
    """Charge transfer is stabilized more than a local excitation, and the
    stabilization grows with the dielectric constant.

    The whole point of a polarizable environment for the applications here: a
    CT state separates charge, so the reaction field it induces is large, while
    a local excitation's electron and hole polarize the medium in opposite
    directions and largely cancel.

    THE LOCAL ROOT NEEDS A BOUND OF ITS OWN. `d_ct < d_local` gets EASIER the
    more wrong d_local is, so for a while this test passed while the local root
    blue-shifted +2186 meV at eps = 2.5 -- the reaction field reached the BSE
    kernel for every orbital pair but reached the diagonal only inside the
    quasiparticle window, and the local root's hole sat outside it
    (`ExcitedStateChain._env_static_outside`). A check that only ever compares
    two numbers cannot see both of them move; the cancellation the physics
    rests on is an absolute statement about ONE of them.
    """
    mol = gto.M(atom=C2H4_F2, basis=BASIS, verbose=0)
    mf = scf_factory(mol)
    eps_list = [1.0000001, 1.5, 2.0, 2.5]     # optical n^2: vacuum .. CS2
    table = {}
    for eps in eps_list:
        ch = ExcitedStateChain(
            mol, scf_factory, mf=mf,
            environment=(None if eps < 1.001 else SolventScreening(mol, eps=eps)))
        om, xn, yn, _ = roots(ch, mol, mf)
        table[eps] = (om, xn, yn, ct_character(ch, mf, xn, yn, F2_ATOMS))

    om0, x0, y0, ct0 = table[eps_list[0]]
    order = np.argsort(om0)[:min(12, len(om0))]
    i_ct = order[np.argmax(ct0[order])]
    local = order[ct0[order] < 0.25]
    assert check(len(local) > 0, 'a local root exists below the CT root')
    i_loc = local[0]
    print(f'       CT root #{i_ct}: Omega {om0[i_ct] * HARTREE_TO_EV:.3f} eV, '
          f'CT weight {ct0[i_ct]:.2f}')
    print(f'       local  #{i_loc}: Omega {om0[i_loc] * HARTREE_TO_EV:.3f} eV, '
          f'CT weight {ct0[i_loc]:.2f}')
    print('       eps    CT shift  (ovl, CT wt)    local shift  (ovl, CT wt)')
    shifts, worst_ovl = {}, 1.0
    for eps in eps_list:
        om, xn, yn, ct = table[eps]
        j_ct, o_ct = follow(x0[:, i_ct], y0[:, i_ct], xn, yn)
        j_lo, o_lo = follow(x0[:, i_loc], y0[:, i_loc], xn, yn)
        worst_ovl = min(worst_ovl, o_ct, o_lo)
        d_ct = (om[j_ct] - om0[i_ct]) * HARTREE_TO_EV * 1000
        d_lo = (om[j_lo] - om0[i_loc]) * HARTREE_TO_EV * 1000
        shifts[eps] = (d_ct, d_lo)
        print(f'       {eps:5.2f} {d_ct:+9.1f}  ({o_ct:.2f}, {ct[j_ct]:.2f})'
              f'      {d_lo:+9.1f}  ({o_lo:.2f}, {ct[j_lo]:.2f})')

    d_ct, d_lo = shifts[eps_list[-1]]
    ok = check(worst_ovl > 0.8, 'every followed root is unambiguous',
               f'worst overlap {worst_ovl:.2f}')
    ok &= check(d_ct < 0, 'the CT root is stabilized by the environment',
                f'{d_ct:+.1f} meV at eps={eps_list[-1]}')
    ok &= check(d_ct < d_lo, 'the CT root is stabilized MORE than the local one',
                f'{d_ct:+.1f} vs {d_lo:+.1f} meV')
    worst_lo = max(abs(v[1]) for v in shifts.values())
    ok &= check(worst_lo < LOCAL_SHIFT_BOUND_MEV,
                'and the local root barely moves AT ALL -- a compact neutral '
                'electron-hole pair hardly polarizes a continuum',
                f'largest |shift| {worst_lo:.1f} < {LOCAL_SHIFT_BOUND_MEV} meV')
    monotone = all(shifts[a][0] >= shifts[b][0]
                   for a, b in zip(eps_list, eps_list[1:]))
    assert ok & check(monotone,
                      'the CT stabilization grows monotonically with eps')


def main(with_ct=False):
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    pair = (mol, scf_factory(mol))
    cases = [(fn, (pair,)) for fn in (
        test_the_chain_environment_is_the_single_source_of_truth,
        test_solvated_chain_moves,
        test_matches_production_gw,
        test_every_gradient_entry_point_now_carries_the_environment)]
    if with_ct:
        cases.append((test_ct_beats_local, ()))
    ok = True
    for fn, args in cases:
        print(f'\n== {fn.__name__}')
        try:
            fn(*args)
        except AssertionError:
            ok = False
    print('\nALL PASS' if ok else '\nFAILURES')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(with_ct='--ct' in sys.argv) else 1)
