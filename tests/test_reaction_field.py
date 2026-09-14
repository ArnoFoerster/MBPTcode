"""Duchemin et al. Eq. (18): the reaction field's quasiparticle shift.

The economy under test is that Delta W needs only ONE chi0. The dressed
screening already contains it, and the bare partner follows by a congruence in
the auxiliary gauge, so an explicit second screening must reproduce it exactly
rather than approximately.

The second half is the cache. Delta W screens with the eigenvalues it is built
at, so an evGW cycle must not read the previous cycle's array -- that is what
the spectrum key buys. But it is also a property of the CAVITY: two continua at
different dielectric constants around one mean field give different shifts, and
a cache that keys on the spectrum alone hands the second one the first one's
array. A solvatochromic shift is exactly that comparison, so the collision is
not hypothetical.

Run: python tests/test_reaction_field.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df, dft, gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import attach_environment, dresses_interaction
from src.Base.solvent_screening import (SolventScreening,
                                        detach_solvent_screening)
from src.SingleReference.GW.reaction_field import (
    bare_gauge_transform, environment_quasiparticle_shift, quasiparticle_shift,
    screened_interaction_difference)
from src.SingleReference.LinearResponse.davidson import isdf_bse_factors

BASIS, EPS_INF = 'cc-pvdz', 1.78
WATER = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def build():
    """Both auxiliary gauges of one mean field: bare, and dressed by a continuum."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis=BASIS + '-jkfit')
    mf.grids.prune = None
    mf.conv_tol = 1e-11
    mf.kernel()
    nocc = mol.nelectron // 2
    env = SolventScreening(mol, eps=EPS_INF)
    aux = df.addons.make_auxmol(mol, auxbasis=BASIS + '-ri')

    detach_solvent_screening(mf)
    x_bare, d_bare, w_bare = isdf_bse_factors(mf, mol, nocc,
                                              auxbasis=BASIS + '-ri')
    attach_environment(mf, env)
    x_dr, d_dr, w_dr = isdf_bse_factors(mf, mol, nocc, auxbasis=BASIS + '-ri')
    detach_solvent_screening(mf)
    return dict(mol=mol, mf=mf, nocc=nocc, env=env, auxmol=aux, x=x_dr,
                d_dr=d_dr, w_dr=w_dr, d_bare=d_bare, w_bare=w_bare,
                x_bare=x_bare)


def test_one_chi0_reproduces_two_screenings(s):
    """The congruence must be EXACT, not merely close: it is what makes the
    reaction-field shift cost one Dyson inversion instead of a second chi0."""
    ok = check(np.abs(s['x'] - s['x_bare']).max() < 1e-12,
               'the collocation is shared between the two gauges')
    explicit = (s['d_dr'] @ s['w_dr'] @ s['d_dr'].T
                - s['d_bare'] @ s['w_bare'] @ s['d_bare'].T)
    cheap = screened_interaction_difference(s['w_dr'], s['d_dr'], s['auxmol'],
                                            s['env'])
    scale = np.abs(explicit).max()
    d = np.abs(cheap - explicit).max()
    ok &= check(scale > 1e-3, 'Delta W is not trivially small', f'|dW|max {scale:.2e}')
    ok &= check(d < 1e-8 * scale,
                'the congruence reproduces two independent screenings',
                f'{d:.1e} against |dW|max {scale:.1e}')
    return ok


def test_delta_w_is_a_reduced_interaction(s):
    """vtilde screens, so W with the reaction field is weaker than without and
    every orbital's self-polarization is negative -- both gap edges move toward
    each other, which is the Born stabilization of cation and anion alike."""
    dw = screened_interaction_difference(s['w_dr'], s['d_dr'], s['auxmol'],
                                         s['env'])
    nocc = s['nocc']
    ok = check(np.abs(dw - dw.T).max() < 1e-10 * np.abs(dw).max(),
               'Delta W is symmetric')
    density = s['x'] ** 2
    self_pol = np.einsum('kp,kl,lp->p', density, dw, density, optimize=True)
    ok &= check((self_pol < 0).all(),
                'a reduced interaction cannot raise <pp|dW|pp>',
                f'largest {self_pol.max():.2e}')
    shift = quasiparticle_shift(s['x'], dw, nocc)
    gap = (shift[nocc] - shift[nocc - 1]) * HARTREE_TO_EV
    ok &= check((shift[:nocc] > 0).all() and (shift[nocc:] < 0).all() and gap < 0,
                'occupied levels rise, virtual levels fall, the gap closes',
                f'HOMO {shift[nocc - 1] * HARTREE_TO_EV:+.3f}, LUMO '
                f'{shift[nocc] * HARTREE_TO_EV:+.3f}, gap {gap:+.3f} eV')
    return ok


def test_the_gas_phase_is_a_no_op(s):
    """No environment, no shift -- so callers add it unconditionally."""
    shift = quasiparticle_shift(s['x'], None, s['nocc'])
    return check(bare_gauge_transform(s['auxmol'], None) is None
                 and screened_interaction_difference(
                     s['w_dr'], s['d_dr'], s['auxmol'], None) is None
                 and shift.shape == (s['x'].shape[1],) and not shift.any(),
                 'the gas phase returns None, and a zero shift of the right shape')


def test_the_transform_follows_the_one_decision(s):
    """The gauge map exists exactly when the environment dresses v.

    `bare_gauge_transform` maps a dressed auxiliary gauge onto the bare one, so
    it exists precisely when there is a dressed interaction to map from; the
    routes read `transform is not None` and are thereby reading
    `dresses_interaction` rather than a second, independent rule.
    """
    return check(dresses_interaction(s['env'], s['auxmol']) is True
                 and bare_gauge_transform(s['auxmol'], s['env']) is not None
                 and dresses_interaction(None, s['auxmol']) is False,
                 'the gauge map exists exactly when the environment dresses v')


def test_it_is_not_the_cohsex_sum(s):
    """The shipped `cohsex_correction` sums over every orbital and uses the
    BARE vtilde; Eq. (18) is a self-element of the SCREENED one. The two differ
    by hundreds of meV on the gap, which is the whole reason for this module."""
    nocc = s['nocc']
    dw = screened_interaction_difference(s['w_dr'], s['d_dr'], s['auxmol'],
                                         s['env'])
    eq18 = quasiparticle_shift(s['x'], dw, nocc)
    gap_eq18 = (eq18[nocc] - eq18[nocc - 1]) * HARTREE_TO_EV
    d = np.diag(s['env'].cohsex_correction(s['mol'], s['mf'].mo_coeff, nocc))
    gap_cohsex = (d[nocc] - d[nocc - 1]) * HARTREE_TO_EV
    return check(gap_cohsex < 0 and gap_eq18 < 0
                 and abs(gap_eq18 - gap_cohsex) > 0.1,
                 'Eq. (18) and the COHSEX sum are different operators',
                 f'gap closure {gap_eq18:.3f} vs {gap_cohsex:.3f} eV')


def test_the_compact_form_matches_the_explicit_delta_w(s):
    """`separable_quasiparticle_shift` never builds Delta W -- (M, M) in a rank
    that grows with the system -- and must land on the same numbers as the
    reference form that does."""
    from src.SingleReference.GW.reaction_field import separable_quasiparticle_shift
    t = bare_gauge_transform(s['auxmol'], s['env'])
    dw = screened_interaction_difference(s['w_dr'], s['d_dr'], s['auxmol'],
                                         s['env'])
    explicit = quasiparticle_shift(s['x'], dw, s['nocc'])
    compact = separable_quasiparticle_shift(s['x'], s['d_dr'], s['w_dr'], t,
                                            s['nocc'])
    d = np.abs(compact - explicit).max()
    return check(d < 1e-10 * np.abs(explicit).max(),
                 'the compact contraction equals the explicit Delta W form',
                 f'{d:.1e} Ha')


# ------------------------------------------------------------ the cache
def test_a_second_environment_gets_its_own_reaction_field():
    """A stronger continuum must not be served the weaker one's Delta W."""
    mol = gto.M(atom=WATER, basis=BASIS, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.kernel()
    nocc = mol.nelectron // 2
    water = SolventScreening(mol, eps=EPS_INF)
    stronger = SolventScreening(mol, eps=3.0)

    def gap(shift):
        return (shift[nocc] - shift[nocc - 1]) * HARTREE_TO_EV

    attach_environment(mf, water)
    s1 = environment_quasiparticle_shift(mf)
    attach_environment(mf, stronger)
    s2 = environment_quasiparticle_shift(mf)
    ok = check(np.abs(s2 - s1).max() > 1e-3,
               'a larger dielectric screens more, so Delta W grows',
               f'gap closure {gap(s1):.4f} eV at eps=1.78 vs {gap(s2):.4f} at 3.0')

    del mf._reaction_field_shift
    ok &= check(np.array_equal(s2, environment_quasiparticle_shift(mf)),
                'the cached array is what the uncached route returns')
    attach_environment(mf, water)
    ok &= check(np.array_equal(environment_quasiparticle_shift(mf), s1),
                'and the first continuum gets its own back, not the resident one')
    detach_solvent_screening(mf)
    ok &= check(environment_quasiparticle_shift(mf) is None,
                'the gas-phase exit sits BEFORE the cache')

    attach_environment(mf, water)
    first = environment_quasiparticle_shift(mf)
    ok &= check(environment_quasiparticle_shift(mf) is first,
                'the cache still hits when nothing changed')
    detach_solvent_screening(mf)
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    s = build()
    all_ok = True
    print('\n-- 1. one chi0, two gauges')
    all_ok &= test_one_chi0_reproduces_two_screenings(s)
    all_ok &= test_the_transform_follows_the_one_decision(s)
    all_ok &= test_the_gas_phase_is_a_no_op(s)
    print('\n-- 2. the shift itself')
    all_ok &= test_delta_w_is_a_reduced_interaction(s)
    all_ok &= test_the_compact_form_matches_the_explicit_delta_w(s)
    all_ok &= test_it_is_not_the_cohsex_sum(s)
    print('\n-- 3. the cache keys on the spectrum AND the cavity')
    all_ok &= test_a_second_environment_gets_its_own_reaction_field()
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
