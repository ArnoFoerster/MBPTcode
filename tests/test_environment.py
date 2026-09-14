"""The Environment seam: a chain sees ALL of an environment or none of it.

An environment enters in two places and only one of them is a choice.
`aux_kernel` dresses the interaction, v -> v + vtilde, and everything built
from the interaction then carries it -- W, the BSE kernel, and the Eq. (18)
self-polarization shift, which is the STATIC APPROXIMATION to
Sigma[W_dressed] - Sigma[W_bare] and therefore follows from `aux_kernel` alone
rather than from a second switch. `static_self_energy` is the other channel and
a different operator, for the routes that never form Delta W.

The gate is that `dresses_interaction` is the single place the first question
is asked, so a route reading `transform is not None` is reading that one
decision and not a rule of its own.

Run: python tests/test_environment.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import df as pyscf_df, gto, qmmm, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.environment import (Environment, NoEnvironment, PointCharges,
                                  attach_environment, attached_environment,
                                  dresses_interaction, environment_of)
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import SolventScreening
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.reaction_field import environment_quasiparticle_shift

ATOM = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
BASIS, AUXBASIS = 'cc-pvdz', 'cc-pvdz-ri'


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def test_every_environment_satisfies_the_protocol(mol):
    """Structural, not by inheritance: `SolventScreening` never names it.

    The check is worth its line because the failure is quiet: a class that
    grows a member the protocol declares and its siblings do not still imports,
    still runs on the route that knows it, and only the routes that trusted the
    contract break.
    """
    import typing
    members = sorted(typing._get_protocol_attrs(Environment))
    ok = True
    for env in (NoEnvironment(), PointCharges([[0, 0, 4.0]], [-0.5]),
                SolventScreening(mol, solvent='water')):
        missing = [a for a in members if not hasattr(env, a)]
        ok &= check(isinstance(env, Environment) and not missing,
                    f'{type(env).__name__} satisfies Environment',
                    f'missing {missing}' if missing else f'{len(members)} members')
    return ok


def test_none_means_not_screening_and_nothing_else(mol):
    """The rule the chokepoints rest on. A form that is absent because the
    environment does not respond returns None and the route carries on with the
    bare interaction, which is correct. A form absent because a SCREENING
    environment has no representation for that route must REFUSE -- answering
    None there hands the route a gas-phase interaction under a dressed
    calculation, and it reads as a converged result."""
    forms = ('aux_kernel', 'whitened_transform', 'kernel_mo', 'kernel_ao')
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    mf = factory(mol)
    ok = True
    for env in (NoEnvironment(), PointCharges([[0, 0, 4.0]], [-0.5])):
        got = (env.aux_kernel(auxmol), env.whitened_transform(mol, mf),
               env.kernel_mo(mol, mf.mo_coeff), env.kernel_ao(mol))
        ok &= check(not env.screens and all(g is None for g in got),
                    f'{type(env).__name__}: screens=False, every form None',
                    ', '.join(forms))
    env = SolventScreening(mol, solvent='water')
    got = (env.aux_kernel(auxmol), env.whitened_transform(mol, mf),
           env.kernel_mo(mol, mf.mo_coeff), env.kernel_ao(mol))
    return ok & check(env.screens and all(g is not None for g in got),
                      'SolventScreening: screens=True, every form built',
                      ', '.join(forms))


def test_only_a_responding_environment_dresses_the_interaction(mol):
    """A fixed charge does not respond, so it screens nothing: it changes the
    mean field and nothing after it."""
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    ok = check(dresses_interaction(None, auxmol) is False
               and dresses_interaction(NoEnvironment(), auxmol) is False,
               'the gas phase does not dress the interaction')
    ok &= check(dresses_interaction(PointCharges([[0, 0, 4.0]], [-0.5]),
                                    auxmol) is False,
                'nor do permanent point charges, which do not respond')
    ok &= check(dresses_interaction(SolventScreening(mol, solvent='water'),
                                    auxmol) is True,
                'a continuum does')
    return ok


def test_the_dressed_metric_is_the_whole_substitution(mol):
    """The interaction reaches a separable factorization only through the
    auxiliary metric, so dressing V^(1/2) IS v -> v + vtilde. It stays positive
    -- a negative eigenvalue means the discretized field over-screens the bare
    interaction, which is an error rather than a truncation."""
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    env = SolventScreening(mol, solvent='water')
    bare = aux_metric_sqrt(auxmol, None)
    dressed = aux_metric_sqrt(auxmol, env)
    v = auxmol.intor('int2c2e', aosym='s1')
    ok = check(np.abs(bare @ bare - v).max() < 1e-8 * np.abs(v).max(),
               'the bare metric squares back to (P|Q)')
    d = dressed @ dressed - (v + env.aux_kernel(auxmol))
    ok &= check(np.abs(d).max() < 1e-8 * np.abs(v).max(),
                'and the dressed one to (P|Q) + vtilde',
                f'{np.abs(d).max():.1e}')
    ok &= check(np.linalg.eigvalsh(dressed).min() > -1e-10,
                'v + vtilde is still a positive kernel')
    return ok


def test_attaching_is_reversible_and_scoped(mol):
    """A route evaluates part of its chain on a mean field the caller may use
    elsewhere; leaving a cavity attached would silently move their numbers."""
    mf = factory(mol)
    env = SolventScreening(mol, solvent='water')
    ok = check(isinstance(environment_of(mf), NoEnvironment),
               'a fresh mean field is in the gas phase')
    attach_environment(mf, env)
    ok &= check(environment_of(mf) is env, 'attaching is what routes read')
    with attached_environment(mf, None):
        ok &= check(environment_of(mf) is not env,
                    'the context manager detaches inside the block')
    ok &= check(environment_of(mf) is env,
                'and puts the previous one back on the way out')
    return ok


def test_the_eq18_shift_follows_the_one_decision(mol):
    """An environment that dresses the interaction carries the shift; one that
    does not carries none. Not a separate switch: the same question."""
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    nocc = mol.nelectron // 2
    ok = True
    for env, dresses in ((NoEnvironment(), False),
                         (PointCharges([[0, 0, 4.0]], [-0.5]), False),
                         (SolventScreening(mol, solvent='water'), True)):
        mf = factory(mol)
        attach_environment(mf, env)
        shift = environment_quasiparticle_shift(mf, mol, nocc)
        ok &= check((shift is not None) == dresses
                    == dresses_interaction(env, auxmol),
                    f'{type(env).__name__}: shift present iff it dresses v',
                    'no shift' if shift is None else
                    f'gap closure '
                    f'{(shift[nocc] - shift[nocc - 1]) * HARTREE_TO_EV:+.3f} eV')
    return ok


def test_point_charges_move_the_mean_field_and_nothing_after_it(mol):
    """Everything the charges do enters through h_core, so the post-SCF routes
    see a different reference and the same interaction."""
    charges, coords = np.array([-0.4, 0.4]), np.array([[0, 0, 4.0], [0, 0, 5.0]])
    env = PointCharges(coords, charges)
    mf = env.mean_field(mol, factory)
    ok = check(isinstance(mf, qmmm.itrf.QMMM) and mf.converged,
               'the SCF re-converges inside the charges',
               f'E = {mf.e_tot:.8f} Ha')
    ok &= check(env.mean_field(mol, lambda _m: mf) is mf,
                'a factory that already carries THESE charges is passed through')
    other = PointCharges(coords, -charges)
    try:
        other.mean_field(mol, lambda _m: mf)
        ok &= check(False, 'a factory carrying DIFFERENT charges is refused')
    except ValueError as exc:
        ok &= check('different set of point charges' in str(exc),
                    'a factory carrying DIFFERENT charges is refused')
    gas = calc_qp_energy(factory(mol), mode='casida')
    attach_environment(mf, env)
    with_q = calc_qp_energy(mf, mode='casida')
    ok &= check(abs(with_q - gas) > 1e-3,
                'the quasiparticle energy moves, through the mean field alone',
                f'{gas:.4f} -> {with_q:.4f} eV')
    return ok


def test_the_gas_phase_is_the_default_everywhere(mol):
    """So a caller holding no environment never special-cases one."""
    auxmol = pyscf_df.addons.make_auxmol(mol, auxbasis=AUXBASIS)
    env = NoEnvironment()
    mf = factory(mol)
    return check(env.aux_kernel(auxmol) is None
                 and env.static_self_energy(mf) is None
                 and env.for_geometry(mol) is env
                 and env.mean_field(mol, factory) is not None,
                 'NoEnvironment answers every channel with nothing')


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis=BASIS, verbose=0)
    all_ok = True
    print('\n-- 1. the contract')
    all_ok &= test_every_environment_satisfies_the_protocol(mol)
    all_ok &= test_none_means_not_screening_and_nothing_else(mol)
    all_ok &= test_the_gas_phase_is_the_default_everywhere(mol)
    all_ok &= test_attaching_is_reversible_and_scoped(mol)
    print('\n-- 2. one decision: does it dress the interaction?')
    all_ok &= test_only_a_responding_environment_dresses_the_interaction(mol)
    all_ok &= test_the_dressed_metric_is_the_whole_substitution(mol)
    all_ok &= test_the_eq18_shift_follows_the_one_decision(mol)
    print('\n-- 3. permanent charges act through the mean field alone')
    all_ok &= test_point_charges_move_the_mean_field_and_nothing_after_it(mol)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
