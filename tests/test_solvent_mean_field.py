"""The ground state a continuum accepts must be the one it relaxed.

Non-equilibrium solvation is TWO dielectric constants. The ground state relaxes
inside PCM(eps_static), because the solvent nuclei have had time to reorient
around a state that is already there; only the response to a fast excitation is
optical, and that is the vtilde `SolventScreening` hands out everywhere else.
Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) split a
solvatochromic shift into those two halves, and for a local excitation the
ground-state half is the larger one.

`mean_field` applies it. It hands back a factory's own PCM mean field rather
than re-converging it, which is what lets a caller pass a solvated reference
straight through. A PCM at some OTHER dielectric constant is a different ground
state, though: keeping it would put the optical response at eps on top of a
reaction field built at the wrong constant, and nothing downstream can see that
it happened.

Run: python tests/test_solvent_mean_field.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto, scf
from pyscf import solvent as pyscf_solvent

from src.Base.solvent_screening import SolventScreening


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def pcm_at(mf, eps):
    """`mf` carrying a PCM at `eps`, unconverged: the check runs before the SCF."""
    wrapped = pyscf_solvent.PCM(mf)
    wrapped.with_solvent.eps = eps
    return wrapped


def test_a_pcm_at_another_constant_is_refused(mol, base_mf, water):
    """The wrong ground state must be named, not adopted."""
    try:
        water.mean_field(mol, lambda _mol: pcm_at(base_mf, 20.0))
        return check(False, 'a PCM at another constant is refused')
    except ValueError as exc:
        return check('20.0' in str(exc) and '78.39' in str(exc),
                     'a PCM at another constant is refused, and both named',
                     str(exc)[:90] + '...')


def test_a_pcm_at_eps_static_is_returned_untouched(mol, base_mf, water):
    """One reaction field, applied once."""
    solvated = pcm_at(base_mf, water.eps_static)
    return check(water.mean_field(mol, lambda _mol: solvated) is solvated,
                 'a PCM already at eps_static is passed straight through')


def test_a_bare_factory_is_wrapped_at_eps_static(mol, base_mf, water):
    """Nothing attached yet: the SCF is put inside PCM(eps_static) here."""
    wrapped = water.mean_field(mol, lambda _mol: base_mf)
    return check(wrapped is not base_mf and wrapped.converged
                 and wrapped.with_solvent.eps == water.eps_static,
                 'a bare factory is wrapped and re-converged at eps_static',
                 f'E = {wrapped.e_tot:.8f} Ha against gas phase '
                 f'{base_mf.e_tot:.8f} Ha')


def test_a_solvent_name_carries_both_constants(mol):
    """A caller cannot take one without the other: the name supplies the
    optical constant the response uses and the static one the ground state
    relaxes in."""
    named = SolventScreening(mol, solvent='water')
    ok = check(abs(named.eps - 1.7764) < 1e-3 and abs(named.eps_static - 78.355) < 1e-2,
               'a named solvent carries eps_inf and eps_static together',
               f'{named.eps:.4f} / {named.eps_static:.3f}')
    bare = SolventScreening(mol, eps=1.78)
    ok &= check(bare.eps_static is None,
                'an explicit optical eps leaves the ground state bare unless told')
    try:
        SolventScreening(mol, eps=1.78, eps_static=1.2)
        ok &= check(False, 'a static constant below the optical one is refused')
    except ValueError as exc:
        ok &= check('cannot be the smaller' in str(exc),
                    'a static constant below the optical one is refused')
    return ok


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    mol = gto.M(atom='O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468',
                basis='cc-pvdz', verbose=0)
    base_mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    base_mf.kernel()
    water = SolventScreening(mol, eps=1.78, eps_static=78.39)

    all_ok = True
    print('\n-- the two constants travel together')
    all_ok &= test_a_solvent_name_carries_both_constants(mol)
    print('\n-- and the ground state is relaxed in the static one, exactly once')
    all_ok &= test_a_bare_factory_is_wrapped_at_eps_static(mol, base_mf, water)
    all_ok &= test_a_pcm_at_eps_static_is_returned_untouched(mol, base_mf, water)
    all_ok &= test_a_pcm_at_another_constant_is_refused(mol, base_mf, water)
    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
