"""ONE assembly of E_0, gated bitwise against every surface that reports it.

E_0 = E_ref + (E_x^HF[rho] - E_xc[rho]) + E_c^dRPA is the plasmon (Klein)
ground state of Toelle, Kitsaras and Loos (arXiv:2507.02160) Eq. (15). It used
to be spelled in eight places, and two of them disagreed; `ground_state_energy`
is now the only one, so what these checks have to establish is that the single
assembly returns the SAME BITS as each surface's own arithmetic, and that the
declaration it carries names the functional the number belongs to.

Every equivalence gate below is also shown to FAIL, once, under the omission it
exists to catch, so that none of them is a check that cannot fail:

    E_c^dRPA dropped from the Hartree-Fock total      6.29 eV
    E_x^HF - E_xc dropped from the PBE0 total         8.55 eV
    `spin` dropped from `DenseBSESurface.refreeze`    0.78 eV, and a triplet
                                                      comes back a singlet

All on water/cc-pVDZ. `==` is deliberate throughout: `pytest.approx` would pass
on a reassociated sum, which is exactly the change that moves an energy under a
geometry optimizer without moving any test.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import dft, gto, scf

from src.Base.constants import (HARTREE_TO_EV, SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.declaration import GroundState
from src.SingleReference.LinearResponse.rpa_energy import (
    declared_ground_state, exx_double_counting, ground_state_energy,
    reference_energy)
from src.gradients.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                          QuasiparticleSurface, mo_eri)
from src.gradients.quasi_boson_adjoint import QPqbAdjoint as QPqb
from src.gradients.rpa_bse_surface import RPABSESurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
#: The working directory a fresh interpreter needs for `import src...`.
REPO_ROOT = Path(__file__).resolve().parents[1]

_SOLUTIONS = {}


def _solved(mol, xc):
    """One SCF solution per (geometry, functional), converged to differentiate.

    Two solves at the same geometry are two solutions of the same equations,
    agreeing only to the convergence threshold; every comparison here is
    bitwise, so they have to be the same object.
    """
    key = (mol.atom_coords().tobytes(), xc)
    if key not in _SOLUTIONS:
        mf = scf.RHF(mol) if xc is None else dft.RKS(mol, xc=xc)
        if xc is not None:
            mf.grids.prune = None
        mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
        mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, 'SCF did not converge; nothing follows from it'
        _SOLUTIONS[key] = mf
    return _SOLUTIONS[key]


def rhf(mol):
    """Hartree-Fock: the reference where the double-counting term vanishes."""
    return _solved(mol, None)


def pbe0(mol):
    """PBE0: the reference where it does not, and E_KS is not E_HF[rho]."""
    return _solved(mol, 'pbe0')


def dense_correlation_energy(mf, mol):
    """E_c^dRPA the way `DenseRPASurface` builds it: the bare (pq|rs), no fit."""
    return QPqb(mf.mo_energy, mo_eri(mf, mol), mol.nelectron // 2,
                screening='rpa').qb.e_corr()


@pytest.fixture(scope='module')
def water():
    return gto.M(atom=H2O, basis=BASIS, verbose=0)


def test_the_dft_ground_state_is_the_mean_fields_own_energy(water):
    """kind='dft' is E_KS and nothing else -- bitwise the mean-field surface.

    The perturbation: adding the double-counting term would make it a
    DIFFERENT functional, 8.55 eV away, which is why that term belongs to
    kind='rpa' and there is no keyword here that reaches it.
    """
    mf = pbe0(water)
    declaration = GroundState('dft', 'pbe0')
    e0 = ground_state_energy(declaration, mf, water)
    assert e0.total == MeanFieldSurface(water, pbe0).total_energy(water)
    assert e0.total == float(mf.e_tot)
    assert e0.terms == {'E_ref': float(mf.e_tot)}
    assert e0.declaration == declaration

    perturbed = e0.total + exx_double_counting(mf, water)
    assert perturbed != e0.total
    assert abs(perturbed - e0.total) * HARTREE_TO_EV == pytest.approx(8.55,
                                                                     abs=0.01)


def test_the_rpa_ground_state_is_the_dense_surfaces_own_arithmetic(water):
    """kind='rpa' on Hartree-Fock is bitwise `DenseRPASurface.total_energy`.

    The oracle surface adds mf.e_tot + E_c in that order and every cubic gate
    is a difference measured from its numbers, so a reassociation here moves
    the reference under all of them at once.

    The perturbation: E_c^dRPA dropped leaves the mean-field energy, 6.29 eV
    above E_0 -- and, being a functional of the geometry, it moves the surface
    rather than only its zero.
    """
    mf = rhf(water)
    e_corr = dense_correlation_energy(mf, water)
    e0 = ground_state_energy(GroundState('rpa', 'hf'), mf, water,
                             e_corr=e_corr)
    assert e0.total == DenseRPASurface(water, rhf).total_energy(water)
    assert e0.terms['E_ref'] == float(mf.e_tot)
    assert e0.terms['E_x^HF - E_xc'] == 0.0
    assert e0.terms['E_c^dRPA'] == float(e_corr)

    perturbed = float(mf.e_tot)
    assert perturbed != e0.total
    assert abs(perturbed - e0.total) * HARTREE_TO_EV == pytest.approx(6.29,
                                                                     abs=0.01)


def test_the_rpa_ground_state_on_a_kohn_sham_reference(water):
    """kind='rpa' on PBE0 is bitwise `RPAGroundStateChain.energy`.

    E_0 is E_HF[rho] + E_c^dRPA whatever the starting point, so the chain's
    E_HF -- the Hartree-Fock energy AT the Kohn-Sham density -- has to come out
    of the same assembly the Hartree-Fock route uses.

    The perturbation: E_x^HF - E_xc dropped puts E_c^dRPA on top of E_KS, which
    counts the correlation inside E_xc a second time and carries an approximate
    exchange where the Klein functional wants the exact one -- 8.55 eV.
    """
    mf = pbe0(water)
    chain = RPAGroundStateChain(water, pbe0, mf=mf)
    e, e_hf, e_c = chain.energy(water, mf)
    e0 = ground_state_energy(declared_ground_state(mf, 'rpa'), mf, water,
                             e_corr=e_c)
    assert e0.total == e
    assert e0.declaration == GroundState('rpa', 'pbe0')
    assert e0.terms['E_ref'] == float(mf.e_tot)
    assert e0.terms['E_ref'] + e0.terms['E_x^HF - E_xc'] == e_hf
    assert e0.terms['E_c^dRPA'] == e_c
    # the middle term is the double counting AS THE TOTAL CARRIES IT, E_HF
    # minus E_ref, so it differs from the operator expression by the rounding
    # of that subtraction and not by any physics
    assert (e0.terms['E_x^HF - E_xc']
            == pytest.approx(exx_double_counting(mf, water), abs=1e-14))

    perturbed = float(mf.e_tot) + e_c
    assert perturbed != e0.total
    assert abs(perturbed - e0.total) * HARTREE_TO_EV == pytest.approx(8.55,
                                                                      abs=0.01)


def test_the_terms_are_the_declarations_and_sum_to_the_total(water):
    """A total alone cannot be checked against another route's: the terms say
    WHICH functional the number is, so they are the declaration's, in order,
    and they sum to the total with no remainder.

    The perturbation: on PBE0 every one of the three carries energy, so
    dropping any single term moves the sum off the total -- which is what a
    term silently left at zero, the double counting above all, would look
    like.
    """
    mf = pbe0(water)
    for declaration, e_corr in ((GroundState('dft', 'pbe0'), None),
                                (GroundState('rpa', 'pbe0'), -0.25)):
        e0 = ground_state_energy(declaration, mf, water, e_corr=e_corr)
        assert tuple(e0.terms) == declaration.terms()
        assert sum(e0.terms.values()) == e0.total
        for dropped in e0.terms:
            rest = sum(v for k, v in e0.terms.items() if k != dropped)
            assert rest != e0.total, dropped


def test_a_correlation_energy_is_refused_where_there_is_no_term_for_it(water):
    """The refusal and the case it is refused FOR, side by side.

    A caller holding an E_c^dRPA means kind='rpa'; dropping it silently into a
    kind='dft' number would report a Kohn-Sham energy under a declaration that
    promises the plasmon one. The mirror is as bad: kind='rpa' without E_c is
    E_0 short by 6.29 eV on this molecule.
    """
    mf = pbe0(water)
    e_corr = -0.2835745496101085
    with pytest.raises(ValueError, match='no correlation term'):
        ground_state_energy(GroundState('dft', 'pbe0'), mf, water,
                            e_corr=e_corr)
    allowed = ground_state_energy(GroundState('rpa', 'pbe0'), mf, water,
                                  e_corr=e_corr)
    assert allowed.terms['E_c^dRPA'] == e_corr

    with pytest.raises(ValueError, match='e_corr is None'):
        ground_state_energy(GroundState('rpa', 'pbe0'), mf, water)
    assert ground_state_energy(GroundState('dft', 'pbe0'), mf,
                               water).total == float(mf.e_tot)


def test_the_double_counting_is_exactly_zero_on_hartree_fock(water):
    """Zero, not small: `DenseRPASurface` and `QuasiparticleSurface` put
    mf.e_tot where the plasmon formula wants E_HF, and that is only the same
    number if this term is identically zero. It is 8.55 eV on PBE0, which is
    what those two surfaces refuse a Kohn-Sham factory over."""
    mf = rhf(water)
    assert exx_double_counting(mf, water) == 0.0
    assert reference_energy(mf, water) == float(mf.e_tot)

    exx = exx_double_counting(pbe0(water), water)
    assert exx != 0.0
    assert exx * HARTREE_TO_EV == pytest.approx(8.55, abs=0.01)


def test_every_surface_declares_the_functional_its_mean_field_carries(water):
    """Two total energies are comparable exactly when these agree, so each
    surface has to say which E_0 it is on rather than leaving it to be inferred
    from the factory that built it."""
    hartree_fock = GroundState('rpa', 'hf')
    assert DenseRPASurface(water, rhf).physics_ground_state == hartree_fock
    assert QuasiparticleSurface(water, rhf).physics_ground_state == hartree_fock
    assert DenseBSESurface(water, scf=rhf).physics_ground_state == hartree_fock
    assert (DenseBSESurface(water, scf=pbe0).physics_ground_state
            == GroundState('rpa', 'pbe0'))

    chain = RPAGroundStateChain(water, pbe0, mf=pbe0(water))
    assert chain.physics_ground_state == GroundState('rpa', 'pbe0')
    composed = RPABSESurface(water, pbe0, mf=pbe0(water), ground=chain)
    assert composed.physics_ground_state == chain.physics_ground_state


def test_refreeze_carries_the_spin_and_every_other_constructor_setting(water):
    """A refreeze that dropped a setting changes which functional the walk is
    on halfway through it.

    NEGATIVE CONTROL, and the reason this test exists: `spin` used not to
    travel, so a triplet surface came back a SINGLET at the first refreeze --
    0.78 eV away on water/cc-pVDZ, in the middle of an optimization, with
    nothing in the output to say the state had changed. The check is therefore
    the energy as well as the attribute: the refrozen surface reports the
    triplet's number bitwise, and the singlet's is a different number.
    """
    triplet = DenseBSESurface(water, 'BSE@GW', state=0, qp_orbs=[3, 4, 5],
                              filter_z=False, scf=rhf, auxbasis=None,
                              eri_blocks=True, spin='triplet')
    refrozen = triplet.refreeze(water)
    for setting in ('spin', 'eri_blocks', 'variant', 'state', '_qp_orbs_in',
                    'filter_z', '_scf', 'auxbasis'):
        assert getattr(refrozen, setting) == getattr(triplet, setting), setting
    assert refrozen.mol0 is water
    assert refrozen.physics_ground_state == triplet.physics_ground_state

    e_triplet = triplet.total_energy(water)
    assert refrozen.total_energy(water) == e_triplet
    singlet = DenseBSESurface(water, 'BSE@GW', state=0, qp_orbs=[3, 4, 5],
                              filter_z=False, scf=rhf, auxbasis=None,
                              eri_blocks=True, spin='singlet')
    e_singlet = singlet.total_energy(water)
    assert e_singlet != e_triplet
    assert (e_singlet - e_triplet) * HARTREE_TO_EV > 0.1, (
        'the singlet and the triplet of this surface are too close for the '
        'negative control to distinguish them')


def _modules_after(statement):
    """Which src.gradients modules a fresh interpreter has loaded after
    `statement`. A separate process is the only honest way to ask: inside
    pytest the whole package is already imported."""
    code = (f'import sys\n{statement}\n'
            "print(sorted(k for k in sys.modules if k.startswith('src.gradients')))")
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_the_production_energy_does_not_import_the_gradient_package():
    """E_0 is a forward energy, so it must be reachable without src.gradients.

    `exx_double_counting` used to live beside the nuclear derivatives that
    differentiate it, which made the ground-state energy of a Kohn-Sham
    reference unreachable without importing the whole gradient package. The
    control below shows the probe can see an import when there is one.
    """
    assert _modules_after(
        'import src.SingleReference.LinearResponse.rpa_energy') == '[]'
    assert _modules_after('import src.gradients.isdf_derivatives') != '[]'
