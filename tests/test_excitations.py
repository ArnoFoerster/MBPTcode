"""The vertical, emission and adiabatic energies of one state, and the
refusals that keep two of them subtractable.

WHAT THIS CURES. An adiabatic energy is a difference of two total energies at
two different geometries, and the number carries no record of which two
surfaces produced it: E_KS + Omega was subtracted from E_HF + (E_x^HF - E_xc)
+ E_c^dRPA + Omega and nothing could refuse it. Here one `SurfaceSpec` builds
BOTH surfaces of the difference, so the ground state under the excited surface
and the ground state under E_0 cannot be two functionals, and two specs that do
declare different functionals are refused by name before an integral is
computed.

Water/cc-pVDZ on RHF throughout, the singlet S1 on two routes: the dense
quasi-boson oracle (chi0='dense-qb', four-index) and the cubic space-time/ISDF
production route with every orbital an explicit quasiparticle
(QPStates('all')).

THE STATE IS DISSOCIATIVE AND THAT IS NOT WHAT IS BEING TESTED. Water's S1
lengthens both O-H bonds without limit at the default trust radius (see
`test_optimizer_reports_a_state_it_cannot_follow` in test_properties.py). At a
0.05 Bohr radius the dense route reaches a stationary point on the shoulder at
r(OH) = 1.256 Angstrom and converges there in 28 cycles, which is what makes a
MEASURED refreeze shift available at all; the cubic route is capped at ten
cycles and its record says `converged: False` with the cycle limit as its
status. What is gated below is what the records carry and what they refuse --
not the physics of that minimum, which no assertion here depends on.

THE EQUIVALENCE GATES WERE SHOWN TO FAIL. Each was run once against a
deliberately broken copy of the file named, which was then restored from a
backup and `cmp` confirmed byte-identical:

  test_the_emission_energy_is_the_gap_at_the_relaxed_geometry
      src/properties/excitations.py: `emission_block` taking E_0 at R0 rather
      than at R*_n -> FAILED: e0_hartree_at_excited_minimum came back
      -76.25806819101113, the ground-state energy at R0 to the last digit, and
      the emission energy with it -- 7.34 eV where the gap at the relaxed
      geometry is 2.21 eV.
  test_the_routine_adds_nothing_to_the_excitation_energy
      `excitation_energy` rebuilding Omega as E_n - E_0 -> FAILED:
      8.428662087951224 eV against the chain's own 8.428662087951114 eV, the
      same quantity to 1.1e-13 eV and a different float.
  test_no_field_of_a_record_is_called_grad_max
      `relaxation_fields` emitting the residual under `grad_max` as well ->
      FAILED: record['grad_max'].
  test_a_record_measures_the_refreeze_or_says_it_did_not[dense]
      `relaxation_fields` dropping the optimizer's `refreeze_denergy` ->
      FAILED: refreeze_shift_meV None on a record whose outer loop ran and
      whose marker says 'measured'.
  test_the_two_routes_declare_one_physics_and_differ_only_in_realization
      `vertical_block` recording the GROUND surface's realization -> FAILED:
      ['chi0', 'factorization', 'grid', 'realizing_class'], four fields short
      of what the two excited surfaces differ in.
  test_a_mean_field_ground_state_is_refused_against_a_drpa_one
      `calc_adiabatic_gap` with BOTH refusals removed -- the one on the
      declarations and `compare_surfaces` on the built surfaces, since either
      alone still refuses -> FAILED: DID NOT RAISE PhysicsMismatch.
  test_vibronic_refuses_two_relaxations_of_different_functionals
      src/properties/vibronic.py: the raise in `adiabatic_gap` removed ->
      FAILED: DID NOT RAISE PhysicsMismatch.

Every check ASSERTS: pytest discards a returned verdict and passes on False.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import fields

import pytest
from pyscf import gto, scf

import src.gradients  # noqa: F401  cycle: src.properties imports src.gradients
from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import (ChargedExcitation, Excitation, GroundState,
                                  PhysicsMismatch, QPStates)
from src.properties.excitations import (SurfaceSpec, calc_adiabatic_excitation,
                                        calc_adiabatic_gap,
                                        calc_vertical_excitation, surface_of)
from src.properties.surfaces import Realization, compare_surfaces
from src.properties.vibronic import adiabatic_gap

SINGLET = Excitation('singlet')
TRIPLET = Excitation('triplet')

#: The dense quasi-boson oracle: the exact (pq|rs), no interpolation grid, no
#: residue backend, E_0 = E_HF + E_c^dRPA.
DENSE = SurfaceSpec(GroundState('rpa', 'hf'), chi0='dense-qb',
                    factorization='four-index')

#: The cubic production route on the same physics, with every orbital carrying
#: an explicitly solved quasiparticle energy rather than the admitted set.
ISDF = SurfaceSpec(GroundState('rpa', 'hf'), chi0='space-time',
                   factorization='isdf', qp_states=QPStates('all'))

#: The mean field's own ground state -- a DIFFERENT functional, which is the
#: whole point of the refusal it appears in.
MEAN_FIELD = SurfaceSpec(GroundState('dft', 'hf'))

#: Water's S1 walks off its shoulder at the optimizer's default 0.1 Bohr trust
#: radius and dissociates; half that keeps the dense route on the shoulder,
#: where it converges. A radius is a step control and not a convergence
#: criterion: every record below is still converged against `GEOM_OPT_CONV`.
#: The engine is named rather than left to 'auto' because a trust radius is the
#: CARTESIAN optimizer's control -- geomeTRIC takes `converge` and `coordsys`
#: instead -- so a machine with geomeTRIC on its path would otherwise be
#: running a different optimization from the one these numbers came off. Both
#: engines write the same record fields, which test_optimize_refreeze gates.
LOOSE = dict(engine='cartesian', trust=0.05, trust_max=0.1, verbose=False)


def water():
    """The one geometry every surface below is built at."""
    return gto.M(atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
                 basis='cc-pvdz', verbose=0)


def rhf(mol):
    """A reference converged tightly enough to differentiate."""
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.kernel()
    return mf


def every_key(record, path='record'):
    """Every key of every mapping in a record, with the path that reaches it."""
    out = []
    if isinstance(record, dict):
        for key, value in record.items():
            out.append((f'{path}[{key!r}]', key))
            out += every_key(value, f'{path}[{key!r}]')
    elif isinstance(record, (list, tuple)):
        for index, value in enumerate(record):
            out += every_key(value, f'{path}[{index}]')
    return out


@pytest.fixture(scope='module')
def mol():
    return water()


@pytest.fixture(scope='module')
def dense_record(mol):
    """S1 through the dense oracle, with the refreeze drift MEASURED.

    The dense surface freezes nothing -- no interpolation layout, no frames --
    so its measured drift is a true zero rather than an unmeasured one, and
    telling those two apart is exactly what the marker is for.
    """
    return calc_adiabatic_excitation(DENSE, SINGLET, mol, rhf, refreeze=1,
                                     max_cycle=40, **LOOSE)


@pytest.fixture(scope='module')
def isdf_record(mol):
    """S1 through the cubic route, capped and with refreeze=0.

    Every gradient here solves all 24 quasiparticles and costs ~6 s; the walk
    is dissociative, so it is stopped at ten cycles and the refreeze outer loop
    -- which is a second full optimization -- is not asked for. The record says
    both things about itself.
    """
    return calc_adiabatic_excitation(ISDF, SINGLET, mol, rhf, refreeze=0,
                                     max_cycle=10, **LOOSE)


def test_the_emission_energy_is_the_gap_at_the_relaxed_geometry(dense_record):
    """Omega(R*_n) = E_n(R*_n) - E_0(R*_n), with E_0 at the EXCITED minimum.

    The ground state is not at its own minimum there, so E_0(R*_n) is not
    E_0(R0) and taking the second would report the vertical gap of a relaxed
    state -- 7.34 eV here instead of 2.21 eV, the Stokes shift of the whole
    relaxation lost. Both energies come off the same pair of frozen
    conventions, the reference geometry's, so their difference is one surface
    pair evaluated twice.
    """
    record = dense_record
    assert record['emission_eV'] == pytest.approx(
        (record['en_hartree_at_excited_minimum']
         - record['e0_hartree_at_excited_minimum']) * HARTREE_TO_EV, rel=1e-12)
    assert record['e0_hartree_at_excited_minimum'] > record['e0_hartree']
    assert record['emission_eV'] < record['omega_eV']
    assert record['relaxation_depth_eV'] == pytest.approx(
        (record['en_hartree'] - record['en_hartree_at_excited_minimum'])
        * HARTREE_TO_EV, rel=1e-12)


def test_the_adiabatic_energy_takes_each_state_to_its_own_minimum(dense_record):
    """E_n(R*_n) - E_0(R*_0): two minima, so E_0 does not cancel.

    It sits between the emission energy (where only the excited state relaxed)
    and the vertical one (where neither did), because each relaxation can only
    lower its own side.
    """
    record = dense_record
    assert record['adiabatic_eV'] == pytest.approx(
        (record['en_hartree_at_excited_minimum']
         - record['e0_hartree_at_ground_minimum']) * HARTREE_TO_EV, rel=1e-12)
    assert record['emission_eV'] < record['adiabatic_eV'] < record['omega_eV']
    assert record['e0_hartree_at_ground_minimum'] < record['e0_hartree']


@pytest.mark.parametrize('name', ('dense', 'isdf'))
def test_a_record_measures_the_refreeze_or_says_it_did_not(name, dense_record,
                                                           isdf_record):
    """A null shift only ever comes from a refreeze nobody asked for.

    The dense record asked for one and carries both measurements -- Bohr and
    meV; the cubic record passed refreeze=0 and carries the marker instead. An
    unmeasured drift that reads like a measured zero is what the marker exists
    to prevent, and both are in the same test so that neither is vacuous.
    """
    record = {'dense': dense_record, 'isdf': isdf_record}[name]
    if name == 'dense':
        assert record['refreeze'] == 'measured'
        assert isinstance(record['refreeze_shift_meV'], float)
        assert isinstance(record['refreeze_shift_bohr'], float)
    else:
        assert record['refreeze'] == 'not measured'
        assert record['refreeze_shift_meV'] is None
    assert isinstance(record['ground_opt_grad_max'], float)
    assert isinstance(record['opt_grad_max'], float)
    assert isinstance(record['driving_force_max'], float)
    # the residual at R* is not the driving force at R0, and neither is a
    # rounding of the other
    assert record['driving_force_max'] > record['opt_grad_max']


@pytest.mark.parametrize('name', ('dense', 'isdf'))
def test_e_0_is_reported_in_its_three_terms_at_every_geometry(name, dense_record,
                                                              isdf_record):
    """E_0 = E_ref + (E_x^HF - E_xc) + E_c^dRPA, printed at all three geometries.

    An E_0 given as one number cannot be checked against another route's: the
    terms are what say which functional the number belongs to. The
    double-counting term is identically zero on a Hartree-Fock reference and is
    carried anyway, because a surface without it is a different functional.
    """
    record = {'dense': dense_record, 'isdf': isdf_record}[name]
    for key in ('e0_terms', 'e0_terms_at_excited_minimum',
                'e0_terms_at_ground_minimum'):
        terms = record[key]
        assert list(terms) == ['E_ref', 'E_x^HF - E_xc', 'E_c^dRPA']
        assert terms['E_x^HF - E_xc'] == 0.0
        assert terms['E_c^dRPA'] < 0.0
    assert record['e0_hartree'] == pytest.approx(sum(record['e0_terms'].values()),
                                                 rel=1e-14)


def test_the_two_routes_declare_one_physics_and_differ_only_in_realization(
        mol, dense_record, isdf_record):
    """One spec's physics, two realizations, and the list of what differs.

    The two routes share nothing numerically -- 8.4534 eV against 8.4291 eV
    vertically -- and that is a route error and not two different quantities
    only because both declare the same functional, the same state and the same
    environment. Every field in the list is a difference of method:

      chi0             the dense quasi-boson block against the imaginary-time
                       polarizability
      residues         the cubic route takes real-axis residues of Sigma^c; the
                       dense one inverts the screening and has no backend
      factorization    the exact (pq|rs) against the ISDF fit
      grid             the interpolation grid, which the dense route has not
      qp_states        the admitted set against QPStates('all')
      qp_explicit      the orbitals those two declarations come out as
      outside_treatment  what the orbitals outside the set carry: the frozen
                       scissor on the cubic route, the bare eigenvalue on the
                       dense one
      realizing_class  which class computed it

    `solver` is NOT in the list: both routes run a dense Casida solve, and one
    resolution of 'auto' is what keeps that from being two different answers.
    """
    assert dense_record['physics'] == isdf_record['physics']
    differs = [field.name for field in fields(Realization)
               if getattr(dense_record['realization'], field.name)
               != getattr(isdf_record['realization'], field.name)]
    assert differs == ['chi0', 'residues', 'factorization', 'grid',
                       'qp_states', 'qp_explicit', 'outside_treatment',
                       'realizing_class']
    report = compare_surfaces(surface_of(DENSE, SINGLET, mol, rhf),
                              surface_of(ISDF, SINGLET, mol, rhf))
    assert report['realization_differs'] == differs


@pytest.mark.parametrize('name', ('dense', 'isdf'))
def test_no_field_of_a_record_is_called_grad_max(name, dense_record,
                                                 isdf_record):
    """`grad_max` means the driving force at R0 in half the repository and the
    residual at R* in the other half.

    An adiabatic record must not inherit that: it reports both numbers, so a
    name that means either would make the pair unreadable. The whole record is
    walked, sub-records included.
    """
    record = {'dense': dense_record, 'isdf': isdf_record}[name]
    named = [path for path, key in every_key(record) if key == 'grad_max']
    assert named == []
    assert [path for path, key in every_key(record)
            if key == 'opt_grad_max']


def test_a_mean_field_ground_state_is_refused_against_a_drpa_one(mol):
    """THE TEST THAT WOULD HAVE REFUSED THE TABLE.

    E_HF + Omega and E_HF + (E_x^HF - E_xc)[rho] + E_c^dRPA + Omega are 6.3 eV
    of correlation energy apart on water, and it does not cancel out of a gap
    that only one side carries. The refusal names both functionals and is taken
    on the DECLARATIONS, so it costs no integral.
    """
    with pytest.raises(PhysicsMismatch) as exc:
        calc_adiabatic_gap(MEAN_FIELD, SINGLET, DENSE, TRIPLET, mol, rhf)
    assert GroundState('dft', 'hf').label() in str(exc.value)
    assert GroundState('rpa', 'hf').label() in str(exc.value)


def test_a_singlet_and_a_triplet_on_one_spec_are_what_a_gap_is(mol):
    """The accepted half of the refusal above: one functional, two states.

    Deliberately short walks -- the triplet dissociates and neither state is at
    a minimum after two cycles -- because what is gated here is that the pair
    is ACCEPTED and reported, not the gap itself. Both records come back whole
    and the realization list is empty, since both states are on one route.
    """
    gap = calc_adiabatic_gap(DENSE, SINGLET, DENSE, TRIPLET, mol, rhf,
                             refreeze=0, max_cycle=2, **LOOSE)
    assert gap['realization_differs'] == []
    assert gap['physics']['record'][0].excitation == SINGLET
    assert gap['physics']['record'][1].excitation == TRIPLET
    assert gap['gap_eV'] == pytest.approx(
        (gap['state_a']['en_hartree_at_excited_minimum']
         - gap['state_b']['en_hartree_at_excited_minimum']) * HARTREE_TO_EV,
        rel=1e-12)
    assert gap['gap_eV'] > 0.0


def test_the_routine_adds_nothing_to_the_excitation_energy(mol):
    """Omega in the record is BITWISE the chain's own.

    This layer records and differences; it may not move a number by so much as
    a rounding. Bitwise and not `approx`, and against the chain's own accessor
    rather than against E_n - E_0: the difference of two -76 Hartree totals is
    the same quantity to 1.1e-13 eV and is NOT the same float, which is why the
    record takes Omega off the surface instead of rebuilding it.
    """
    record = calc_vertical_excitation(MEAN_FIELD, SINGLET, mol, rhf)
    chain = surface_of(MEAN_FIELD, SINGLET, mol, rhf)
    assert record['omega_eV'] == chain.excitation() * HARTREE_TO_EV
    assert record['omega_eV'] != (record['en_hartree']
                                  - record['e0_hartree']) * HARTREE_TO_EV
    assert record['omega_eV'] == pytest.approx(
        (record['en_hartree'] - record['e0_hartree']) * HARTREE_TO_EV,
        rel=1e-12)
    assert list(record['e0_terms']) == ['E_ref']


def test_a_charged_state_carries_its_pole_strength_and_its_route(mol):
    """Z and the residue backend, where the surface's gradient exposes them.

    A quasiparticle root with Z near zero is a satellite and not the state
    asked for, and which backend produced the residues is a choice
    `residues='auto'` makes per orbital -- so both belong in the record of any
    surface that reports them. A neutral excitation folds its whole set through
    one solve and reports neither; None there is not a Z of zero.
    """
    ionization = calc_vertical_excitation(
        SurfaceSpec(GroundState('rpa', 'hf')), ChargedExcitation(4, -1), mol,
        rhf)
    assert 0.5 < ionization['qp_z'] < 1.0
    assert ionization['residue_route_taken'] == 'explicit'
    assert ionization['omega_eV'] > 0.0


def test_vibronic_refuses_two_relaxations_of_different_functionals():
    """`adiabatic_gap` on two records refuses what it can check and nothing else.

    Two records that carry a declaration are refused when the declarations are
    not subtractable; two that carry none are differenced exactly as before,
    because a hand-built surface or a toy has nothing to check and refusing it
    would break every caller that never had one.
    """
    rpa = SurfaceSpec(GroundState('rpa', 'hf')).physics(SINGLET)
    mean_field = MEAN_FIELD.physics(SINGLET)
    with pytest.raises(PhysicsMismatch):
        adiabatic_gap({'e_total': -75.9, 'physics': rpa},
                      {'e_total': -76.2, 'physics': mean_field})
    assert adiabatic_gap({'e_total': -75.9, 'physics': rpa},
                         {'e_total': -76.2, 'physics': rpa}) == \
        pytest.approx(0.3, abs=1e-12)
    assert adiabatic_gap({'e_total': -113.72080728},
                         {'e_total': -113.74816714}) == \
        pytest.approx(0.02735986, abs=1e-8)
