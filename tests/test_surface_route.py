"""The ONE entry point: every dispatch row, every refusal, and the numbers unchanged.

`potential_energy_surface` exists because a surface used to be a constructor
call with its physics implicit in it. Two relaxations of one molecule were 0.9
eV apart on an adiabatic energy because one E_0 was `mf.e_tot` and the other
`mf.e_tot + (E_x^HF - E_xc) + E_c^dRPA`; two incompatible quasiparticle windows
ran under one name; `solver` had three defaults; and an ISDF grid nobody
validated arrived at 148 points per atom whenever `counts` was omitted. So the
tests here are about what is DECLARED and what is REFUSED, and two of them
check that declaring it changed no number.

Water/cc-pVDZ throughout, RHF for the rpa rows and PBE0 for the dft one.

THE EQUIVALENCE GATES WERE SHOWN TO FAIL. Each was run once against a
deliberately broken copy of `src/properties/surfaces.py`, then the file was
restored from a backup and `cmp` confirmed it byte-identical:

  test_every_dispatch_row_builds_its_class
      the ('rpa', 'Excitation', 'space-time', 'isdf') row given the charged
      row's class -> FAILED: RPAQPSurface where RPABSESurface is declared.
  test_the_dispatcher_adds_nothing_to_the_excited_state_energy
      `build_excited_chain` handed `state=root` instead of `state=root - 1`
      -> FAILED: -75.99627708 Ha against the chain's -76.06219971 Ha, the
      dispatched surface sitting one root above it.
  test_the_default_grid_is_the_validated_level
      `resolve_grid` falling back to the 148-point counts instead of
      `SURFACE_GRID_ACCURACY` -> FAILED at A1 = 8 against G2's 16.
  test_the_default_quasiparticle_set_is_the_admitted_one
      `resolve_states` resolving `QPStates('frontier')` instead of the spec it
      was given -> FAILED: (3, 4, 5, 6) where the pole condition admits
      (2, 3, 4, 5, 6, 7, 8).
  test_comparing_two_ground_states_is_refused
      `comparable_with` replaced by agreement on the environment alone
      -> FAILED: DID NOT RAISE PhysicsMismatch, E_HF differenced against
      E_HF + E_c^dRPA.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.constants import BSE_DENSE_MAX_GB, ISDF_GRID_ACCURACY
from src.Base.declaration import (ChargedExcitation, Excitation, GroundState,
                                  PhysicsMismatch, QPStates)
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.gradients.dense_surfaces import (DenseBSESurface, DenseRPASurface,
                                          QuasiparticleSurface)
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface, RPAQPSurface
from src.gradients.rpa_ground_state import RPAGroundStateChain
from src.properties.optimize import MeanFieldSurface
from src.properties.surfaces import (compare_surfaces, potential_energy_surface,
                                     qp_set_degeneracy_tol)

#: The G2 counts water/cc-pVDZ must resolve to, read off the validated table
#: rather than respelled here.
G2_COUNTS = dict(zip(('A1', 'A2', 'A3', 'B1'), ISDF_GRID_ACCURACY['cc-pvdz']['G2']))

RPA_HF = GroundState('rpa', 'hf')
DFT_HF = GroundState('dft', 'hf')
DFT_PBE0 = GroundState('dft', 'pbe0')


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


def pbe0(mol):
    """The Kohn-Sham starting point of the dft row."""
    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    mf.grids.level = 5
    mf.kernel()
    return mf


#: (name, class, the keywords that select the row) for every row of the table.
#: The charged rows name the water HOMO, orbital 4 of 5 occupied.
ROWS = [
    ('rpa/none/space-time/isdf', RPAGroundStateChain, rhf,
     dict(ground_state=RPA_HF)),
    ('rpa/none/dense-qb/four-index', DenseRPASurface, rhf,
     dict(ground_state=RPA_HF, chi0='dense-qb', factorization='four-index')),
    ('rpa/none/dense-qb/df', DenseRPASurface, rhf,
     dict(ground_state=RPA_HF, chi0='dense-qb', factorization='df')),
    ('rpa/Excitation/space-time/isdf', RPABSESurface, rhf,
     dict(ground_state=RPA_HF, excitation=Excitation('singlet'))),
    ('rpa/Excitation/dense-qb', DenseBSESurface, rhf,
     dict(ground_state=RPA_HF, excitation=Excitation('singlet'),
          chi0='dense-qb', factorization='four-index')),
    ('rpa/ChargedExcitation/space-time/isdf', RPAQPSurface, rhf,
     dict(ground_state=RPA_HF, excitation=ChargedExcitation(4, -1))),
    ('rpa/ChargedExcitation/dense-qb', QuasiparticleSurface, rhf,
     dict(ground_state=RPA_HF, excitation=ChargedExcitation(4, -1),
          chi0='dense-qb', factorization='four-index')),
    ('dft/none', MeanFieldSurface, pbe0, dict(ground_state=DFT_PBE0)),
    ('dft/Excitation/space-time/isdf', ExcitedStateChain, pbe0,
     dict(ground_state=DFT_PBE0, excitation=Excitation('singlet'))),
]


@pytest.fixture(scope='module')
def mol():
    return water()


@pytest.mark.parametrize('name,expected,factory,kw',
                         ROWS, ids=[row[0] for row in ROWS])
def test_every_dispatch_row_builds_its_class(mol, name, expected, factory, kw):
    """Each row of the table returns the realizing class's OWN instance.

    Not a proxy: every reach-through into a chain -- `surface.excited`,
    `_forward`, `refreeze` -- has to keep working, which is what makes this
    entry point additive rather than a second way to do everything.
    """
    surface = potential_energy_surface(mol, factory, **kw)
    assert type(surface) is expected


def test_a_combination_with_no_class_lists_the_ones_that_have_one(mol):
    """A refusal that does not say what IS available sends the caller guessing."""
    with pytest.raises(ValueError) as exc:
        potential_energy_surface(mol, rhf, ground_state=DFT_HF,
                                 excitation=ChargedExcitation(4, -1))
    assert 'RPAQPSurface' in str(exc.value)
    assert 'MeanFieldSurface' in str(exc.value)


def test_the_dense_bse_surface_refuses_a_mean_field_ground_state(mol):
    """`GroundState('dft', xc)` has no dense realization, and is not approximated.

    The dense quasi-boson BSE surface puts E_HF + E_c^dRPA under every
    excitation, so answering a 'dft' declaration with it would hand back a
    surface 6.3 eV of E_c away from the one asked for.
    """
    with pytest.raises(ValueError, match='refused'):
        potential_energy_surface(mol, rhf, ground_state=DFT_HF,
                                 excitation=Excitation('singlet'),
                                 chi0='dense-qb', factorization='four-index')


def test_the_physics_survives_a_change_of_realization(mol):
    """One `SurfacePhysics` computed two ways is still one physics.

    The cubic space-time/ISDF route and the dense quasi-boson oracle share
    nothing numerically; that they declare the same functional and the same
    state is what makes their difference a route error rather than two
    different answers.
    """
    cubic = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                     excitation=Excitation('singlet'))
    dense = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                     excitation=Excitation('singlet'),
                                     chi0='dense-qb', factorization='four-index')
    assert cubic.physics == dense.physics
    assert cubic.realization != dense.realization


def test_comparing_two_ground_states_is_refused(mol):
    """E_HF and E_HF + E_c^dRPA are not differenceable, whatever else agrees."""
    mean_field = potential_energy_surface(mol, rhf, ground_state=DFT_HF)
    drpa = potential_energy_surface(mol, rhf, ground_state=RPA_HF)
    with pytest.raises(PhysicsMismatch):
        compare_surfaces(mean_field, drpa)


def test_comparing_a_singlet_with_a_triplet_is_what_a_gap_is(mol):
    """A differing EXCITATION is allowed; the realization differences are reported.

    The list is exactly the six fields these two routes disagree on, and every
    one of them is a real difference of method rather than of physics:

      chi0             imaginary-time chi0 against the dense quasi-boson block
      residues         the space-time route takes real-axis residues; the dense
                       one inverts the screening and has no residue backend
      factorization    ISDF against the exact (pq|rs)
      grid             the ISDF interpolation grid, which the dense route has not
      outside_treatment  the cubic chain gives every orbital outside the
                       quasiparticle set a frozen scissor calibrated on the
                       explicit roots; the dense quasi-boson route leaves
                       them at their mean-field eigenvalue
      realizing_class  which class computed it

    `solver`, `qp_states` and `qp_explicit` do NOT differ: both routes resolve
    the same declaration on the same reference spectrum, which is the point of
    resolving it in one place.
    """
    singlet = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('singlet'))
    triplet = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('triplet'),
                                       chi0='dense-qb', factorization='four-index')
    report = compare_surfaces(singlet, triplet)
    assert report['realization_differs'] == [
        'chi0', 'residues', 'factorization', 'grid', 'outside_treatment',
        'realizing_class']
    assert report['realization']['qp_explicit'][0] == \
        report['realization']['qp_explicit'][1]
    assert report['excitation'] == (Excitation('singlet'), Excitation('triplet'))


def test_two_realizations_of_one_route_differ_in_nothing(mol):
    """Same physics, same realization: an empty difference list.

    The gate on the one above -- a list that is never empty reports noise.
    """
    singlet = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('singlet'))
    triplet = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('triplet'))
    assert compare_surfaces(singlet, triplet)['realization_differs'] == []


def test_a_numeric_keyword_the_realization_does_not_read_is_refused(mol):
    """`ntau_rpa` is the dRPA energy's tau count and `ntau_gw` the self-energy's.

    They integrate different integrands, so a route that reads one and is
    handed the other has been given a setting the caller believes is in force.
    The accepted one is passed in the same test, or the refusal could be of
    everything.
    """
    with pytest.raises(TypeError) as exc:
        potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                 excitation=Excitation('singlet'), ntau_rpa=12)
    assert 'RPABSESurface' in str(exc.value)
    assert 'ntau_rpa' in str(exc.value)
    surface = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('singlet'),
                                       ntau_gw=20)
    assert surface.numerics['ntau_gw'] == 20


def test_the_pole_model_refuses_a_core_state_by_its_reach(mol):
    """`residues='sop'` on the oxygen 1s is refused, not fallen back from.

    A fallback would make the surface's cost and its error a function of the
    geometry. The reach is the worst swept pole in units of the particle-hole
    gap, so the message says by how much the state misses.
    """
    with pytest.raises(ValueError) as exc:
        potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                 excitation=ChargedExcitation(0, -1),
                                 residues='sop')
    assert 'orbital 0' in str(exc.value) and 'reach' in str(exc.value)
    # the HOMO is inside the wall, so the refusal is of the state and not of
    # the route
    potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                             excitation=ChargedExcitation(4, -1), residues='sop')


def test_a_route_with_no_adjoint_refuses_rather_than_differencing(mol):
    """evGW and the imaginary-frequency chi0 have no reverse pass.

    Each is paired with the route that does have one, so the refusal is of the
    missing adjoint and not of the surface.
    """
    with pytest.raises(NotImplementedError, match='adjoint'):
        potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                 excitation=Excitation('singlet', qp='evgw'))
    with pytest.raises(NotImplementedError, match='adjoint'):
        potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                 excitation=Excitation('singlet'),
                                 chi0='imagfrequency')
    potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                             excitation=Excitation('singlet', qp='g0w0'),
                             chi0='space-time')


def test_describe_prints_every_term_of_e_0(mol):
    """Three terms for the dRPA ground state, one for the mean field's own.

    The disease was two surfaces whose E_0 differed by E_c^dRPA with nothing
    printing either number, so `describe` evaluates the terms rather than
    naming them.
    """
    drpa = potential_energy_surface(mol, rhf, ground_state=RPA_HF).describe()
    for term in ('E_ref', 'E_x^HF - E_xc', 'E_c^dRPA'):
        assert term in drpa
    mean_field = potential_energy_surface(mol, pbe0,
                                          ground_state=DFT_PBE0).describe()
    assert 'E_ref' in mean_field
    assert 'E_c^dRPA' not in mean_field


def test_the_default_grid_is_the_validated_level(mol):
    """The default is G2's counts, never the 148-point silent fallback."""
    surface = potential_energy_surface(mol, rhf, ground_state=RPA_HF)
    assert surface.counts == G2_COUNTS
    assert 'G2' in surface.realization.grid


def test_a_basis_with_no_validated_level_is_refused(mol):
    """cc-pVQZ has no row at any accuracy, and nothing is substituted for it.

    A missing level was never measured, so the nearest one is a guess dressed
    as an answer; the refusal is `resolve_isdf_grid`'s own message. Paired with
    cc-pVDZ, which does have the row, so the refusal is of the basis.
    """
    qz = gto.M(atom=mol.atom, basis='cc-pvqz', verbose=0)
    with pytest.raises(ValueError, match='grid does not exist'):
        potential_energy_surface(qz, rhf, ground_state=RPA_HF)
    potential_energy_surface(mol, rhf, ground_state=RPA_HF)


def test_the_solver_is_resolved_and_recorded(mol):
    """Water's pair space fits the dense Casida pair, so 'auto' is 'dense'."""
    surface = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('singlet'))
    assert surface.realization.solver == 'dense'
    n_ov = 5 * (mol.nao - 5)
    assert 2 * n_ov ** 2 * 8 / 1e9 <= BSE_DENSE_MAX_GB


def test_the_default_quasiparticle_set_is_the_admitted_one(mol):
    """The resolved set is `resolve_qp_states`' own answer, not a second window.

    Two incompatible windows under one name is half of what this entry point
    cures, so the set is compared against a direct call on the same spectrum.
    """
    surface = potential_energy_surface(mol, rhf, ground_state=RPA_HF,
                                       excitation=Excitation('singlet'))
    mf = rhf(mol)
    direct = resolve_qp_states(QPStates(), mf.mo_energy, mol.nelectron // 2,
                               mol=mol,
                               degeneracy_tol=qp_set_degeneracy_tol({}))
    assert surface.realization.qp_explicit == direct.explicit
    assert tuple(np.asarray(surface.excited.qp_set)) == direct.explicit


def test_an_integer_window_still_gives_the_set_it_always_gave(mol):
    """The additive `qp_window` sequence must not move the int branch.

    `_qp_set` is the frozen convention every existing surface was built on, so
    a sequence being accepted has to leave a half-width bitwise where it was.
    """
    eps = np.asarray(rhf(mol).mo_energy, float)
    by_width = ExcitedStateChain._qp_set(eps, 5, 2, 1e-4)
    assert np.array_equal(by_width, np.arange(3, 7))
    by_sequence = ExcitedStateChain._qp_set(eps, 5, (6, 3, 5, 4), 1e-4)
    assert np.array_equal(by_sequence, by_width)
    assert np.array_equal(ExcitedStateChain._qp_set(eps, 5, 'all', 1e-4),
                          np.arange(len(eps)))


def test_the_dispatcher_adds_nothing_to_the_excited_state_energy(mol):
    """E_KS + Omega through the entry point is BITWISE the chain's own.

    The entry point declares and records; it may not move a number. Bitwise,
    not `approx`: a dispatcher that resolved one grid point or one tau
    differently would pass a tolerance and still be a different functional.
    """
    surface = potential_energy_surface(mol, pbe0, ground_state=DFT_PBE0,
                                       excitation=Excitation('singlet'))
    direct = ExcitedStateChain(
        mol, pbe0, spin='singlet', state=0, counts=G2_COUNTS, n_start=8,
        qp_window=surface.realization.qp_explicit, scissor='calibrate',
        outside='scissor', residue_route='explicit', solver='dense')
    assert surface.total_energy() == direct.total_energy()


def test_the_dispatcher_adds_nothing_to_the_drpa_ground_state(mol):
    """E_HF + E_c^dRPA through the entry point is BITWISE the chain's own."""
    surface = potential_energy_surface(mol, rhf, ground_state=RPA_HF)
    direct = RPAGroundStateChain(mol, rhf, counts=G2_COUNTS, n_start=8)
    assert surface.total_energy() == direct.total_energy()
