"""Gates for `outside=`: what the orbitals outside the quasiparticle set carry.

`ExcitedStateChain` solves the quasiparticle equation for the orbitals in its
set and puts those roots on the BSE diagonal. `outside='mean-field'` leaves
every other orbital at its eigenvalue, which is the class default and is what
the frozen baseline recorded; `outside='scissor'` gives each of them the frozen
shift `calibrate_scissor` reads off the explicit roots at the reference
geometry -- explicit inside the window, the scissor outside it -- and is what
`potential_energy_surface` asks the two cubic rows for.

WHAT THE SCISSOR IS WORTH. Water/cc-pVDZ, RHF/cc-pVDZ-ri, the default window
(qp_set 3-6, so orbitals 0-2 and 7-23 outside). `calibrate_scissor` gives every
outside orbital the correction of the explicit orbital nearest it in orbital
energy, which on this set is two tiers:

    occupied outside (0, 1, 2)     +0.9796 eV
    virtual outside (7 ... 23)     -0.3167 eV

and the roots move

    S1   8.456841 eV -> 8.454158 eV     -2.68 meV
    T1   7.705947 eV -> 7.699904 eV     -6.04 meV

THE SHIFT IS A CONSTANT, so d eps^QP_p/dR = d eps_p/dR outside the set and the
adjoint chain needs no term of its own. The finite-difference gate below is
what says so, at the step and the tolerance tests/test_excited_state.py uses:
h = 1e-4, four-point, worst deviation relative to the largest force < 1e-6.

THE EQUIVALENCE GATES WERE SHOWN TO FAIL. Each was run once against a
deliberately broken copy of `src/gradients/excited_state.py`, then the file was
restored from a backup and `cmp` confirmed it byte-identical:

  test_the_default_outside_is_bitwise_the_baseline[water-singlet]
      the scissor applied whatever `outside` says -> FAILED: R0 at
      -75.71716505145758 Ha against the record's -75.71706644131916 Ha, which
      is the 2.68 meV the scissor moves S1 by.
  test_with_every_orbital_explicit_the_scissor_is_the_mean_field
  test_the_outside_orbitals_carry_the_calibrated_shift
      one perturbation for both: `calibrate_scissor` handed every orbital
      instead of the outside ones, and its shift added to every eps^QP
      -> FAILED at -75.77691249101866 Ha against -75.71751677143605 Ha with
      the window at 'all', and on a chain map carrying four entries
      (3, 4, 5, 6) the direct calibration does not.
  test_the_scissor_gradient_follows_the_energy
      the freeze guard dropped, so every geometry recalibrates its own shift
      -> FAILED: worst 6.28e-05 Ha/Bohr on a gradient of 0.0981, 6.4e-04
      relative where the gate is 1e-06.
  test_refreeze_recalibrates_the_shift_at_the_new_geometry
      `refreeze` carrying the reference geometry's shifts onto the new chain
      -> FAILED: the refrozen chain arrived with {0: 0.035998229649215285,
      ...} already on it instead of None.
  test_the_entry_point_asks_for_the_scissor_and_records_it
      `excited_kwargs` recording the row's outside treatment without passing
      it (the perturbation is in src/properties/surfaces.py) -> FAILED:
      'mean-field' == 'scissor', the record and the chain disagreeing.
  test_an_unknown_outside_treatment_is_refused
      the vocabulary check removed -> FAILED: DID NOT RAISE ValueError.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import pathlib

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import HARTREE_TO_EV
from src.Base.declaration import Excitation, GroundState
from src.SingleReference.GW.qp_states import calibrate_scissor, frozen_scissor
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.rpa_bse_surface import RPABSESurface
from src.properties.surfaces import potential_energy_surface

#: wicks' own baseline_3f09ac0.json, copied verbatim: tools/refactor_gates/
#: was not ported, so the record lives beside the test that reads it.
GATES = pathlib.Path(__file__).resolve().parent

#: The two tiers the default window calibrates to on water, in eV, and the
#: move they put on the two lowest roots, in meV. Gated so the numbers in the
#: docstring are measured rather than remembered.
WATER_TIERS_EV = (0.9795617311565742, -0.31674210348796533)
WATER_MOVE_MEV = {'singlet': -2.683318564419586, 'triplet': -6.04250622349466}

#: The finite-difference step and tolerance tests/test_excited_state.py gates
#: its own gradients at. The same numbers, because a force with a frozen
#: scissor on the diagonal is the same chain with a different constant on it.
FD_STEP = 1e-4
FD_TOL = 1e-6


def newest_baseline():
    """The newest `baseline_<sha7>.json`, repeats excluded."""
    paths = [p for p in glob.glob(str(GATES / 'baseline_*.json'))
             if not p.endswith('_repeat.json')]
    if not paths:
        pytest.skip(f'no baseline_*.json in {GATES}; run record_baseline.py')
    return pathlib.Path(max(paths, key=os.path.getmtime))


RECORD = json.loads(newest_baseline().read_text())


def scf_factory(mol):
    """The baseline record's own mean field, to its own tolerances: the gate
    below differences THIS reference, so the construction is read off the
    record rather than respelled."""
    settings = RECORD['scf']
    mf = scf.RHF(mol).density_fit(auxbasis=settings['auxbasis'])
    mf.conv_tol = settings['conv_tol']
    mf.conv_tol_grad = settings['conv_tol_grad']
    mf.max_cycle = settings['max_cycle']
    mf.kernel()
    assert mf.converged
    return mf


def molecule(name):
    """One of the record's own molecules, at its own geometry and basis."""
    spec = RECORD['geometries'][name]
    return gto.M(atom=spec['atom'], basis=RECORD['basis'], verbose=0)


def at_coords(mol, coords):
    """`mol` moved onto the coordinates the record stored, in Bohr."""
    out = mol.copy()
    out.set_geom_(np.asarray(coords, float), unit='Bohr')
    return out


def outside_of(chain, norb):
    """The orbitals with no quasiparticle equation of their own."""
    inside = {int(p) for p in chain.qp_set}
    return [p for p in range(norb) if p not in inside]


@pytest.fixture(scope='module')
def water():
    """The record's water, with the mean field every chain below reads."""
    mol = molecule('water')
    return mol, scf_factory(mol)


# ----------------------------------------------------- (a) the default is legacy
@pytest.mark.parametrize('name,spin', [('water', 'singlet'),
                                       ('water', 'triplet'),
                                       ('formaldehyde', 'singlet'),
                                       ('formaldehyde', 'triplet')])
def test_the_default_outside_is_bitwise_the_baseline(name, spin):
    """`outside='mean-field'` is the class default and moves NO number.

    The whole point of putting the treatment behind a keyword is that a chain
    built the way every existing caller builds it computes what it computed
    before. Bitwise against the frozen record, at both of its geometries, with
    the reference geometry evaluated first -- the frozen conventions are
    decided there and the displaced geometry then spends them.
    """
    recorded = RECORD['surfaces'][name][f'ExcitedStateChain[{spin}]']
    assert recorded['status'] == 'ok'
    mol = molecule(name)
    chain = ExcitedStateChain(mol, scf_factory, spin=spin)
    assert chain.outside == 'mean-field'
    for label, entry in recorded['geometries'].items():
        at = mol if label == 'R0' else at_coords(mol, entry['atom_coords_bohr'])
        assert at.atom_coords().tolist() == entry['atom_coords_bohr'], label
        _, mf = chain.mean_field(at)
        assert float(mf.e_tot) == entry['e_scf'], label
        assert float(chain.total_energy(at, mf)) == entry['total_energy'], label
        grad, energy, _ = chain.total_gradient(at, mf)
        assert float(energy) == entry['gradient_energy'], label
        assert np.array_equal(np.asarray(grad, float),
                              np.asarray(entry['gradient'], float)), label
    assert chain.outside_shift is None


# -------------------------------------------- (b) nothing outside, nothing to do
def test_with_every_orbital_explicit_the_scissor_is_the_mean_field(water):
    """`qp_window='all'` leaves no orbital outside, so the two treatments are
    ONE surface -- bitwise, energy and gradient.

    The gate on the sign of everything below: a scissor that moved a number
    here would be shifting orbitals that carry an explicitly solved
    quasiparticle energy.
    """
    mol, mf = water
    mean = ExcitedStateChain(mol, scf_factory, mf=mf, qp_window='all')
    sciss = ExcitedStateChain(mol, scf_factory, mf=mf, qp_window='all',
                              outside='scissor')
    assert mean.total_energy(mol, mf) == sciss.total_energy(mol, mf)
    assert np.array_equal(mean.total_gradient(mol, mf)[0],
                          sciss.total_gradient(mol, mf)[0])
    assert sciss.outside_shift == {}


# ------------------------------------------------- (c) the shift, and what it moves
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_the_outside_orbitals_carry_the_calibrated_shift(water, spin):
    """eps^QP_p = eps_p + shift(p) outside the set, the explicit roots untouched.

    `shift` is `calibrate_scissor` on the SAME roots, called here directly:
    the chain may not invent a calibration of its own. The two tiers and the
    move they put on the root are in this module's docstring.
    """
    mol, mf = water
    mean = ExcitedStateChain(mol, scf_factory, mf=mf, spin=spin)
    sciss = ExcitedStateChain(mol, scf_factory, mf=mf, spin=spin,
                              outside='scissor')
    om_mean, pieces_mean = mean._forward(mol, mf)
    om_sciss, pieces_sciss = sciss._forward(mol, mf)
    eps, eps_qp_mean, eps_qp = pieces_mean[6], pieces_mean[7], pieces_sciss[7]
    qp_set = [int(p) for p in sciss.qp_set]
    roots = {p: float(eps_qp_mean[p]) for p in qp_set}
    outside = outside_of(sciss, len(eps))
    expect = calibrate_scissor(eps, sciss.nocc, roots, outside)
    assert sciss.outside_shift == expect
    for p in outside:
        assert eps_qp[p] == eps[p] + frozen_scissor(expect, p), p
    assert np.array_equal(eps_qp[qp_set], eps_qp_mean[qp_set])
    tiers = sorted({v for v in expect.values()}, reverse=True)
    assert np.allclose([t * HARTREE_TO_EV for t in tiers], WATER_TIERS_EV,
                       atol=1e-6)
    moved = (float(om_sciss[0]) - float(om_mean[0])) * HARTREE_TO_EV * 1e3
    assert moved == pytest.approx(WATER_MOVE_MEV[spin], abs=1e-3)


# ------------------------------------------------------- (d) the force follows it
def test_the_scissor_gradient_follows_the_energy(water):
    """dE_ex/dR with the frozen scissor, against a finite difference of E_ex.

    A frozen shift is a constant, so the adjoint chain carries no term for it
    and the gradient is the mean-field eigenvalue's. The finite difference is
    what says the energy agrees: it differences a surface whose outside
    orbitals carry the REFERENCE geometry's shift at every displaced geometry,
    which is what freezing means.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, outside='scissor')
    grad, _, diags = chain.total_gradient(mol, mf)
    assert diags['outside'] == 'scissor'
    outside = outside_of(chain, len(mf.mo_energy))
    assert len(diags['outside_shift_ev']) == len(outside)
    worst = 0.0
    for ia in range(mol.natm):
        for x in range(3):
            v = []
            for k in (-2, -1, 1, 2):
                step = np.zeros((mol.natm, 3))
                step[ia, x] = k * FD_STEP
                m = mol.copy()
                m.set_geom_(mol.atom_coords() + step, unit='Bohr')
                m.build(False, False)
                v.append(chain.energy(m)[0])
            fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * FD_STEP)
            worst = max(worst, abs(fd - grad[ia, x]))
    assert worst / np.abs(grad).max() < FD_TOL


# --------------------------------------------------------- (e) refreeze rebuilds it
def test_refreeze_recalibrates_the_shift_at_the_new_geometry(water):
    """A refrozen chain calibrates its OWN scissor, and records the drift.

    The shift is frozen against the geometries the surface is differenced over
    and rebuilt where the whole surface is: refreezing is how the optimizer
    measures the drift of the conventions it walked on, so the new shifts and
    the ones they replaced both have to be readable.
    """
    mol, mf = water
    recorded = RECORD['surfaces']['water']['ExcitedStateChain[singlet]']
    chain = ExcitedStateChain(mol, scf_factory, mf=mf, outside='scissor')
    chain.excitation(mol, mf)
    before = dict(chain.outside_shift)
    moved_to = at_coords(mol, recorded['geometries']['R0+0.05x_atom0']
                         ['atom_coords_bohr'])
    fresh = chain.refreeze(moved_to)
    assert fresh.outside == 'scissor'
    assert fresh.outside_shift is None
    fresh.excitation()
    assert set(fresh.outside_shift) == set(before)
    assert all(fresh.outside_shift[p] != before[p] for p in before)
    record = fresh.outside_record()
    assert record['outside_shift_before_ev'] == {
        p: v * HARTREE_TO_EV for p, v in sorted(before.items())}
    assert record['outside_shift_ev'] == {
        p: v * HARTREE_TO_EV for p, v in sorted(fresh.outside_shift.items())}
    assert record['outside_shift_moved_ev'] > 0.0


def test_the_composed_surface_forwards_and_refreezes_the_treatment(water):
    """`RPABSESurface` hands the treatment to its excited half, and keeps it.

    The dRPA ground state has no BSE diagonal, so what the orbitals outside
    the set carry is the excited half's alone -- but it is the SURFACE that an
    optimizer refreezes, and a treatment dropped there would leave the walk on
    one surface and its final fit on another.
    """
    mol, mf = water
    recorded = RECORD['surfaces']['water']['ExcitedStateChain[singlet]']
    default = RPABSESurface(mol, scf_factory, mf=mf)
    assert default.outside == 'mean-field'
    assert default.excited.outside == 'mean-field'
    surface = RPABSESurface(mol, scf_factory, mf=mf, outside='scissor')
    assert surface.outside == 'scissor'
    moved_to = at_coords(mol, recorded['geometries']['R0+0.05x_atom0']
                         ['atom_coords_bohr'])
    refrozen = surface.refreeze(moved_to)
    assert refrozen.outside == 'scissor'
    assert refrozen.excited.outside == 'scissor'


# ------------------------------------------------------------ (f) the entry point
def test_the_entry_point_asks_for_the_scissor_and_records_it(water):
    """`potential_energy_surface` declares the outside treatment and gets it.

    Bitwise against the chain built with the same arguments by hand: the entry
    point records what the row asked for, and the row's request is what the
    chain actually ran.
    """
    mol, _ = water
    surface = potential_energy_surface(mol, scf_factory,
                                       ground_state=GroundState('dft', 'hf'),
                                       excitation=Excitation('singlet'),
                                       chi0='space-time')
    assert surface.realization.outside_treatment == 'scissor'
    assert surface.outside == 'scissor'
    direct = ExcitedStateChain(
        mol, scf_factory, spin='singlet', state=0, counts=surface.counts,
        n_start=surface.n_start, qp_window=surface.realization.qp_explicit,
        scissor='calibrate', outside='scissor', residue_route='explicit',
        solver='dense')
    assert surface.total_energy() == direct.total_energy()


# ------------------------------------------------------------------ (g) refusals
def test_an_unknown_outside_treatment_is_refused(water):
    """A third answer to what the outside carries is a different surface, and
    the refusal names both of the two that exist."""
    mol, mf = water
    with pytest.raises(ValueError) as exc:
        ExcitedStateChain(mol, scf_factory, mf=mf, outside='nonsense')
    assert "'scissor'" in str(exc.value)
    assert "'mean-field'" in str(exc.value)


def test_the_scissor_needs_roots_to_calibrate_on(water):
    """`at_mean_field=True` solves no quasiparticle equation, so there is
    nothing to calibrate a shift from: the combination is refused rather than
    silently shifting by zero."""
    mol, mf = water
    with pytest.raises(ValueError, match='at_mean_field'):
        ExcitedStateChain(mol, scf_factory, mf=mf, outside='scissor',
                          at_mean_field=True)
