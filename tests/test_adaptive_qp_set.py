"""Gates for the adaptive explicit set on the chain (`qp_select=
QPStates('adaptive')`): the analytic continuation SELECTS which admitted
orbitals are solved and never supplies an energy or a force.

Formaldehyde/cc-pVDZ, HF, G1 ISDF factors, sum-over-poles residues, the
admitted window (11 candidates) and a 1 meV budget per target (S1 and T1):

  * the adaptive Omega sits within its recorded budget of the admitted one,
    and tol 0 is the admitted surface bit for bit;
  * the force is the derivative of the energy on a partition with holes;
  * the partition is frozen over a walk and shared by spin views, `refreeze`
    selects it again at its geometry, grow-only (a state explicit on the
    surface it refreezes stays explicit, even where a fresh selection drops
    it, the record names what each pass carried and added, and a set grown
    to the whole window is the admitted surface bit for bit), and a JSON
    round trip of it rebuilds the same surface;
  * THE GRADIENT NEVER READS AC: no continuation runs once the partition is
    frozen (a raising stand-in changes no bit), AC shifts offset by 0.05 eV
    give the same bits, and the reverse chain's modules do not import the
    selection.
"""
import ast
import json
import pathlib
import warnings

import numpy as np
import pytest
from pyscf import dft, gto, scf

import src.gradients.excited_state as excited_state
from src.Base.constants import (ADAPTIVE_HOLE_MAX_EV, ADAPTIVE_QP_TOL_MEV,
                                HARTREE_TO_EV)
from src.Base.declaration import Excitation, GroundState, QPStates
from src.Base.separable_ri import resolve_isdf_grid
from src.SingleReference.GW.qp_selection import AdaptivePartition
from src.SingleReference.GW.qp_states import resolve_qp_states
from src.gradients.excited_state import ExcitedStateChain
from src.gradients.state_manifold import StateManifold
from src.properties.surfaces import potential_energy_surface

REPO = pathlib.Path(__file__).resolve().parent.parent
ATOM = ('C 0 0 0; O 0 0 1.205; H 0 0.9429 -0.5876; H 0 -0.9429 -0.5876')
TARGETS = (('singlet', 0), ('triplet', 0))
#: the finite-difference gate of tests/test_outside_scissor.py, at a step
#: above the energy's noise floor: on the 32-point pole-model grid the worst
#: relative difference goes as 1/h, 3.4e-6 / 1.5e-6 / 9.4e-7 / 4.2e-7 at
#: h = 5e-5 / 1e-4 / 2e-4 / 4e-4 Bohr (64 points: 1.3e-6 / - / 5.3e-7 / -).
#: With the frozen frames turning with the molecule (`body_frame`) it is
#: 1.2e-6 / 1.1e-6 / 5.4e-7 at h = 2e-4 / 4e-4 / 8e-4, so the step is 8e-4,
#: at the 5e-7 the in-plane forces miss by at every h.
FD_STEP = 8e-4
FD_TOL = 1e-6
#: Below this many meV adaptive minus admitted is rounding, not the hole's
#: first-order error: under the hole cap formaldehyde's holes carry no target
#: weight (budget 6e-25 meV) and the two R0 solves, on different explicit
#: sets, differ by 3.6e-10 meV in S1.
BUDGET_FLOOR_MEV = 1e-6
#: Thioformaldehyde and its pi orbital, HOMO-1: no target weighs it at R0,
#: and without the hole cap it is a hole borrowing the n orbital's (HOMO)
#: shift, 0.59 eV off, which pi -> pi* T2 carries onto T1 along the C=S
#: stretch.
THIO_ATOM = ('C 0.0 0.0 -0.7936; S 0.0 0.0 0.8177; '
             'H 0.0 0.9243 -1.3828; H 0.0 -0.9243 -1.3828')
THIO_PI = 10
#: sum_p |n_p| of a single excitation: 2 in Tamm-Dancoff, a little more with
#: the de-excitations of the full BSE
SINGLE_EXCITATION_WEIGHT = 2.2
#: AC shifts moved uniformly by this many eV select the same partition; it
#: stays below ADAPTIVE_AC_CALIBRATION_MAX_EV, past which the frontier gate
#: rightly falls back to the whole window
AC_OFFSET_EV = 0.05


def factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


def pbe0_factory(mol):
    mf = dft.RKS(mol, xc='pbe0').density_fit(auxbasis='cc-pvdz-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def system():
    warnings.simplefilter('ignore')
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0)
    mf = factory(mol)
    eps = np.asarray(mf.mo_energy, float)
    nocc = mol.nelectron // 2
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', elements,
                                        auxbasis='cc-pvdz-ri')
    window = list(resolve_qp_states(QPStates('adaptive'), eps, nocc,
                                    degeneracy_tol=1e-4).explicit)
    displaced = mol.copy()
    displaced.set_geom_(mol.atom_coords() + 0.01 * np.array(
        [[0.3, -0.2, 0.1], [0.0, 0.4, -0.3], [0.2, 0.1, 0.0],
         [-0.1, 0.0, 0.2]]), unit='Bohr')
    return dict(mol=mol, mf=mf, counts=counts, n_start=n_start,
                window=window, displaced=displaced)


def chain(system, select=None, partition=None, spin='singlet'):
    return ExcitedStateChain(
        system['mol'], factory, spin=spin, state=0, qp_window=system['window'],
        residue_route='sop', scissor='calibrate', outside='scissor',
        solver='dense', counts=system['counts'], n_start=system['n_start'],
        mf=system['mf'], qp_select=select, qp_partition=partition)


def adaptive(tol=None):
    return QPStates('adaptive', tol_meV=tol, targets=TARGETS)


def evaluated(system, select=None, partition=None):
    """(the manifold's driven chain, its R0 evaluation over S1 and T1)."""
    man = StateManifold(chain(system, select, partition), states=TARGETS)
    ev = man.evaluate(system['mol'], system['mf'], gradients=TARGETS)
    return man, ev


@pytest.fixture(scope='module')
def admitted(system):
    return evaluated(system)


@pytest.fixture(scope='module')
def selected(system):
    return evaluated(system, adaptive())


def forces(man, ev):
    return {k: ev.gradient[k] for k in TARGETS}


def test_an_unstable_window_bse_selects_on_tamm_dancoff_weights(system):
    """PBE0: the continuation's LUMO shift is past ADAPTIVE_AC_SHIFT_MAX_EV,
    so the window BSE keeps that state's PBE0 energy and its full form is
    unstable (A - B not positive definite), where the production solve is
    not. The selection reads Tamm-Dancoff weights, says so, and adaptive
    stays within the tolerance of admitted (S1 0.02 meV; its first-order
    budget, 0.006 meV, is not a bound at that size)."""
    mol = system['mol']
    mf = pbe0_factory(mol)
    window = list(resolve_qp_states(
        QPStates('adaptive'), np.asarray(mf.mo_energy, float),
        mol.nelectron // 2, degeneracy_tol=1e-4).explicit)
    out = []
    for select in (None, adaptive()):
        ch = ExcitedStateChain(
            mol, pbe0_factory, spin='singlet', state=0, qp_window=window,
            residue_route='sop', scissor='calibrate', outside='scissor',
            solver='dense', counts=system['counts'],
            n_start=system['n_start'], mf=mf, qp_select=select)
        man = StateManifold(ch, states=TARGETS)
        out.append((man, man.evaluate(mol, mf)))
    (_, ev0), (man, ev) = out
    record = man.driven.selection_record()
    triplet = record['window_bse']['triplet']
    assert triplet['tda_fallback'] and triplet['weights'] == 'tda'
    assert triplet['not_positive_definite'] == 'A - B'
    assert record['verified']
    for row in record['targets']:
        key = (row['spin'], row['root'])
        actual = abs(ev.omega[key] - ev0.omega[key]) * HARTREE_TO_EV * 1e3
        assert row['budget_meV'] <= ADAPTIVE_QP_TOL_MEV, (key, row)
        assert actual <= ADAPTIVE_QP_TOL_MEV, (key, actual, row)


def test_the_selection_leaves_holes(selected, system):
    """The gates below differentiate a partition that has holes."""
    man, _ = selected
    part = man.driven.qp_partition
    assert part is not None and part.candidates == tuple(system['window'])
    assert 0 < len(part.holes) < len(part.candidates)
    record = man.driven.selection_record()
    assert record['verified'] and record['fell_back'] is None
    # Hartree-Fock's window BSE is stable: the full weights
    assert not any(v['tda_fallback'] for v in record['window_bse'].values())


def test_adaptive_is_within_its_budget_of_admitted(selected, admitted):
    (man, ev), (_, ev0) = selected, admitted
    record = man.driven.selection_record()
    for row in record['targets']:
        key = (row['spin'], row['root'])
        actual = abs(ev.omega[key] - ev0.omega[key]) * HARTREE_TO_EV * 1e3
        assert row['budget_meV'] <= ADAPTIVE_QP_TOL_MEV
        assert actual <= max(row['budget_meV'], BUDGET_FLOOR_MEV), (
            key, actual, row)


def test_tol_zero_is_bitwise_admitted(system, admitted):
    man, ev = evaluated(system, adaptive(tol=0.0))
    _, ev0 = admitted
    assert tuple(man.driven.qp_set) == tuple(admitted[0].driven.qp_set)
    assert man.driven.qp_partition.tier_of == {}
    for k in TARGETS:
        assert ev.omega[k] == ev0.omega[k]
        assert np.array_equal(ev.gradient[k], ev0.gradient[k])


def test_the_adaptive_force_is_the_derivative_of_its_energy(selected):
    man, ev = selected
    surface = man.surface(('singlet', 0))
    mol0 = man.driven.mol0
    analytic = ev.gradient[('singlet', 0)]
    fd = np.zeros_like(analytic)
    for ia in range(mol0.natm):
        for x in range(3):
            e = []
            for k in (-2, -1, 1, 2):
                m = mol0.copy()
                d = np.zeros((mol0.natm, 3))
                d[ia, x] = k * FD_STEP
                m.set_geom_(mol0.atom_coords() + d, unit='Bohr')
                m.build(False, False)
                e.append(surface.total_energy(m))
            fd[ia, x] = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * FD_STEP)
    worst = np.abs(analytic - fd).max() / np.abs(analytic).max()
    assert worst < FD_TOL, worst


def counting(monkeypatch, raising=False):
    """A list that grows by one per continuation the chain runs."""
    calls, real = [], excited_state.ac_quasiparticle_shifts

    def ac(*args, **kwargs):
        calls.append(1)
        if raising:
            raise AssertionError('a continuation ran after the freeze')
        return real(*args, **kwargs)
    monkeypatch.setattr(excited_state, 'ac_quasiparticle_shifts', ac)
    return calls


def test_no_ac_call_after_the_partition_is_frozen(system, monkeypatch):
    calls = counting(monkeypatch)
    ch = chain(system, adaptive())
    mol, mf = system['mol'], system['mf']
    ch.energy(mol, mf)
    assert len(calls) == 1
    m = system['displaced']
    e1 = ch.energy(m)
    g1, _ = ch.excitation_gradient(m)
    gt, _ = ch.spin_view('triplet').excitation_gradient(m)
    assert len(calls) == 1
    calls = counting(monkeypatch, raising=True)
    assert ch.energy(m) == e1
    assert np.array_equal(ch.excitation_gradient(m)[0], g1)
    assert np.array_equal(ch.spin_view('triplet').excitation_gradient(m)[0],
                          gt)
    assert not calls


def test_ac_values_never_reach_the_surface(system, selected, monkeypatch):
    """The same partition reached three ways -- selected, adopted with no
    continuation at all, and selected on AC shifts offset by 0.05 eV -- is
    one surface, bit for bit, at R0 and displaced."""
    man, ev = selected
    part = man.driven.qp_partition
    adopted = evaluated(system, partition=AdaptivePartition.from_record(
        json.loads(json.dumps(part.as_record()))))
    real = excited_state.ac_quasiparticle_shifts

    def offset(*args, **kwargs):
        a, z = real(*args, **kwargs)
        return {p: v + AC_OFFSET_EV / HARTREE_TO_EV for p, v in a.items()}, z
    monkeypatch.setattr(excited_state, 'ac_quasiparticle_shifts', offset)
    shifted = evaluated(system, adaptive())
    monkeypatch.undo()
    for other_man, other_ev in (adopted, shifted):
        got = other_man.driven.qp_partition
        assert got.explicit == part.explicit and got.tier_of == part.tier_of
        for k in TARGETS:
            assert other_ev.omega[k] == ev.omega[k]
            assert np.array_equal(other_ev.gradient[k], ev.gradient[k])
    m = system['displaced']
    ref = man.evaluate(m, gradients=TARGETS)
    for other_man, _ in (adopted, shifted):
        got = other_man.evaluate(m, gradients=TARGETS)
        for k in TARGETS:
            assert got.omega[k] == ref.omega[k]
            assert np.array_equal(got.gradient[k], ref.gradient[k])


def test_partition_is_frozen_over_the_walk_and_selected_again_by_refreeze(
        system, monkeypatch):
    """Within a walk the partition is frozen; `refreeze` selects it again by
    its own continuation at the new geometry, so the hole cap and the budget
    hold where the conventions were rebuilt. A partition handed in without a
    selection travels through `refreeze` as it is."""
    ch = chain(system, adaptive())
    ch.energy(system['mol'], system['mf'])
    part, shift0 = ch.qp_partition, dict(ch.outside_shift)
    m = system['displaced']
    ch.energy(m)
    assert ch.qp_partition is part and ch.outside_shift == shift0
    calls = counting(monkeypatch)
    fresh = ch.refreeze(m)
    fresh.energy(m)
    assert calls, 'refreeze carried the partition instead of selecting'
    again = fresh.qp_partition
    assert again is not part and again.candidates == part.candidates
    record = fresh.selection_record()
    assert record['verified']
    for p, d in record['holes_detail'].items():
        assert abs(d['delta_eV']) + d['u_eV'] <= ADAPTIVE_HOLE_MAX_EV, (p, d)
    assert tuple(int(p) for p in fresh.qp_set) == again.explicit
    assert fresh.outside_shift_before == shift0
    calls.clear()
    carried = chain(system, partition=part).refreeze(m)
    carried.energy(m)
    assert not calls and carried.qp_partition is part


def with_extra(part, extra):
    """`part` with the holes `extra` made explicit (their tiers dropped)."""
    extra = set(int(p) for p in extra)
    return AdaptivePartition(
        explicit=tuple(part.explicit) + tuple(sorted(extra)),
        tier_of={h: b for h, b in part.tier_of.items() if h not in extra},
        candidates=part.candidates, targets=part.targets,
        tol_meV=part.tol_meV)


def test_refreeze_is_grow_only(system, selected):
    """A refreeze pass may add states to the explicit set, never drop them.
    A hole of a fresh selection at the displaced geometry, made explicit on
    the surface refrozen there, stays explicit; so does everything explicit at the next pass; and the record
    names what each pass carried and added. Negative control: with the
    floor removed from `_select_qp_set` the carried hole is dropped."""
    m = system['displaced']
    part = selected[0].driven.qp_partition
    fresh = chain(system, adaptive()).refreeze(m)
    assert fresh.qp_floor is None, 'nothing was selected to carry'
    fresh.energy(m)
    dropped = [p for p in fresh.qp_partition.holes if p not in part.explicit]
    assert dropped, 'the gate needs a state a fresh selection drops'
    x = dropped[0]
    floor = with_extra(part, [x])
    first = chain(system, adaptive(), partition=floor).refreeze(m)
    assert first.qp_floor == floor.explicit
    first.energy(m)
    explicit1 = first.qp_partition.explicit
    assert set(floor.explicit) <= set(explicit1) and x in explicit1
    growth = first.qp_growth
    assert growth == {'carried': list(floor.explicit),
                      'added': sorted(set(explicit1) - set(floor.explicit)),
                      'not_explicit': []}
    record = first.selection_record()
    assert record['refreeze_growth'] == growth
    assert record['why'][x] == 'carried'
    assert record['verified']
    second = first.refreeze(system['mol'])
    assert second.qp_floor == explicit1
    second.energy(system['mol'])
    assert set(explicit1) <= set(second.qp_partition.explicit)
    assert second.qp_growth['carried'] == list(explicit1)


class DemoteOne:
    """`qp_set_gradient` reading a pole strength outside (0, 1] for one
    orbital, so its root is rejected (`qp_demoted`), as a LUMO+3 at Z
    1.04-1.05 is along a formaldehyde T1 walk."""

    def __init__(self, orbital):
        self.orbital, self.real = int(orbital), excited_state.qp_set_gradient

    def __call__(self, *args, **kw):
        out = self.real(*args, **kw)
        states = [int(p) for p in np.atleast_1d(args[7])]
        route_out = kw.get('route_out')
        if self.orbital in states and route_out and 'z' in route_out:
            route_out['z'] = np.array(route_out['z'], float)
            route_out['z'][states.index(self.orbital)] = -0.01
        return out


@pytest.mark.parametrize('demote', [None, 1], ids=['none', 'lumo+1'])
def test_a_set_grown_to_the_window_is_the_admitted_surface(system, demote,
                                                           monkeypatch):
    """The admitted window is the outer bound: a refreeze that carries the
    whole window solves the admitted set, and the surface is the admitted
    one refrozen at the same geometry, bit for bit, energy and force. Also
    where a root is rejected: the admitted window gives that orbital the
    shift of the explicit orbital nearest in energy, and so does the
    adaptive set (an AC-matched probe would put it 0.58 meV apart on
    formaldehyde T1)."""
    m = system['displaced']
    cands = tuple(system['window'])
    if demote is not None:
        orbital = system['mol'].nelectron // 2 + demote
        monkeypatch.setattr(excited_state, 'qp_set_gradient',
                            DemoteOne(orbital))
    whole = AdaptivePartition(explicit=cands, tier_of={}, candidates=cands,
                              targets=TARGETS, tol_meV=ADAPTIVE_QP_TOL_MEV)
    grown = chain(system, adaptive(), partition=whole).refreeze(m)
    man = StateManifold(grown, states=TARGETS)
    ev = man.evaluate(m, gradients=TARGETS)
    ref_man = StateManifold(chain(system).refreeze(m), states=TARGETS)
    ref = ref_man.evaluate(m, gradients=TARGETS)
    assert tuple(man.driven.qp_set) == tuple(ref_man.driven.qp_set)
    assert man.driven.qp_growth['added'] == []
    if demote is None:
        assert man.driven.qp_partition.tier_of == {}
    else:
        assert sorted(man.driven.qp_demoted) == [orbital]
        assert man.driven.qp_growth['not_explicit'] == [orbital]
        assert list(man.driven.qp_partition.tier_of) == [orbital]
        assert (man.driven.outside_shift[orbital]
                == ref_man.driven.outside_shift[orbital])
    for k in TARGETS:
        assert ev.omega[k] == ref.omega[k]
        assert ev.energy[k] == ref.energy[k]
        assert np.array_equal(ev.gradient[k], ref.gradient[k])


def test_spin_views_share_one_partition(selected):
    man, _ = selected
    view = man.driven.spin_view('triplet')
    assert view.qp_partition is man.driven.qp_partition
    assert np.array_equal(view.qp_set, man.driven.qp_set)


def test_adaptive_is_refused_where_it_cannot_scissor(system):
    with pytest.raises(ValueError):
        ExcitedStateChain(system['mol'], factory, qp_window=system['window'],
                          outside='mean-field', mf=system['mf'],
                          counts=system['counts'], n_start=system['n_start'],
                          qp_select=adaptive())


def import_closure(path, seen=None):
    """Every src module file `path` imports, recursively, by its own import
    statements (package __init__ side effects excluded)."""
    seen = set() if seen is None else seen
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith('src'):
            names = [node.module] + [f'{node.module}.{a.name}'
                                     for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names if a.name.startswith('src')]
        for name in names:
            target = REPO / (name.replace('.', '/') + '.py')
            if target.exists() and str(target) not in seen:
                seen.add(str(target))
                import_closure(target, seen)
    return seen


@pytest.mark.parametrize('module', [
    'src/gradients/qp_space_time.py',
    'src/SingleReference/LinearResponse/isdf_bse_adjoint.py',
    'src/gradients/space_time_adjoint.py',
    'src/gradients/bse_isdf.py'])
def test_reverse_chain_does_not_import_the_selection(module):
    closure = import_closure(REPO / module)
    assert closure, module
    assert not any(p.endswith('GW/qp_selection.py') for p in closure)


def test_the_entry_point_selects_and_refuses_the_reference(system):
    """`potential_energy_surface` hands the declaration to the cubic chain,
    whose qp_window is then the admitted candidates, and refuses it on the
    dense quasi-boson reference, which solves every state of its set."""
    mol, mf = system['mol'], system['mf']
    surface = potential_energy_surface(
        mol, factory, ground_state=GroundState('dft', 'hf'),
        excitation=Excitation('singlet'), residues='sop',
        qp_states=adaptive(), mf=mf, grid_accuracy='G1')
    assert surface.qp_select == adaptive()
    assert surface.realization.qp_explicit == tuple(system['window'])
    with pytest.raises(ValueError, match='reference'):
        potential_energy_surface(
            mol, factory, ground_state=GroundState('rpa', 'hf'),
            excitation=Excitation('singlet'), chi0='dense-qb',
            factorization='df', qp_states=adaptive(), mf=mf)


def test_the_hole_cap_solves_a_hole_no_target_reaches():
    """Thioformaldehyde, the R0 selection of S1 and T1 (n -> pi*): neither
    weighs the pi orbital, so the budget alone leaves it a hole 0.59 eV off.
    The cap solves it and records why; every hole left is within
    ADAPTIVE_HOLE_MAX_EV of its own root, so every root the check reads, not
    only the targets, has a first-order error within its weight times it."""
    mol = gto.M(atom=THIO_ATOM, basis='cc-pvdz', verbose=0)
    mf = factory(mol)
    eps = np.asarray(mf.mo_energy, float)
    nocc = mol.nelectron // 2
    assert THIO_PI == nocc - 2
    elements = sorted({mol.atom_pure_symbol(i) for i in range(mol.natm)})
    counts, n_start = resolve_isdf_grid('G1', 'cc-pvdz', elements,
                                        auxbasis='cc-pvdz-ri')
    window = list(resolve_qp_states(QPStates('adaptive'), eps, nocc,
                                    degeneracy_tol=1e-4).explicit)
    assert THIO_PI in window
    ch = ExcitedStateChain(
        mol, factory, spin='singlet', state=0, qp_window=window,
        residue_route='sop', scissor='calibrate', outside='scissor',
        solver='dense', counts=counts, n_start=n_start, mf=mf,
        qp_select=adaptive())
    man = StateManifold(ch, states=TARGETS)
    man.evaluate(mol, mf)
    part = man.driven.qp_partition
    record = man.driven.selection_record()
    assert THIO_PI in part.explicit and THIO_PI not in part.holes
    assert THIO_PI in record['hole_cap']['kept_explicit']
    assert record['why'][THIO_PI]['reason'] == 'hole_cap'
    assert record['why'][THIO_PI]['error_meV'] > 1e3 * ADAPTIVE_HOLE_MAX_EV
    assert record['hole_cap']['max_eV'] == ADAPTIVE_HOLE_MAX_EV
    assert record['verified']
    for p, d in record['holes_detail'].items():
        assert abs(d['delta_eV']) + d['u_eV'] <= ADAPTIVE_HOLE_MAX_EV, (p, d)
    largest = record['hole_cap']['largest_hole_meV']
    assert largest is None or largest <= 1e3 * ADAPTIVE_HOLE_MAX_EV
    bound = SINGLE_EXCITATION_WEIGHT * 1e3 * ADAPTIVE_HOLE_MAX_EV
    rows = record['targets'] + [r for v in record['other_roots'].values()
                                for r in v]
    assert len(rows) > len(TARGETS)
    for row in rows:
        assert row['budget_meV'] <= bound, row
