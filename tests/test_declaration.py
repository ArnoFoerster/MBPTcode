"""Gates for src/Base/declaration.py: the vocabulary that separates WHAT a
surface computes from HOW it is realized.

If GroundState/Excitation/ChargedExcitation/QPStates stop validating their
vocabulary, or start comparing unequal by value, energy and gradient code
downstream can silently mix incompatible functionals or states -- exactly the
defect this module exists to make impossible. If SurfacePhysics.label() drifts
from the strings fixed here, every printed energy breakdown drifts with it. If
comparable_with ever returns True for two different ground states or
environments, a gap gets taken between two numbers that are not the same
physics. If the module starts importing from `src`, gradient code that must
not depend on production numerics gains a hidden dependency on it.
"""
import ast
import inspect

import pytest

from src.Base.declaration import (ChargedExcitation, Excitation,
                                   GroundState, PhysicsMismatch, QPStates,
                                   SurfacePhysics)


# ---------------------------------------------------------------------------
# dependency isolation
# ---------------------------------------------------------------------------

def test_module_imports_only_dataclasses_and_typing():
    """No import in declaration.py names anything outside dataclasses/typing."""
    import src.Base.declaration as declaration_module
    source = inspect.getsource(declaration_module)
    tree = ast.parse(source)
    allowed = {'dataclasses', 'typing'}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                seen.add(node.module.split('.')[0])
    assert seen, 'expected at least the dataclasses/typing imports to be found'
    assert seen <= allowed, f'declaration.py imports outside {allowed}: {seen - allowed}'


# ---------------------------------------------------------------------------
# construction with defaults
# ---------------------------------------------------------------------------

def test_ground_state_constructs():
    gs = GroundState(kind='rpa', xc='pbe0')
    assert gs.kind == 'rpa'
    assert gs.xc == 'pbe0'


def test_excitation_constructs_with_defaults():
    exc = Excitation(spin='singlet')
    assert exc.root == 1
    assert exc.irrep is None
    assert exc.kernel == 'bse'
    assert exc.qp == 'g0w0'
    assert exc.screening == 'rpa'


def test_charged_excitation_constructs():
    ce = ChargedExcitation(orbital=0, charge_change=-1)
    assert ce.orbital == 0
    assert ce.charge_change == -1


def test_qp_states_constructs_with_defaults():
    qp = QPStates()
    assert qp.kind == 'admitted'
    assert qp.threshold == 'gap'
    assert qp.half_width == 2
    assert qp.extra_virtuals == 10
    assert qp.filter_z is False


def test_surface_physics_constructs_with_default_environment():
    sp = SurfacePhysics(ground_state=GroundState(kind='dft', xc='pbe0'), excitation=None)
    assert sp.environment == 'gas'


# ---------------------------------------------------------------------------
# xc is stored lower-case
# ---------------------------------------------------------------------------

def test_ground_state_xc_is_lowered():
    gs = GroundState(kind='dft', xc='CAM-B3LYP')
    assert gs.xc == 'cam-b3lyp'


# ---------------------------------------------------------------------------
# equality and hashing
# ---------------------------------------------------------------------------

def test_ground_state_equal_constructions_hash_equal():
    a = GroundState(kind='rpa', xc='pbe0')
    b = GroundState(kind='rpa', xc='pbe0')
    assert a == b
    assert hash(a) == hash(b)


def test_ground_state_differing_field_breaks_equality():
    a = GroundState(kind='rpa', xc='pbe0')
    b = GroundState(kind='dft', xc='pbe0')
    assert a != b


def test_excitation_equal_constructions_hash_equal():
    a = Excitation(spin='singlet', root=2, kernel='bse-tda')
    b = Excitation(spin='singlet', root=2, kernel='bse-tda')
    assert a == b
    assert hash(a) == hash(b)


def test_excitation_differing_field_breaks_equality():
    a = Excitation(spin='singlet', root=1)
    b = Excitation(spin='singlet', root=2)
    assert a != b


def test_charged_excitation_equal_constructions_hash_equal():
    a = ChargedExcitation(orbital=3, charge_change=1)
    b = ChargedExcitation(orbital=3, charge_change=1)
    assert a == b
    assert hash(a) == hash(b)


def test_charged_excitation_differing_field_breaks_equality():
    a = ChargedExcitation(orbital=3, charge_change=1)
    b = ChargedExcitation(orbital=4, charge_change=1)
    assert a != b


def test_qp_states_equal_constructions_hash_equal():
    a = QPStates(kind='valence', extra_virtuals=5)
    b = QPStates(kind='valence', extra_virtuals=5)
    assert a == b
    assert hash(a) == hash(b)


def test_qp_states_differing_field_breaks_equality():
    a = QPStates(kind='valence', extra_virtuals=5)
    b = QPStates(kind='valence', extra_virtuals=6)
    assert a != b


def test_surface_physics_equal_constructions_hash_equal():
    a = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'),
                        excitation=Excitation(spin='triplet'))
    b = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'),
                        excitation=Excitation(spin='triplet'))
    assert a == b
    assert hash(a) == hash(b)


def test_surface_physics_differing_field_breaks_equality():
    a = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'), excitation=None)
    b = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'), excitation=None,
                        environment='pcm(water)')
    assert a != b


# ---------------------------------------------------------------------------
# GroundState.terms() and .label()
# ---------------------------------------------------------------------------

def test_ground_state_terms_dft():
    assert GroundState(kind='dft', xc='pbe0').terms() == ('E_ref',)


def test_ground_state_terms_rpa():
    assert GroundState(kind='rpa', xc='pbe0').terms() == ('E_ref', 'E_x^HF - E_xc', 'E_c^dRPA')


def test_ground_state_label_rpa_pbe0():
    gs = GroundState(kind='rpa', xc='pbe0')
    assert gs.label() == 'E_PBE0 + (E_x^HF - E_xc)[rho] + E_c^dRPA'


def test_ground_state_label_rpa_hf():
    gs = GroundState(kind='rpa', xc='hf')
    assert gs.label() == 'E_HF + (E_x^HF - E_xc)[rho] + E_c^dRPA'


def test_ground_state_label_dft_pbe0():
    gs = GroundState(kind='dft', xc='pbe0')
    assert gs.label() == 'E_PBE0'


def test_ground_state_label_dft_hf():
    gs = GroundState(kind='dft', xc='hf')
    assert gs.label() == 'E_HF'


def test_ground_state_label_uppercases_hyphenated_functional():
    gs = GroundState(kind='dft', xc='cam-b3lyp')
    assert gs.label() == 'E_CAM-B3LYP'


# ---------------------------------------------------------------------------
# SurfacePhysics.label()
# ---------------------------------------------------------------------------

def test_surface_physics_label_no_excitation():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=None)
    assert sp.label() == gs.label()


def test_surface_physics_label_excitation_on_rpa():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=Excitation(spin='singlet'))
    assert sp.label() == gs.label() + ' + Omega'


def test_surface_physics_label_excitation_on_dft():
    gs = GroundState(kind='dft', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=Excitation(spin='singlet'))
    assert sp.label() == gs.label() + ' + Omega (mean-field ground state)'


def test_surface_physics_label_charged_removal():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=ChargedExcitation(orbital=5, charge_change=-1))
    assert sp.label() == gs.label() + ' - eps^QP_p'


def test_surface_physics_label_charged_addition():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=ChargedExcitation(orbital=5, charge_change=1))
    assert sp.label() == gs.label() + ' + eps^QP_p'


def test_surface_physics_label_appends_nongas_environment():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=None, environment='pcm(water)')
    assert sp.label() == gs.label() + ' in pcm(water)'


def test_surface_physics_label_gas_environment_not_appended():
    gs = GroundState(kind='rpa', xc='pbe0')
    sp = SurfacePhysics(ground_state=gs, excitation=None, environment='gas')
    assert sp.label() == gs.label()


# ---------------------------------------------------------------------------
# comparable_with
# ---------------------------------------------------------------------------

def test_comparable_with_true_for_same_ground_state_different_excitation():
    gs = GroundState(kind='rpa', xc='pbe0')
    a = SurfacePhysics(ground_state=gs, excitation=Excitation(spin='singlet', root=1))
    b = SurfacePhysics(ground_state=gs, excitation=Excitation(spin='triplet', root=3))
    assert a.comparable_with(b)
    assert b.comparable_with(a)


def test_comparable_with_false_for_rpa_vs_dft():
    a = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'), excitation=None)
    b = SurfacePhysics(ground_state=GroundState(kind='dft', xc='pbe0'), excitation=None)
    assert not a.comparable_with(b)


def test_comparable_with_false_for_gas_vs_solvent():
    gs = GroundState(kind='rpa', xc='pbe0')
    a = SurfacePhysics(ground_state=gs, excitation=None, environment='gas')
    b = SurfacePhysics(ground_state=gs, excitation=None, environment='pcm(water)')
    assert not a.comparable_with(b)


# ---------------------------------------------------------------------------
# PhysicsMismatch
# ---------------------------------------------------------------------------

def test_physics_mismatch_carries_a_and_b_and_both_labels():
    a = SurfacePhysics(ground_state=GroundState(kind='rpa', xc='pbe0'), excitation=None)
    b = SurfacePhysics(ground_state=GroundState(kind='dft', xc='pbe0'), excitation=None)
    exc = PhysicsMismatch(a, b)
    assert exc.a is a
    assert exc.b is b
    message = str(exc)
    assert a.label() in message
    assert b.label() in message


# ---------------------------------------------------------------------------
# validation: one test per rule, each paired with a construction that succeeds
# ---------------------------------------------------------------------------

def test_ground_state_kind_rejects_unknown_value():
    GroundState(kind='rpa', xc='pbe0')
    with pytest.raises(ValueError, match='kind'):
        GroundState(kind='ks', xc='pbe0')


def test_excitation_spin_rejects_unknown_value():
    Excitation(spin='singlet')
    with pytest.raises(ValueError, match='spin'):
        Excitation(spin='quartet')


def test_excitation_root_rejects_below_one():
    Excitation(spin='singlet', root=1)
    with pytest.raises(ValueError, match='root'):
        Excitation(spin='singlet', root=0)


def test_excitation_kernel_rejects_unknown_value():
    Excitation(spin='singlet', kernel='bse-tda')
    with pytest.raises(ValueError, match='kernel'):
        Excitation(spin='singlet', kernel='tda')


def test_excitation_qp_rejects_unknown_value():
    Excitation(spin='singlet', qp='evgw')
    with pytest.raises(ValueError, match='qp'):
        Excitation(spin='singlet', qp='gw0')


def test_excitation_screening_rejects_unknown_value():
    Excitation(spin='singlet', screening='rpa')
    with pytest.raises(ValueError, match='screening'):
        Excitation(spin='singlet', screening='gwgamma')


def test_charged_excitation_orbital_rejects_negative():
    ChargedExcitation(orbital=0, charge_change=1)
    with pytest.raises(ValueError, match='orbital'):
        ChargedExcitation(orbital=-1, charge_change=1)


def test_charged_excitation_charge_change_rejects_other_values():
    ChargedExcitation(orbital=0, charge_change=-1)
    ChargedExcitation(orbital=0, charge_change=1)
    with pytest.raises(ValueError, match='charge_change'):
        ChargedExcitation(orbital=0, charge_change=0)


def test_qp_states_kind_rejects_unknown_value():
    QPStates(kind='frontier')
    with pytest.raises(ValueError, match='kind'):
        QPStates(kind='core')


def test_qp_states_threshold_rejects_unknown_value():
    QPStates(kind='admitted', threshold='omega1')
    with pytest.raises(ValueError, match='threshold'):
        QPStates(kind='admitted', threshold='window')


def test_qp_states_half_width_rejects_below_one():
    QPStates(kind='frontier', half_width=1)
    with pytest.raises(ValueError, match='half_width'):
        QPStates(kind='frontier', half_width=0)


def test_qp_states_extra_virtuals_rejects_negative():
    QPStates(kind='valence', extra_virtuals=0)
    with pytest.raises(ValueError, match='extra_virtuals'):
        QPStates(kind='valence', extra_virtuals=-1)
