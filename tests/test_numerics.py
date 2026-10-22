"""The one numerics object: immutable, hashed, derived only with a reason.

What is pinned here: its record round-trips, any choice changes its
identity, provenance does not, `derive` writes down its parent, the change
and the reason, and its defaults are the chain's own -- a second default is
exactly the inconsistency the object exists to remove. Memory and layout are recorded and not
hashed, and a bar is no choice.
"""
from dataclasses import FrozenInstanceError, fields

import pytest

from src.Base.declaration import QPStates
from src.Base.numerics import (FLAT_FIELDS, IDENTITY_EXCLUDED, IDENTITY_FIELDS,
                               SECTIONS,
                               BSENumerics, IsdfNumerics, LayoutNumerics,
                               Numerics, numerics_of)
from src.Base.record_io import record_hash
from src.Base import separable_ri

#: A value other than the default, per field, for the identity test.
OTHER = {bool: lambda v: not v, int: lambda v: v + 1,
         float: lambda v: v * 2.0, str: lambda v: v + '-other',
         type(None): lambda v: 3}


def every_field():
    for name, cls in SECTIONS.items():
        for f in fields(cls):
            yield name, f.name


def test_the_object_is_immutable():
    n = Numerics()
    with pytest.raises(FrozenInstanceError):
        n.isdf = IsdfNumerics()
    with pytest.raises(FrozenInstanceError):
        n.isdf.level = 'G3'


def test_a_record_round_trips():
    n = Numerics().derive('a finer Hessian grid', isdf={'level': 'GH'})
    back = Numerics.from_record(n.to_record())
    assert back == n and back.identity() == n.identity()
    assert record_hash(back.to_record()) == record_hash(n.to_record())


@pytest.mark.parametrize('section,name', list(every_field()))
def test_every_choice_moves_the_identity(section, name):
    base = Numerics()
    value = getattr(getattr(base, section), name)
    other = OTHER.get(type(value), lambda v: 'other')(value)
    if (section, name) == ('bse', 'solver'):
        other = 'dense'
    if (section, name) == ('bse', 'adjoint'):
        other = 'grid'
    if (section, name) == ('bse', 'track'):
        other = 'overlap'
    if (section, name) == ('qp', 'outside_scissor'):
        other = 'curve'
    if (section, name) == ('qp', 'residue_route'):
        other = 'sop'
    if (section, name) == ('isdf', 'fit'):
        other = 'rows'
    if (section, name) == ('layout', 'z_mode'):
        other = 'dense'
    if (section, name) == ('bse', 'preconditioner'):
        other = 'bare'
    if (section, name) == ('qp', 'scissor'):
        other = 'calibrate'
    if (section, name) == ('qp', 'select'):
        other = QPStates('adaptive')
    changed = base.derive('test', **{section: {name: other}})
    assert record_hash(changed.values()) != record_hash(base.values())
    if section == 'layout' or (section, name) in IDENTITY_EXCLUDED:
        # the layout, and a declaration that permits a calculation without
        # changing any (`IsdfNumerics.runtime_radii`)
        assert (section not in IDENTITY_FIELDS
                or (section, name) in IDENTITY_EXCLUDED)
        assert changed.identity() == base.identity(), (section, name)
    else:
        assert changed.identity() != base.identity(), (section, name)


def test_provenance_is_not_a_choice():
    """Two objects with the same choices are one calculation, however each
    came to be."""
    base = Numerics()
    there = base.derive('there', bse={'nroots': 9})
    back = there.derive('and back', bse={'nroots': base.bse.nroots})
    assert back.identity() == base.identity()
    assert back.parent == there.identity() and back.reason == 'and back'


def test_derive_writes_down_parent_change_and_reason():
    base = Numerics()
    child = base.derive('FD Hessian on the Hessian grid',
                        layout={'tile_gb': 2.0},
                        isdf={'level': 'GH'}, walk={'fd_step': 5e-4})
    assert child.parent == base.identity()
    assert child.reason == 'FD Hessian on the Hessian grid'
    assert set(child.change) == {('isdf.level', None, 'GH'),
                                 ('layout.tile_gb', None, 2.0),
                                 ('walk.fd_step', base.walk.fd_step, 5e-4)}


@pytest.mark.parametrize('kw', [dict(reason=''), dict(reason='  ')])
def test_derive_without_a_reason_is_refused(kw):
    with pytest.raises(ValueError, match='reason'):
        Numerics().derive(isdf={'level': 'GH'}, **kw)


def test_derive_of_nothing_or_of_no_field_is_refused():
    with pytest.raises(ValueError, match='changes nothing'):
        Numerics().derive('same', bse={'nroots': Numerics().bse.nroots})
    with pytest.raises(ValueError, match='no field'):
        Numerics().derive('typo', bse={'n_roots': 3})
    with pytest.raises(ValueError, match='no section'):
        Numerics().derive('typo', grids={'ntau': 3})


def test_the_records_validation_holds():
    with pytest.raises(ValueError, match='auto'):
        Numerics().derive('auto', qp={'residue_route': 'auto'})
    with pytest.raises(ValueError, match='solver'):
        Numerics().derive('bad', bse={'solver': 'lanczos'})


def test_the_fit_defaults_are_the_fits():
    """The fit reads the object (section 5 step 4): the values it runs on
    where nothing is derived are the object's defaults."""
    isdf = IsdfNumerics()
    assert separable_ri.fit_settings(
        None, 'l_max_second', 'pair_tol', 'regularization') == (
        isdf.l_max_second, isdf.pair_tol, isdf.regularization)


def test_a_section_may_be_given_as_its_fields():
    n = Numerics(bse={'nroots': 7}, layout={'sliced': True})
    assert n.bse.nroots == 7 and n.layout.sliced
    assert n == Numerics(bse=BSENumerics(nroots=7),
                         layout=LayoutNumerics(sliced=True))
    with pytest.raises(TypeError):
        Numerics(bse={'nroot': 7})


def test_numerics_of_maps_the_flat_names():
    n = numerics_of({'solver': 'dense', 'bse_conv_tol': 1e-7,
                     'qp_window': 3, 'sliced': True, 'ntau_w': 12})
    assert n.bse.solver == 'dense' and n.bse.conv_tol == 1e-7
    assert n.qp.window == 3 and n.layout.sliced
    assert n.quadrature.ntau_w == 12
    for name, field in FLAT_FIELDS.items():
        section, key = field.split('.')
        assert hasattr(getattr(Numerics(), section), key), name
    with pytest.raises(ValueError, match='orbital_tol'):
        numerics_of({'orbital_tol': None})


def test_a_bar_is_no_choice():
    """O5: the orbital residual bound is a constant, not a choice."""
    assert not hasattr(IsdfNumerics(), 'orbital_tol')
