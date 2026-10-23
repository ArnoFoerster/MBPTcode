"""A quasiparticle-Hamiltonian member that leaves the pole model INSIDE a
frozen set, at a displaced geometry, is handled as the freeze-time drop is:
dropped from its group and recorded, and the walk goes on.

`_frozen_qp_subspaces` drops a member off the pole model only where the
subspaces are frozen (`test_qp_hamiltonian_drops_non_pole_member`). Once they
are, `_qp_hamiltonian` raised NotImplementedError for a member whose route at
a later geometry is not the pole model, which ends a walk. Formaldehyde's
orbital 11 does this on the DF mean field (Z -0.07 at a displaced geometry);
the geometry was not recorded, so here the member's route is set off the pole
model at a displaced geometry: 'scissor', the route DOBNA's orbital 46 took at
a freeze-time drop.

Formaldehyde/cc-pVDZ, DF Hartree-Fock, set 4-17: the reference freezes the
pair (12, 13) (`test_qp_hamiltonian`); C 0.05 Bohr along x, where 13 is set
off the pole model.
"""
import warnings

import numpy as np
import pytest

from src.properties.excitations import qp_bookkeeping
from tests.test_qp_hamiltonian import (COORDINATES, FD_STEP, FD_TOL, PAIR,
                                       chain, displaced)

#: The route a member takes off the pole model in this test.
OFF_ROUTE = 'scissor'


def entries(record):
    """Every dict anywhere inside a record."""
    if isinstance(record, dict):
        yield record
        for value in record.values():
            yield from entries(value)
    elif isinstance(record, (list, tuple)):
        for value in record:
            yield from entries(value)


def off_pole_model(ch, member):
    """`ch` reports `member`'s route off the pole model from here on."""
    taken = ch.quasiparticles._routes_taken

    def routes(route_out):
        out = taken(route_out)
        if member in out:
            out[member] = OFF_ROUTE
        return out
    ch.quasiparticles._routes_taken = routes


def test_a_member_lost_inside_a_frozen_set_is_dropped_and_recorded():
    ch = chain(qp_hamiltonian=True)
    ch.excitation()
    assert ch.resolved.qp_subspaces[1] == (PAIR,)
    member = PAIR[-1]
    off_pole_model(ch, member)
    moved = displaced(ch.mol0, 1, 0, 0.05)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        omega = ch.excitation(moved)
    assert np.isfinite(omega)
    assert any(str(member) in str(w.message) for w in caught)
    lost = [d for d in entries(qp_bookkeeping(ch, moved)['subspaces'])
            if d.get('orbital') == member and d.get('route') == OFF_ROUTE]
    assert lost, f'the drop of orbital {member} is recorded nowhere'
    # the reference keeps the frozen record alone
    assert 'dropped_here' not in qp_bookkeeping(ch, ch.mol0)['subspaces']


def test_the_force_past_the_drop_is_the_derivative_of_its_energy():
    """Past the drop the member carries its own diagonal root, and the force
    is that surface's: central difference about the displaced geometry."""
    ch = chain(qp_hamiltonian=True)
    ch.excitation()
    off_pole_model(ch, PAIR[-1])
    moved = displaced(ch.mol0, 1, 0, 0.05)
    warnings.simplefilter('ignore')
    grad = ch.excitation_gradient(moved)[0]
    for atom, axis in COORDINATES[:2]:
        fd = (ch.excitation(displaced(moved, atom, axis, FD_STEP))
              - ch.excitation(displaced(moved, atom, axis, -FD_STEP))) \
            / (2 * FD_STEP)
        assert abs(fd - grad[atom, axis]) < FD_TOL, (atom, axis, fd,
                                                     grad[atom, axis])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
