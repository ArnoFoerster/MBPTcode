"""A quasiparticle-Hamiltonian subspace member that takes no pole model where
the subspaces are frozen leaves its subspace, and the record says so
(`ExcitedStateChain._frozen_qp_subspaces`).

Sigma^c_pq is built on the pole model alone. A refrozen chain decides
pole-model admission afresh at its own geometry but carries the previous
surface's subspaces (`carried_subspaces`), so a carried member can arrive
off the pole model (DOBNA/cc-pVDZ, PBE0: orbital 46, 'scissor', at the first
forward of a refrozen surface). The member is dropped there and keeps its own
diagonal correction; a
subspace left with one member is none.

Formaldehyde/cc-pVDZ, PBE0, C 0.1 Bohr out of plane, set 4-17: the pole
model admits 4-10 and 11-17 take the quadrature. The chain is handed the
carried subspace (9, 10, 11) exactly as `refreeze` hands it one, so orbital
11 is a carried member that is not admitted at this geometry:
  - the first forward completes, 11 leaves, and (9, 10) stays coupled;
  - the drop is recorded on the chain and in `qp_bookkeeping`, with the
    subspace 11 left, the route it took and that it was carried;
  - the subspaces stay frozen at a displaced geometry, and a further
    refreeze carries only what remained.
"""
import warnings

import pytest

from src.gradients.gwbse.diagnostics import qp_bookkeeping
from tests.test_qp_hamiltonian import chain, displaced, pbe0

CARRIED = (9, 10, 11)
NOT_ADMITTED = 11
REMAINING = (9, 10)


def carried_chain():
    ch = chain(pbe0, qp_hamiltonian=True)
    ch.conventions.carried_subspaces = (CARRIED,)
    return ch


def test_a_carried_member_off_the_pole_model_is_dropped_and_recorded():
    ch = carried_chain()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ch.excitation()
    assert any('leave their subspaces' in str(w.message) for w in caught)
    assert REMAINING in ch.resolved.qp_subspaces[1]
    assert all(NOT_ADMITTED not in b for b in ch.resolved.qp_subspaces[1])
    record = ch.conventions.qp_subspace_record
    assert record['carried'] == [list(CARRIED)]
    drop = [d for d in record['dropped'] if d['orbital'] == NOT_ADMITTED]
    assert len(drop) == 1
    assert drop[0]['subspace'] == list(CARRIED)
    assert drop[0]['route'] not in (None, 'sop')
    assert drop[0]['carried'] is True
    assert drop[0]['remaining'] == list(REMAINING)
    assert list(REMAINING) in record['subspaces']
    assert qp_bookkeeping(ch, ch.mol0)['subspaces'] == record

    ch.excitation(displaced(ch.mol0, 1, 0, 0.05))
    assert REMAINING in ch.resolved.qp_subspaces[1]
    assert ch.conventions.qp_subspace_record is record
    assert ch.refreeze(ch.mol0).resolved.carried_subspaces == ch.resolved.qp_subspaces[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
