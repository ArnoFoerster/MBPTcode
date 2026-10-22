"""What a refused match of the frozen conventions reports, and that an
explicit orbital carried grow-only into a rebuilt chain is matched on a walk
step (`gradients.conventions`, `ExcitedStateChain.refreeze`).

  - an explicit orbital spread over the orbitals of a larger unit beside it
    gets none of them back: the refusal names how much of it the best
    orbital here holds, which orbital, and the unit that took it, rather
    than an empty match that reads like a lost key (the DMABN T1 walk from
    the twisted S1 minimum: the window edge, orbital 57, 0.458 in one
    orbital 0.12 Bohr along the start nudge, the 146 outside virtuals
    holding 0.529 of it);
  - formaldehyde/cc-pVDZ, HF, the adaptive set of tests/test_adaptive_qp_set:
    a state made explicit on a refrozen surface, carried as the floor into
    R0's chain rebuilt there (the T1 stage's `rebuilt_at`), is explicit on
    the rebuilt chain and matched to itself on a step from it.
"""
import warnings

import numpy as np
import pytest

from src.gradients.conventions import ConventionsLost
from tests.test_adaptive_qp_set import adaptive, chain, system, with_extra  # noqa: F401
from tests.test_frozen_conventions import displaced, frozen

#: weights of the edge orbital over itself and two outside orbitals
SPREAD = (0.4, 0.3, 0.3)
#: Bohr, a walk step from the rebuilt chain's reference
STEP = 0.02 * np.array([[0.1, 0.3, -0.2], [-0.2, 0.0, 0.3],
                        [0.3, -0.1, 0.0], [0.0, 0.2, 0.1]])


def spread_over(norb, p, others, weights):
    """Overlaps under which key `p` holds `weights` of orbitals p, *others."""
    t = np.eye(norb)
    v = np.sqrt(np.asarray(weights, float))
    u = np.zeros(len(v))
    u[0] = 1.0
    u -= v
    u /= np.linalg.norm(u)
    idx = [p, *others]
    t[np.ix_(idx, idx)] = np.eye(len(v)) - 2.0 * np.outer(u, u)
    return t


def test_an_empty_match_names_the_weight_and_the_unit_that_took_it(
        monkeypatch):
    c = frozen()
    t = spread_over(10, 7, (8, 9), SPREAD)
    with pytest.raises(ConventionsLost) as exc:
        c.at(*displaced(monkeypatch, t))
    msg = str(exc.value)
    assert 'explicit [7] gets orbitals [] holding none of its span' in msg
    assert (f'slot 7 is at most {max(SPREAD):.3f} in one orbital here, 7, '
            f'of which shared [4, 8-9] holds {1 - max(SPREAD):.3f}') in msg


def test_a_carried_explicit_orbital_is_matched_on_a_step(system):
    warnings.simplefilter('ignore')
    r0 = chain(system, adaptive())
    r0.energy(system['mol'], system['mf'])
    part = r0.resolved.qp_partition
    assert part.holes, 'the gate needs a hole at R0'
    x = int(part.holes[0])
    m = system['displaced']
    walked = chain(system, adaptive(), partition=with_extra(part, [x]))
    walked = walked.refreeze(m)
    walked.energy(m)
    explicit = walked.resolved.qp_partition.explicit
    assert x in explicit
    rebuilt = r0.refreeze(m, qp_floor=explicit)
    rebuilt.energy(m)
    assert x in set(int(p) for p in rebuilt.resolved.qp_set)
    assert rebuilt.qp_growth['not_explicit'] == []
    step = m.copy()
    step.set_geom_(m.atom_coords() + STEP, unit='Bohr')
    triplet = rebuilt.spin_view('triplet', 0)
    triplet.energy(step)
    view = triplet.resolved
    assert view.index(x) == x
    assert x in set(int(p) for p in view.qp_set)
