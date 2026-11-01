"""Three defects of the diabatic and solvated surfaces' plumbing, on stand-ins
(no SCF):

  * `DiabaticSurface(..., omega0=None)`, the documented default, builds: the
    reference partition chooses Omega_0, which every later geometry reuses
    (it raised AttributeError, reading self.omega0 before setting it);
  * `DiabaticSurface.mean_field(mol, mf)` hands `mf` to its method, as the
    `PotentialEnergySurface` protocol has it (it dropped the argument);
  * a `SolvatedSurface` whose state is seeded from S0's equilibrium charges
    logs the seeding loop's numerical flags (they were dropped).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyscf import gto

from src.gradients import fragment_diabatic, solvated_surface
from src.gradients.fragment_diabatic import DiabaticSurface
from src.gradients.solvated_surface import SolvatedSurface

#: Ha: the Omega_0 the stand-in reference partition takes when given none
TAKEN = 0.3125


class Method:
    """A fragment method of which the surface reads mol0 and mean_field."""

    def __init__(self):
        self.mol0, self.asked = 'mol0', []

    def mean_field(self, mol=None, mf=None):
        self.asked.append((mol, mf))
        return mol, mf


def stand_in_gradient(seen):
    class Gradient:
        def __init__(self, method, fragments, sites, ct, omega0=None, **kw):
            seen.append(omega0)
            self.partition = SimpleNamespace(
                omega0=TAKEN if omega0 is None else omega0)
            self.orbitals = 'orbitals'
    return Gradient


def test_a_diabat_without_omega0_takes_the_references(monkeypatch):
    seen = []
    monkeypatch.setattr(fragment_diabatic, 'DiabaticGradient',
                        stand_in_gradient(seen))
    surf = DiabaticSurface(Method(), 'fragments', 'sites', 'ct 0->1')
    assert surf.omega0 == TAKEN
    surf._gradient_object('mol', 'mf')
    assert seen == [None, TAKEN]


def test_the_mean_field_takes_the_callers_mf(monkeypatch):
    monkeypatch.setattr(fragment_diabatic, 'DiabaticGradient',
                        stand_in_gradient([]))
    method = Method()
    surf = DiabaticSurface(method, 'fragments', 'sites', 'ct 0->1', omega0=0.2)
    assert surf.mean_field('mol', 'mf') == ('mol', 'mf')
    assert method.asked[-1] == ('mol', 'mf')


def test_the_seeding_loops_flags_are_logged(monkeypatch):
    mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
    ground, state = object(), SimpleNamespace(freezes_on_first_call=True)
    flag = {'flag': 'loop_not_converged', 'message': 'the S0 seed stopped short'}

    def equilibrate(model, field, **kw):
        return SimpleNamespace(q=np.zeros(field.nsites), energy=0.0, info={},
                               flags=[flag] if model is ground else [],
                               history=[], polarization=0.0)
    monkeypatch.setattr(solvated_surface.sl, 'equilibrate', equilibrate)
    surf = SolvatedSurface(mol, lambda m: SimpleNamespace(nsites=4),
                           lambda m, f: state, make_ground=lambda m, f: ground)
    monkeypatch.setattr(surf, '_fraction', lambda s: None)
    surf.total_energy(mol)
    assert surf.flags_at(mol) == ['loop_not_converged']
