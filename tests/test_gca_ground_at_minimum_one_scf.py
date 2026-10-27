"""E_0 at an excited minimum stands on the walk's own mean field.

`ground_at_minimum` rebuilds the ground surface at the minimum; a bare
`ground.refreeze(mol)` would converge a second SCF there beside the one the
relaxed record holds, and `ground_state_at` would read the rebuilt surface's
own. `RPAGroundStateChain.refreeze` takes `mf=`: the record's mean field
is handed through, so E_0 and the excited energy at R* come from one SCF.
A ground surface whose `refreeze` takes no mean field is rebuilt with it alone.
"""
from pyscf import gto, scf

from src.properties.excitations import ground_at_minimum
from src.gradients.mean_field_surface import MeanFieldSurface


class Ground:
    """A ground surface whose refreeze takes the mean field
    (`RPAGroundStateChain.refreeze(mol, factorization=None, mf=None)`)."""

    def __init__(self):
        self.handed = 'never called'

    def refreeze(self, mol, factorization=None, mf=None):
        self.handed = mf
        return self


def relaxed_record(surface):
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    return {'mol': mol, 'mf': scf.RHF(mol).run(), 'surface': surface}


def test_the_ground_surface_is_rebuilt_on_the_walks_mean_field():
    ground = Ground()
    relaxed = relaxed_record(object())
    ground_at_minimum(ground, object(), relaxed)
    assert ground.handed is relaxed['mf']


def test_a_ground_surface_without_a_mean_field_argument_is_rebuilt():
    relaxed = relaxed_record(object())
    ground = MeanFieldSurface(relaxed['mol'], lambda m: scf.RHF(m).run())
    rebuilt = ground_at_minimum(ground, object(), relaxed)
    assert rebuilt.mol0 is relaxed['mol']


def test_the_walks_own_surface_is_kept():
    ground, excited = Ground(), object()
    assert ground_at_minimum(ground, excited, relaxed_record(excited)) is ground
    assert ground.handed == 'never called'
