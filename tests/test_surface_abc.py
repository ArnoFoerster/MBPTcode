"""Every potential-energy surface inherits the abstract base class `PotentialEnergySurface`
: a surface that lacks one of its six members is refused when it is
created."""
import pytest

from src.gradients.surface import PotentialEnergySurface

MEMBERS = ('total_energy', 'total_gradient', 'refreeze', 'label', 'mean_field',
           'identity')


def toy(skip=None):
    body = {m: (lambda self, *a, **k: None) for m in MEMBERS if m != skip}
    return type(f'Toy_{skip}', (PotentialEnergySurface,), body)


def test_a_surface_with_every_member_is_created():
    assert isinstance(toy()(), PotentialEnergySurface)


@pytest.mark.parametrize('missing', MEMBERS)
def test_a_surface_missing_a_member_is_refused_at_creation(missing):
    with pytest.raises(TypeError, match=missing):
        toy(missing)()


def test_a_part_listed_before_surface_supplies_the_member():
    """PotentialEnergySurface comes last among the bases: a part's method is found first."""
    class Part:
        def mean_field(self, mol=None, mf=None):
            return 'part'

    body = {m: (lambda self, *a, **k: None) for m in MEMBERS if m != 'mean_field'}
    cls = type('Composed', (Part, PotentialEnergySurface), body)
    assert cls().mean_field() == 'part'


def test_the_mean_field_surfaces_are_surfaces():
    from src.gradients.mean_field_surface import MeanFieldSurface
    from src.gradients.surface import FiniteDifferenceGradient
    for cls in (MeanFieldSurface, FiniteDifferenceGradient):
        assert issubclass(cls, PotentialEnergySurface) and not cls.__abstractmethods__, cls


def test_the_dense_surfaces_are_surfaces():
    from src.gradients.gwbse import dense_surfaces as ds
    for cls in (ds.DenseRPASurface, ds.QuasiparticleSurface, ds.DenseBSESurface):
        assert issubclass(cls, PotentialEnergySurface) and not cls.__abstractmethods__, cls


def test_the_composed_surfaces_are_surfaces():
    from src.gradients.gwbse import rpa_bse_surface as rb
    for cls in (rb.RPAQPSurface, rb.MeanFieldQPSurface, rb.RPABSESurface):
        assert issubclass(cls, PotentialEnergySurface) and not cls.__abstractmethods__, cls


def test_the_factor_chains_are_surfaces_with_their_reference_mean_field():
    """Their mean_field is ReferenceMeanField's, found before the abstract one."""
    from src.gradients.gwbse.excited_state import ExcitedStateChain
    from src.gradients.reference_mean_field import ReferenceMeanField
    from src.gradients.rpa.ground_state import RPAGroundStateChain
    for cls in (ExcitedStateChain, RPAGroundStateChain):
        assert issubclass(cls, PotentialEnergySurface) and not cls.__abstractmethods__, cls
        assert cls.mean_field is ReferenceMeanField.mean_field, cls


def test_the_diabatic_and_solvated_surfaces_are_surfaces():
    from src.gradients.gwbse.fragment_diabatic import DiabaticSurface
    from src.gradients.solvated_surface import SolvatedSurface
    for cls in (DiabaticSurface, SolvatedSurface):
        assert issubclass(cls, PotentialEnergySurface) and not cls.__abstractmethods__, cls


def surface_classes():
    """Every concrete surface class of src.gradients and src.properties."""
    import importlib
    import inspect
    import pkgutil
    for package in ('src.gradients', 'src.properties'):
        pkg = importlib.import_module(package)
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f'{package}.'):
            importlib.import_module(info.name)

    def below(c):
        for s in c.__subclasses__():
            yield s
            yield from below(s)
    return sorted({c for c in below(PotentialEnergySurface)
                   if c.__module__.startswith('src.') and not inspect.isabstract(c)},
                  key=lambda c: f'{c.__module__}.{c.__name__}')


def held_ingredients(cls):
    """The ingredients a class holds by its source: methods, class attributes
    and `self.<name> =` stores of every src class in its MRO."""
    import ast
    import inspect
    import textwrap
    from src.gradients.surface import INGREDIENT_DOCS
    found = set()
    for k in cls.__mro__:
        if not k.__module__.startswith('src.') or k is PotentialEnergySurface:
            continue
        node = ast.parse(textwrap.dedent(inspect.getsource(k))).body[0]
        found |= {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        found |= {t.id for n in node.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name)}
        found |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
                  and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
                  and n.value.id == 'self'}
    return found & set(INGREDIENT_DOCS)


def test_every_surface_declares_exactly_the_ingredients_it_holds():
    """`provides` reads the declaration; a declaration that
    left out what the class holds, or claimed what it does not, would answer
    a property routine wrongly."""
    classes = surface_classes()
    assert len(classes) >= 12
    for cls in classes:
        assert cls.INGREDIENTS == held_ingredients(cls), cls.__name__
