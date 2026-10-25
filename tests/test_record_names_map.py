"""Records that name classes by their old modules still load.

Modules moved, and a record names each class it holds as
'module:qualname'. `record_io.record_class` reads an old name through the
generated map `src/Base/utils/record_names.json`. The fixture
`tests/data/pre_r3/pre_r3_record.json` was written by the tree before the move
of the record classes and names Numerics and its nested settings,
NoEnvironment and FirstPoint by their old modules.
"""
import importlib
import json
import pathlib

import numpy as np

from src.Base.environment_seam import NoEnvironment
from src.Base.numerics.numerics import Numerics
from src.Base.utils import record_io
from src.gradients.surface_identity import FirstPoint

FIXTURE = pathlib.Path(__file__).resolve().parent / 'data' / 'pre_r3' / 'pre_r3_record.json'


def test_a_record_written_before_r3_reads_back_as_the_moved_classes():
    value = record_io.read_record(FIXTURE)
    assert type(value['numerics']) is Numerics
    assert value['numerics'] == Numerics()
    assert type(value['gas']) is NoEnvironment
    point = value['first_point']
    assert type(point) is FirstPoint
    np.testing.assert_array_equal(point.coords, np.arange(6.0).reshape(2, 3))
    assert point.result[1] == -76.25 and point.conventions == 'abc'


def test_the_fixture_names_the_old_modules():
    """A fixture rewritten with the new names would test nothing."""
    text = FIXTURE.read_text()
    assert 'src.Base.numerics:Numerics' in text
    assert 'src.Base.environment:NoEnvironment' in text
    assert 'src.gradients.excited_state:FirstPoint' in text


def test_every_mapped_class_exists_and_no_old_name_does():
    """Each target imports, and an old name that still resolves resolves to
    the same class, so the map can never shadow a class that is still where
    the record says."""
    names = json.loads(record_io.RECORD_NAMES.read_text())
    assert names
    for old, new in names.items():
        module, qualname = new.split(':')
        try:
            target = importlib.import_module(module)
        except ImportError as exc:
            # an optional library (pyelpa) the module needs; never a src module
            assert not (exc.name or '').startswith('src'), (new, exc)
            continue
        assert hasattr(target, qualname), new
        old_module, old_name = old.split(':')
        try:
            held = importlib.import_module(old_module)
        except ImportError:
            continue
        # still reachable under the old name only as the same class (a module
        # that imports it for its own use), never as another one
        assert getattr(held, old_name, None) in (None, getattr(target, qualname)), old


def test_a_current_name_passes_through_unchanged():
    name = 'src.Base.numerics.numerics:Numerics'
    assert record_io.current_record_name(name) == name
    assert record_io.record_class(name) is Numerics
