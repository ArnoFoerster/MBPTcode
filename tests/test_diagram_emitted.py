"""Emitted diagram-ADC modules == the Goldstone engine, machine precision.

The emitter (src/CodeGeneration/diagrammatic/emit.py) writes m{n}_blocks.py;
generated output is not part of the repository, so each module is emitted into
a temporary directory here and imported from there. The gate proves each
emitted assemble_m{n} agrees with the in-memory engine that passed the
exact-oracle gates, so the emitted code inherits those validations
transitively. n=4 is included: the emitted module is large (one function per
block, one term per Goldstone diagram) but runs in seconds at validation scale.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='the diagram engine needs pdaggerq (optional)')


@pytest.fixture(scope='module')
def h4():
    from src.CodeGeneration.diagrammatic import exact_oracle as EO
    mol, mf = EO.h4_chain()
    return EO.spinorb_system(mol, mf)


@pytest.mark.parametrize('order', [2, 3, 4])
def test_emitted_equals_engine(order, h4, tmp_path):
    from src.CodeGeneration.diagrammatic import goldstone as GS
    from src.CodeGeneration.diagrammatic.emit import emit_order
    path = emit_order(order, str(tmp_path))
    spec = importlib.util.spec_from_file_location(f'm{order}_blocks', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    eps, h1, g, nocc = h4
    omega = 0.4 + 0.6j
    emitted = getattr(mod, f'assemble_m{order}')(eps, g, nocc, omega)
    engine = GS.assemble_m(eps, g, nocc, order, [omega])[0]
    assert np.max(np.abs(emitted - engine)) < 1e-14


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
