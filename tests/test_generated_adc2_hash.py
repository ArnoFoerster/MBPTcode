"""The ADC(2) generators emit exactly the recorded text.

Generated ADC modules are not part of the repository (src/SingleReference/ADC/
generatedADC/ is ignored), so the reproducibility record is the generator plus
these hashes: generate_adc2 and generate_adc2_restricted write into a temporary
directory, and the SHA-256 of each module must match the value recorded when the
generators were ported (2026-10-19, pdaggerq f3a1f29). A change of the emitted
text, from the generator or from pdaggerq, fails here; re-record only after
checking the new modules against the old ones.

At recording, both bodies were identical to the modules the same generator had
produced before the port; only the header line naming the generator's path and
the einsum import (cached_einsum) differed. The same two hashes came out of
a second pdaggerq build (a cluster, Python 3.11). Larger modules do not: the ADC(4)
l3 Laplace module's term list differs between those two builds while every
block agrees with its dense twin to 1.6e-11, so a text hash there would pin the
build, not the physics; tests/test_l3_laplace.py checks it numerically instead.

Re-recorded 2026-10-22 (one block emitter for the spin-orbital and
restricted twins, the codegen_common header): the new modules are the old ones
up to the header comment, the docstrings and the order of the functions -- every
function has the same name and the same AST body (as checked for the
record), and the old hashes still came out of the base generators.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.CodeGeneration.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='the generators need pdaggerq (optional)')

EXPECTED = {
    'adc2_blocks.py':
        'a9ff42c0db3da9a1efd2858c7be5b27889987d4e915bf379ecb0d4c77a9f563f',
    'adc2_blocks_restricted.py':
        'e63c4005e3d1d33a7efb6f78cad96f35f4f9165d777bc6b0f187a0e4a6e861af',
}


def test_adc2_modules_hash_as_recorded(tmp_path, monkeypatch):
    from src.CodeGeneration import codegen_common, generate_adcn
    monkeypatch.setattr(codegen_common, 'ADC_OUT_DIR', str(tmp_path))
    generate_adcn.generate_adc2()
    generate_adcn.generate_adc2(restricted=True)
    for name, sha in EXPECTED.items():
        with open(tmp_path / 'adc2' / name, 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha, name


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
