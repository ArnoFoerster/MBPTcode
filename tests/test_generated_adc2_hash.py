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
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from src.Base.pdaggerq_interface import pdaggerq_available

pytestmark = pytest.mark.skipif(not pdaggerq_available(),
                                reason='the generators need pdaggerq (optional)')

EXPECTED = {
    'adc2_blocks.py':
        'ea9ff2f3d449567045e59dacbf635506b2276bad18eaf2b4e71548ea3c493c97',
    'adc2_blocks_restricted.py':
        'e5af8d1e06bfd37c4763a103a90cbddd5b87685e5603af2992e5f803962bc245',
}


def test_adc2_modules_hash_as_recorded(tmp_path, monkeypatch):
    from src.CodeGeneration import generate_adcn
    monkeypatch.setattr(generate_adcn, 'OUT_DIR', str(tmp_path))
    generate_adcn.generate_adc2()
    generate_adcn.generate_adc2_restricted()
    for name, sha in EXPECTED.items():
        with open(tmp_path / 'adc2' / name, 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha, name


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
