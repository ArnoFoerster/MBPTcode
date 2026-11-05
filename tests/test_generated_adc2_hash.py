"""The ADC(2) generators emit exactly the recorded text.

Generated ADC modules are not part of the repository (src/SingleReference/ADC/
generatedADC/ is ignored), so the reproducibility record is the generator plus
these hashes: generate_adc2 and generate_adc2_restricted write into a temporary
directory, and the SHA-256 of each module must match the recorded value (pdaggerq
commit f3a1f29). A change of the emitted
text, from the generator or from pdaggerq, fails here; re-record only after
checking the new modules against the old ones.

The same two hashes came out of a second pdaggerq build (Python 3.11 on a
Linux cluster). Larger modules do not: the ADC(4)
l3 Laplace module's term list differs between those two builds while every
block agrees with its dense twin to 1.6e-11, so a text hash there would pin the
build, not the physics; tests/test_l3_laplace.py checks it numerically instead.

The hashes were re-recorded when the spin-orbital and restricted twins moved to
one block emitter with the codegen_common header: the new modules are the old
ones up to the header comment, the docstrings and the order of the functions --
every function has the same name and the same AST body.
They were re-recorded again when cached_einsum moved to src.Base.utils: the new
modules are the old ones up to their einsum import line. A generatedADC/ cache
written before that move imports the old path and must be deleted and regenerated.
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
        '575e314d43eca32523ab6e3a6a437397310a972b0e38279316e86e2965ec6ff9',
    'adc2_blocks_restricted.py':
        '468ed44d382ad8b3b22785527f46829e13518763613798922a9fa8dbbf3da495',
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
