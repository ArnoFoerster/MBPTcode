"""The code generators are optional and stop at ADC(5).

Checks, none of which needs pdaggerq:
  1. every module of src/CodeGeneration and src/CodeGeneration/pdaggerq_interface.py
     imports with pdaggerq absent (simulated by MBPT_NO_PDAGGERQ=1 in a child
     process), and pq_helper then raises ImportError instead of failing later;
  2. every producer the generation REGISTRY dispatches (dotted names,
     resolved only when generation runs) exists;
  3. the order-general emitters refuse ADC(6) and above before deriving
     anything: the U block, the non-Dyson and union cross blocks, the
     generation plan, the diagram engine and enumerator, and the MP5 density
     and amplitudes (the ADC(6) static self-energy input).
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import pytest

from src.SingleReference.ADC.adc_order_rule import MAX_ADC_ORDER

_CHILD = r'''
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
names = ['src.CodeGeneration.pdaggerq_interface'] + [
    '.'.join(f.relative_to(root).with_suffix('').parts)
    for f in sorted((root / 'src' / 'CodeGeneration').rglob('*.py'))]
for name in names:
    importlib.import_module(name)
from src.CodeGeneration.pdaggerq_interface import pdaggerq_available, pq_helper
assert not pdaggerq_available()
try:
    pq_helper('fermi')
except ImportError:
    print('IMPORTED', len(names))
'''


def test_generators_import_without_pdaggerq():
    env = dict(os.environ, MBPT_NO_PDAGGERQ='1')
    out = subprocess.run([sys.executable, '-c', _CHILD, ROOT], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert 'IMPORTED' in out.stdout, out.stdout


def test_every_registered_producer_exists():
    from src.CodeGeneration import generate_adc
    calls = {d for reg in generate_adc.REGISTRY.values() for d, _ in reg.get('calls', [])}
    assert calls
    missing = []
    for dotted in sorted(calls):
        try:
            generate_adc._resolve_call(dotted)
        except AttributeError:
            missing.append(dotted)
    assert not missing, missing


def test_the_ceiling_is_adc5():
    assert MAX_ADC_ORDER == 5


@pytest.mark.parametrize('call', [
    lambda: __import__('src.CodeGeneration.generate_adcn', fromlist=['x'])
    .generate_adc_u(5, restricted=True),
    lambda: __import__('src.CodeGeneration.generate_adcn', fromlist=['x'])
    .generate_nondyson_pair(2, 2, 2),
    lambda: __import__('src.CodeGeneration.generate_adcn', fromlist=['x'])
    .generate_adc_cross_pair(4, 'k1k1'),
    lambda: __import__('src.CodeGeneration.sigma_codegen', fromlist=['x'])
    .generate_adc_cross_pair_apply_restricted(2, 'k2k2'),
    lambda: __import__('src.CodeGeneration.generate_adc', fromlist=['x']).plan(6),
    lambda: __import__('src.CodeGeneration.generate_adc', fromlist=['x']).nondyson_plan(6),
    lambda: __import__('src.CodeGeneration.diagrammatic.skeleton_enum',
                       fromlist=['x']).kept_topologies(6),
    lambda: __import__('src.CodeGeneration.diagrammatic.goldstone',
                       fromlist=['x']).build_skeletons(6, 'oo'),
    lambda: __import__('src.CodeGeneration.generate_mpn_density',
                       fromlist=['x']).density_pieces_for_order(5),
    lambda: __import__('src.CodeGeneration.generate_mpn_density', fromlist=['x'])
    .generate_amplitude_numerator_module([5], 'never_written'),
    lambda: __import__('src.CodeGeneration.generate_adcn', fromlist=['x'])
    .generate_adc_u_k2((4,)),
    lambda: __import__('src.CodeGeneration.generate_adcn', fromlist=['x'])
    .generate_adc_class_pair(2, 'k2k2'),
    lambda: __import__('src.CodeGeneration.sigma_codegen', fromlist=['x'])
    .generate_adc_u_k2_apply_restricted(orders=(1, 2, 3, 4)),
], ids=['u_restricted', 'nondyson_pair', 'cross_pair', 'cross_pair_apply', 'plan',
        'nondyson_plan', 'skeletons', 'goldstone', 'mp5_density', 'mp5_amplitudes',
        'u_k2', 'class_pair', 'u_k2_apply'])
def test_adc6_is_refused(call, tmp_path, monkeypatch):
    from src.CodeGeneration import codegen_common
    monkeypatch.setattr(codegen_common, 'ADC_OUT_DIR', str(tmp_path))
    with pytest.raises(ValueError, match='ADC\\(5\\)'):
        call()
    assert not any(tmp_path.iterdir())


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
