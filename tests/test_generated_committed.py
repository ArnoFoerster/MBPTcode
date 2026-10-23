"""Every committed generated module is what its generator writes today.

The generated modules under src/SingleReference/CC/generated_*,
src/SingleReference/DensityMatrix/generated_mpn* and src/MultiReference/QDPT
are committed, so a generator change that is not regenerated (or a hand edit of
a generated file) leaves code nothing reproduces. Each generator here writes into
a temporary directory and its output is compared byte for byte with the
committed file. A deliberate change regenerates the committed file in the same
commit.

KNOWN_STALE: committed modules written by an older emitter that their generator
no longer reproduces -- the MP3/MP4 Laplace pieces predate laplace_codegen's
flatten-p emission (one einsum per term with the quadrature as a batch axis;
the committed files loop over tau). Regenerating them changes the summation
order, so it is a numerics decision of its own; they are strict xfails here, so
the regeneration that brings them in line also has to drop them from the list.

The comparison is byte for byte, so it holds only on the pdaggerq build the
committed files were recorded with: the workstation's conda env py310 (Python
3.10, pdaggerq 0.0.1, macOS arm64). Other builds order the factors of a term
differently (a cluster's py311 build writes t1 before l1, and reorders terms); there
the base generators miss the same files, so this is a build pin, not a defect.
_recording_build() checks pdaggerq's factor order on one small density term and
skips the test on any other build.

Needs pdaggerq (skipped without it); about 8 minutes at 2 threads.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import pytest

from src.CodeGeneration.pdaggerq_interface import pdaggerq_available, pq_helper

# pdaggerq's terms for <0|L1 e1(m,n) e^T|0>, alpha spin, on the recording build
RECORDED_PROBE = [['-1.00', 'l1_aa(n,a)', 't1_aa(a,m)']]


def _recording_build():
    """True if this pdaggerq orders factors as the build the committed files were written with."""
    pq = pq_helper('fermi')
    pq.set_left_operators([['l1']])
    pq.add_st_operator(1.0, ['e1(m,n)'], ['t1', 't2', 't3'])
    pq.simplify()
    terms = pq.strings(spin_labels={'m': 'a', 'n': 'a'})
    pq.clear()
    return terms == RECORDED_PROBE


pytestmark = [
    pytest.mark.skipif(not pdaggerq_available(), reason='the generators need pdaggerq (optional)'),
    pytest.mark.skipif(pdaggerq_available() and not _recording_build(),
                       reason='this pdaggerq build orders factors unlike the recording build'),
]

_CC = os.path.join(ROOT, 'src', 'SingleReference', 'CC')
_DM = os.path.join(ROOT, 'src', 'SingleReference', 'DensityMatrix')
_QDPT = os.path.join(ROOT, 'src', 'MultiReference', 'QDPT')

KNOWN_STALE = {
    'generated_mpn_restricted/mp3_t3_laplace_restricted.py',
    'generated_mpn_restricted/mp3_t3_laplace_restricted_df.py',
    'generated_mpn_restricted/mp4_laplace_restricted.py',
    'generated_mpn_restricted/mp4_t1_4_laplace_restricted.py',
    'generated_mpn_restricted/mp4_t2_3_laplace_restricted.py',
    'generated_mpn_unrestricted/mp3_t3_laplace_unrestricted.py',
    'generated_mpn_unrestricted/mp3_t3_laplace_unrestricted_df.py',
}


def _with_global(module, name, value, run):
    """run() with module.<name> = value, restored afterwards."""
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        run()
    finally:
        setattr(module, name, saved)


def _run_eom(out):
    from src.CodeGeneration import generate_eom_ccsdt as gen
    _with_global(gen, 'OUT_DIR', os.path.join(out, 'generated_eom'), gen.generate)


def _run_ccsdt(out):
    from src.CodeGeneration import generate_restricted_ccsdt as gen
    _with_global(gen, 'OUT_DIR', os.path.join(out, 'generated_restricted'), gen.generate)


def _run_mpn(out):
    from src.CodeGeneration import codegen_common, generate_mpn_density, generate_mpn_density_spin, generate_mpn_laplace
    _with_global(generate_mpn_density, 'OUT_DIR', os.path.join(out, 'generated_mpn'),
                 generate_mpn_density.generate)

    def spin():
        generate_mpn_density_spin.generate_restricted()
        generate_mpn_density_spin.generate_unrestricted()
    _with_global(codegen_common, 'DM_OUT_DIR', out, spin)
    _with_global(codegen_common, 'DM_OUT_DIR', out,
                 lambda: [generate_mpn_laplace.generate(m) for m in generate_mpn_laplace.MODULES])


def _run_qdpt(out):
    from src.CodeGeneration import generate_qdpt_evaluators
    generate_qdpt_evaluators.generate(out)


# (generator, committed directory, files it writes relative to that directory)
CASES = {
    'eom_ccsdt': (_run_eom, _CC, ['generated_eom/ip_sigma.py', 'generated_eom/ea_sigma.py',
                                  'generated_eom/ee_sigma.py', 'generated_eom/ee_density.py']),
    'restricted_ccsdt': (_run_ccsdt, _CC, ['generated_restricted/amplitudes_restricted.py',
                                          'generated_restricted/lambda_residual_restricted.py',
                                          'generated_restricted/d1_blocks_restricted.py']),
    'mpn': (_run_mpn, _DM, ['generated_mpn/mpn_density_pieces.py',
                            'generated_mpn_restricted/mpn_density_pieces_restricted.py',
                            'generated_mpn_restricted/mpn_density_pieces_restricted_df.py',
                            'generated_mpn_unrestricted/mpn_density_pieces_unrestricted.py',
                            'generated_mpn_unrestricted/mpn_density_pieces_unrestricted_df.py']
            + sorted(KNOWN_STALE)),
    'qdpt': (_run_qdpt, _QDPT, ['generated_evaluators.py', 'generated_evaluators_spinfree.py']),
}


@pytest.fixture(scope='module')
def regenerated(tmp_path_factory):
    """case -> temporary directory its generator wrote into (run on first use)."""
    done = {}

    def get(case):
        if case not in done:
            out = tmp_path_factory.mktemp(case)
            CASES[case][0](str(out))
            done[case] = str(out)
        return done[case]
    return get


PARAMS = [pytest.param(case, rel, marks=[pytest.mark.xfail(
              strict=True, reason='committed by an older emitter (module docstring)')]
              if rel in KNOWN_STALE else [], id=f'{case}:{rel}')
          for case, (_, _, files) in CASES.items() for rel in files]


@pytest.mark.parametrize('case, rel', PARAMS)
def test_committed_module_regenerates(regenerated, case, rel):
    _, committed, _ = CASES[case]
    with open(os.path.join(regenerated(case), rel), 'rb') as fh:
        new = fh.read()
    with open(os.path.join(committed, rel), 'rb') as fh:
        old = fh.read()
    assert new == old, f'{rel}: the generator no longer writes the committed file'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
