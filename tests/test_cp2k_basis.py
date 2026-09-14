"""cp2k_basis: every aug-MOLOPT block PySCF builds has the count its header implies.

Reads the CP2K files through src.Base.basis.cp2k_basis from MBPT_CP2K_DATA or the
cache, never downloading; without either it prints SKIPPED and exits 0 with no
verdict.
Counts: sum over sets and l of contractions times 2l + 1, and the header's own
recipe for C aug-SZV-MOLOPT-ae, 'STO-6G + 1s + 1p + 1d' = 3s2p1d = 14.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto
from pyscf.gto.basis import parse_cp2k
from pyscf.lib.exceptions import BasisNotFoundError

from src.Base.basis import cp2k_basis as cb


def check(ok, label, detail=''):
    tag = 'ok' if ok else 'FAIL'
    print(f'  [{tag}] {label}' + (f'   ({detail})' if detail else ''))
    return bool(ok)


def nao(element, basis):
    return gto.M(atom=f'{element} 0 0 0', basis={element: basis}, verbose=0,
                 spin=gto.charge(element) % 2).nao


def elements_in(path):
    return sorted({el for el, _, _ in cb._blocks(path)})


if __name__ == '__main__':
    try:
        orbital = cb.data_file('orbital', download=False)
        ri = cb.data_file('ri', download=False)
    except (OSError, ValueError) as exc:
        print(f'SKIPPED: no CP2K data ({exc}); set MBPT_CP2K_DATA or run '
              '"python -m src.Base.basis.cp2k_basis fetch" once')
        sys.exit(0)
    all_ok = True

    # The reason the module exists: pyscf's own loader splits a file on
    # `# BASIS SET` delimiters, which these files do not carry, and where a file
    # has them it takes the first block for the element, never a named one.
    try:
        first = str(nao('C', parse_cp2k.load(orbital, 'C')))
    except BasisNotFoundError as exc:
        first = type(exc).__name__
    ours = nao('C', cb.load_basis('aug-SZV-MOLOPT-ae', ['C'], orbital)['C'])
    all_ok &= check(ours == 14 and first != '14',
                    'C aug-SZV-MOLOPT-ae by name is 14 functions, pyscf load() is not',
                    f'ours {ours}, load() {first}')
    ae = cb.basis_block('aug-SZV-MOLOPT-ae', 'C', orbital)
    all_ok &= check(cb.pattern(ae) == '3s2p1d', 'C recipe STO-6G + 1s + 1p + 1d',
                    cb.pattern(ae))
    # Same count and pattern, different block: the SR set's first s shell has 3
    # primitives where the STO-6G one has 6, so the name and not the shape decides.
    sr = cb.basis_block('aug-SZV-MOLOPT-ae-SR', 'C', orbital)
    nprim = lambda block: int(block[2].split()[3])
    all_ok &= check(ae != sr and nprim(ae) == 6 and nprim(sr) == 3,
                    'C aug-SZV-MOLOPT-ae and -SR are distinct blocks of equal shape',
                    f'first-shell primitives {nprim(ae)} and {nprim(sr)}')
    # H's header carries two names; both must resolve to the same block.
    h_ae = cb.basis_block('aug-SZV-MOLOPT-ae', 'H', orbital)
    h_mini = cb.basis_block('aug-SZV-MOLOPT-ae-mini', 'H', orbital)
    all_ok &= check(h_ae == h_mini and cb.nao_from_block(h_ae) == 6,
                    'H aug-SZV-MOLOPT-ae and -mini share one 6-function block')

    bad, nblocks = 0, 0
    for name in cb.BASIS_NAMES:
        for el in elements_in(orbital):
            try:
                lines = cb.basis_block(name, el, orbital)
            except ValueError:
                continue                       # the name has no block for this element
            bad += nao(el, cb.parse('\n'.join(lines))) != cb.nao_from_block(lines)
            nblocks += 1
    all_ok &= check(bad == 0 and nblocks >= 70,
                    'every orbital block of every element builds with its count',
                    f'{nblocks} blocks, {bad} mismatches')

    bad, ntiers = 0, 0
    for name in cb.BASIS_NAMES:
        for el in elements_in(ri):
            for ri_name, n_name, err, pat in cb.ri_tiers(name, el, ri):
                lines = cb.basis_block(ri_name, el, ri)
                got = nao(el, cb.parse('\n'.join(lines)))
                bad += got != n_name or got != cb.nao_from_block(lines)
                ntiers += 1
    all_ok &= check(bad == 0 and ntiers >= 500,
                    'every RI tier of every element builds with the count in its name',
                    f'{ntiers} tiers, {bad} mismatches')

    tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'C', 1e-4, path=ri)
    all_ok &= check(tier[1] == 48 and tier[2] <= 1e-4,
                    'C aug-SZV-MOLOPT-ae tier at Delta-I 1e-4 is the 48-function set',
                    f'{tier[1]} functions, Delta-I {tier[2]:.1e}')
    tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'H', 1e-4, min_lmax=2, path=ri)
    all_ok &= check(tier[1] == 20 and 'd' in tier[3],
                    'H tier at Delta-I 1e-4 with l_max >= 2 is the 20-function set',
                    f'{tier[1]} functions, {tier[3]}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'N', 1e-4, min_lmax=4, path=ri)
    all_ok &= check(tier[1] == 78 and len(caught) == 1,
                    'N has no g tier: the tightest one is returned with a warning',
                    f'{tier[1]} functions, {len(caught)} warning')
    try:
        cb.basis_block('aug-SZV-MOLOPT-ae', 'Xx', orbital)
        raised = False
    except ValueError:
        raised = True
    all_ok &= check(raised, 'an element without a block raises ValueError')
    prov = cb.provenance('orbital', orbital)
    local = bool(os.environ.get('MBPT_CP2K_DATA'))
    all_ok &= check(prov['commit'] is not None or local,
                    'orbital file matches the pinned CP2K commit', prov['sha256'][:12])

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
