"""cp2k_basis: every aug-MOLOPT block PySCF builds has the count its header implies.

Reads the CP2K files through src.Base.cp2k_basis (MBPT_CP2K_DATA, the cache, or a
download); without any of these it prints SKIPPED and exits 0 with no verdict.
Counts: sum over sets and l of contractions times 2l + 1, and the header's own
recipe for C aug-SZV-MOLOPT-ae, 'STO-6G + 1s + 1p + 1d' = 3s2p1d = 14.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto
from pyscf.gto.basis import parse_cp2k
from pyscf.lib.exceptions import BasisNotFoundError

from src.Base import cp2k_basis as cb


def check(ok, label, detail=''):
    tag = 'ok' if ok else 'FAIL'
    print(f'  [{tag}] {label}' + (f'   ({detail})' if detail else ''))
    return bool(ok)


def nao(element, basis):
    return gto.M(atom=f'{element} 0 0 0', basis={element: basis}, verbose=0,
                 spin=gto.charge(element) % 2).nao


if __name__ == '__main__':
    try:
        orbital, ri = cb.data_file('orbital'), cb.data_file('ri')
    except (OSError, ValueError) as exc:
        print(f'SKIPPED: no CP2K data ({exc}); set MBPT_CP2K_DATA or allow a download')
        sys.exit(0)
    elements = ['H', 'C', 'N', 'O']
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
    lines = cb.basis_block('aug-SZV-MOLOPT-ae', 'C', orbital)
    all_ok &= check(cb.pattern(lines) == '3s2p1d', 'C recipe STO-6G + 1s + 1p + 1d',
                    cb.pattern(lines))
    # H's header carries two names; both must resolve to the same block.
    h_ae = cb.basis_block('aug-SZV-MOLOPT-ae', 'H', orbital)
    h_mini = cb.basis_block('aug-SZV-MOLOPT-ae-mini', 'H', orbital)
    all_ok &= check(h_ae == h_mini and cb.nao_from_block(h_ae) == 6,
                    'H aug-SZV-MOLOPT-ae and -mini share one 6-function block')

    bad = 0
    for name in cb.BASIS_NAMES:
        for el in elements:
            lines = cb.basis_block(name, el, orbital)
            got = nao(el, cb.parse('\n'.join(lines)))
            bad += got != cb.nao_from_block(lines)
    all_ok &= check(bad == 0, 'every orbital block builds with its implied count',
                    f'{len(cb.BASIS_NAMES) * len(elements)} blocks, {bad} mismatches')

    bad, ntiers = 0, 0
    for name in cb.BASIS_NAMES:
        for el in elements:
            for ri_name, n_name, err, pat in cb.ri_tiers(name, el, ri):
                lines = cb.basis_block(ri_name, el, ri)
                got = nao(el, cb.parse('\n'.join(lines)))
                bad += got != n_name or got != cb.nao_from_block(lines)
                ntiers += 1
    all_ok &= check(bad == 0, 'every RI tier builds with the count in its name',
                    f'{ntiers} tiers, {bad} mismatches')
    tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'C', 1e-4, ri)
    all_ok &= check(tier[1] == 48 and tier[2] <= 1e-4,
                    'C aug-SZV-MOLOPT-ae tier at Delta-I 1e-4 is the 48-function set',
                    f'{tier[1]} functions, Delta-I {tier[2]:.1e}')
    prov = cb.provenance('orbital', orbital)
    all_ok &= check(prov['commit'] is not None or os.environ.get('MBPT_CP2K_DATA'),
                    'orbital file matches the pinned CP2K commit', prov['sha256'][:12])

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
