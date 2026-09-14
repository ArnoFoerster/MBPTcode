"""cp2k_basis: every aug-MOLOPT block PySCF builds has the count its header implies.

Reads the CP2K files through src.Base.basis.cp2k_basis from MBPT_CP2K_DATA or the
cache, never downloading; without either it prints SKIPPED and exits 0 with no
verdict.
Counts: sum over sets and l of contractions times 2l + 1, and the header's own
recipe for C aug-SZV-MOLOPT-ae, 'STO-6G + 1s + 1p + 1d' = 3s2p1d = 14.
"""
import contextlib
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np

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


def raises(exc, call, match):
    """(True, message) when `call()` raises `exc` with `match` in its message."""
    try:
        call()
    except exc as err:
        return match in str(err), str(err)
    return False, 'nothing raised'


@contextlib.contextmanager
def environ(**values):
    """Set environment variables for the block; None unsets one."""
    def apply(settings):
        for k, v in settings.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    saved = {k: os.environ.get(k) for k in values}
    try:
        apply(values)
        yield
    finally:
        apply(saved)


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
    tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'O', path=ri)
    tightest = min(cb.ri_tiers('aug-SZV-MOLOPT-ae', 'O', ri), key=lambda t: t[2])
    all_ok &= check(tier == tightest and tier[1] == 108,
                    'O aug-SZV-MOLOPT-ae default tier is the tightest, 108 functions',
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
    tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'C', 1.7e-4, path=ri)
    all_ok &= check(tier[1] == 47 and tier[2] == 1.7e-4,
                    'max_error equal to a tier\'s Delta-I includes that tier',
                    f'C at 1.7e-4: {tier[1]} functions')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        o_tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'O', min_lmax=4, path=ri)
        n_tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'N', min_lmax=4, path=ri)
    msg = str(caught[0].message) if caught else 'no warning'
    all_ok &= check(o_tier[1] == 108 and n_tier[1] == 78 and len(caught) == 1
                    and 'l_max >= 4' in msg and 'Delta-I' not in msg,
                    'min_lmax without max_error: tightest with g, else a warning for N',
                    f'O {o_tier[1]}, N {n_tier[1]}; {msg}')
    try:
        cb.basis_block('aug-SZV-MOLOPT-ae', 'Xx', orbital)
        raised = False
    except ValueError:
        raised = True
    all_ok &= check(raised, 'an element without a block raises ValueError')
    block = list(cb.basis_block('aug-SZV-MOLOPT-ae', 'C', orbital))
    block[3] = '__import__("os") ' + ' '.join(block[3].split()[1:])
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, 'BASIS_BAD')
        with open(bad, 'w') as fh:
            fh.write('\n'.join(block) + '\n')
        ok, msg = raises(ValueError,
                         lambda: cb.basis_block('aug-SZV-MOLOPT-ae', 'C', bad),
                         'non-numeric')
    all_ok &= check(ok, 'a non-numeric data line raises before pyscf can eval it', msg)
    with tempfile.TemporaryDirectory() as tmp, \
            environ(MBPT_CP2K_DATA=None, MBPT_CP2K_CACHE=tmp):
        ok_offline, msg_offline = raises(
            FileNotFoundError, lambda: cb.data_file('ri', download=False),
            'download=False')
        ok_fetch, msg_fetch = raises(
            OSError, lambda: cb.data_file('ri', commit='0' * 40), 'could not download')
    all_ok &= check(ok_offline, 'an uncached file with download=False raises',
                    msg_offline[:60])
    all_ok &= check(ok_fetch, 'a failed download raises OSError naming the way out',
                    msg_fetch[:60])

    # A registered name must build exactly what the dict builds, element by element,
    # for every set: the tightest RI name, and one threshold name.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                # S has no tier within 1e-4
        _, ri_loose = cb.register('aug-SZV-MOLOPT-ae', max_error=1e-4)
        same, counted = True, {'orbital': 0, 'ri': 0, 'ri 1e-4': 0}
        for basis_name in cb.BASIS_NAMES:
            _, ri_name = cb.register(basis_name)
            for el in sorted({el for el, names, _ in cb._blocks(orbital)
                              if basis_name in names}):
                cases = [('orbital', basis_name,
                          cb.load_basis(basis_name, [el], orbital)[el])]
                if cb.ri_tiers(basis_name, el, ri):
                    cases.append(('ri', ri_name,
                                  cb.load_ri_basis(basis_name, [el], path=ri)[el]))
                if basis_name == 'aug-SZV-MOLOPT-ae':
                    cases.append(('ri 1e-4', ri_loose, cb.load_ri_basis(
                        basis_name, [el], 1e-4, path=ri)[el]))
                spin = gto.charge(el) % 2
                for kind, name, bas in cases:
                    a = gto.M(atom=f'{el} 0 0 0', basis=name, spin=spin, verbose=0)
                    b = gto.M(atom=f'{el} 0 0 0', basis={el: bas}, spin=spin,
                              verbose=0)
                    same &= (np.array_equal(a._env, b._env)
                             and np.array_equal(a._bas, b._bas))
                    counted[kind] += 1
    all_ok &= check(same and counted == {'orbital': 74, 'ri': 72, 'ri 1e-4': 17},
                    'registered names build the same basis as the dicts, bit for bit',
                    ', '.join(f'{v} {k}' for k, v in counted.items()))
    ri_name = cb.register('aug-SZV-MOLOPT-ae')[1]
    all_ok &= check(ri_name == 'aug-SZV-MOLOPT-ae-ri'
                    and ri_loose == 'aug-SZV-MOLOPT-ae-ri-0.0001',
                    'the RI name encodes its tier rule', f'{ri_name}, {ri_loose}')
    mini = 'aug-SZV-MOLOPT-ae-mini'
    no_ri = [el for el in sorted({el for el, names, _ in cb._blocks(orbital)
                                  if mini in names}) if not cb.ri_tiers(mini, el, ri)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                # pyscf's basis-set-exchange hint
        ok_k, msg_k = raises(BasisNotFoundError, lambda: gto.M(
            atom='K 0 0 0', basis='aug-SZV-MOLOPT-ae', spin=1, verbose=0), 'not found')
        ok_mini, msg_mini = raises(BasisNotFoundError, lambda: gto.M(
            atom=f'{no_ri[0]} 0 0 0', basis=f'{mini}-ri',
            spin=gto.charge(no_ri[0]) % 2, verbose=0), 'not found')
    all_ok &= check(ok_k and ok_mini and len(no_ri) == 2,
                    'an element outside a registered set raises BasisNotFoundError',
                    f'K in aug-SZV-MOLOPT-ae; {no_ri[0]} in {mini}-ri')
    ok_name, msg_name = raises(ValueError, lambda: cb.register('aug-cc-pVDZ'),
                               'is not one of')
    ok_err, msg_err = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae',
                                                             max_error=1.0),
                             'threshold')
    all_ok &= check(ok_name and ok_err,
                    'register rejects an unknown set and max_error outside (0, 1)',
                    f'{msg_name[:40]}; {msg_err[:40]}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cb.register('aug-SZV-MOLOPT-ae', max_error=1e-4)
    msg = str(caught[0].message) if caught else 'no warning'
    all_ok &= check(len(caught) == 1 and 'for S,' in msg,
                    'register warns once, naming the elements outside max_error', msg)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(orbital, tmp)
        with open(ri) as src, open(os.path.join(tmp, cb.FILES['ri']), 'w') as dst:
            dst.write(src.read() + '# not the pinned file\n')
        with environ(MBPT_CP2K_DATA=tmp):
            refused, msg = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae'),
                                  'pinned CP2K commit')
    all_ok &= check(refused, 'register refuses data other than the pinned commit')
    prov = cb.provenance('orbital', orbital)
    local = bool(os.environ.get('MBPT_CP2K_DATA'))
    all_ok &= check(prov['commit'] is not None or local,
                    'orbital file matches the pinned CP2K commit', prov['sha256'][:12])

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
