"""cp2k_basis: the CP2K files parse, the names build the same sets, the guards fire.

Reads the CP2K files through src.Base.basis.cp2k_basis from MBPT_CP2K_DATA or the
cache and never fetches them; without either it prints SKIPPED and exits 0 with no
verdict. One check requests a CP2K commit that does not exist, which fails with a
404 online and a network error offline, and passes either way.

Checks, in order:
  parsing    a block is found by its header name; every orbital block and RI
             tier builds with the count its header implies (per set and l,
             contractions times 2l + 1; C aug-SZV-MOLOPT-ae is STO-6G + 1s + 1p
             + 1d = 3s2p1d = 14)
  tier rule  threshold, tightest by default, min_lmax, a tier's own Delta-I
  guards     unknown element, non-numeric data line, offline, failed download
  register   names build the dicts bit for bit; one RI name per set of tiers; a
             threshold call leaves `-ri` alone; elements outside a set and bad
             arguments raise; one warning with the Delta-I an element gets; other
             data are refused by element, comments and spacing pass
"""
import contextlib
import os
import re
import shutil
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyscf import gto
from pyscf.gto import basis as pyscf_basis
from pyscf.gto.basis import parse_cp2k
from pyscf.lib.exceptions import BasisNotFoundError

from src.Base.basis import cp2k_basis as cb


def check(ok, label, detail=''):
    """Print one verdict line and return the verdict."""
    tag = 'ok' if ok else 'FAIL'
    print(f'  [{tag}] {label}' + (f'   ({detail})' if detail else ''))
    return bool(ok)


def nao(element, basis):
    """Function count PySCF builds for one atom in an internal-format basis."""
    return gto.M(atom=f'{element} 0 0 0', basis={element: basis}, verbose=0,
                 spin=gto.charge(element) % 2).nao


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

    # pyscf's own loader has no lookup by block name, which is why this module exists.
    try:
        first = str(nao('C', parse_cp2k.load(orbital, 'C')))
    except BasisNotFoundError as exc:
        first = type(exc).__name__
    ours = nao('C', cb.load_basis('aug-SZV-MOLOPT-ae', ['C'], orbital)['C'])
    all_ok &= check(ours == 14 and first != '14',
                    'C aug-SZV-MOLOPT-ae by name is 14 functions; pyscf load() raises '
                    'or differs', f'ours {ours}, load() {first}')
    ae = cb.basis_block('aug-SZV-MOLOPT-ae', 'C', orbital)
    all_ok &= check(cb.pattern(ae) == '3s2p1d', 'C recipe STO-6G + 1s + 1p + 1d',
                    cb.pattern(ae))
    # Same shape, different block: the name decides, not the shape.
    sr =cb.basis_block('aug-SZV-MOLOPT-ae-SR', 'C', orbital)
    nprim = lambda block: int(block[2].split()[3])
    all_ok &= check(ae != sr and cb.pattern(sr) == cb.pattern(ae)
                    and nprim(ae) == 6 and nprim(sr) == 3,
                    'C aug-SZV-MOLOPT-ae and -SR are distinct blocks of equal shape',
                    f'first-shell primitives {nprim(ae)} and {nprim(sr)}')
    # H's header names both sets; both resolve to the one block.
    h_ae =cb.basis_block('aug-SZV-MOLOPT-ae', 'H', orbital)
    h_mini = cb.basis_block('aug-SZV-MOLOPT-ae-mini', 'H', orbital)
    all_ok &= check(h_ae == h_mini and cb.nao_from_block(h_ae) == 6,
                    'H aug-SZV-MOLOPT-ae and -mini share one 6-function block')

    # Every orbital block builds with the count its set lines imply.
    bad, nblocks = 0, 0
    for el, _, lines in cb._blocks(orbital):
        bad += nao(el, cb.parse('\n'.join(lines))) != cb.nao_from_block(lines)
        nblocks += 1
    all_ok &= check(bad == 0 and nblocks >= 70,
                    'every orbital block of every element builds with its count',
                    f'{nblocks} blocks, {bad} mismatches')

    # Every RI block too, plus the count in the tier's name.
    bad, ntiers = 0, 0
    for el, names, lines in cb._blocks(ri):
        n_name = [int(m['n']) for n in names if (m := cb._RI_NAME.match(n))]
        got = nao(el, cb.parse('\n'.join(lines)))
        bad += got != cb.nao_from_block(lines) or got not in n_name
        ntiers += 1
    all_ok &= check(bad == 0 and ntiers >= 500,
                    'every RI tier of every element builds with the count in its name',
                    f'{ntiers} tiers, {bad} mismatches')

    # The tier rule: smallest within a threshold, tightest without, min_lmax, the edge.
    tier =cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'C', 1e-4, path=ri)
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
    all_ok &= check(tier[1] == 78 and 'g' not in tier[3] and len(caught) == 1,
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

    # Data guards: missing element, non-numeric line (pyscf would eval() it),
    # offline mode, failed fetch.
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
    all_ok &= check(ok_fetch and 'MBPT_CP2K_DATA' in msg_fetch and 'fetch' in msg_fetch,
                    'a failed download raises OSError naming the way out',
                    msg_fetch[:60])

    # A registered name builds exactly what the dict builds, for every set.
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

    # One RI name per set of tiers: 8e-5 and 1e-4 pick the same tiers and share it.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        names = {me: cb.register('aug-SZV-MOLOPT-ae', me)[1]
                 for me in (None, 0, 8e-5, 1e-4, 1e-5, 1, float('inf'))}
    all_ok &= check(names[None] == names[0] == 'aug-SZV-MOLOPT-ae-ri'
                    and names[8e-5] == names[1e-4] == 'aug-SZV-MOLOPT-ae-ri-7.3e-05'
                    and names[1] == names[float('inf')]
                    == 'aug-SZV-MOLOPT-ae-ri-4.8e-02'
                    and names[1e-5] != names[1e-4],
                    'one RI name per set of tiers, the smallest threshold picking it',
                    f'{names[1e-4]} for 8e-5 and 1e-4; {names[1]} for 1 and inf')
    # A threshold call leaves -ri alone, so a route that defaults to it raises.
    key =pyscf_basis._format_basis_name('aug-SZV-MOLOPT-ae-ri')
    pyscf_basis.USER_BASIS_ALIAS.pop(key, None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb.register('aug-SZV-MOLOPT-ae', max_error=1e-4)
        ok_default, msg_default = raises(BasisNotFoundError, lambda: gto.M(
            atom='O 0 0 0', basis='aug-SZV-MOLOPT-ae-ri', verbose=0),
            'aug-SZV-MOLOPT-ae-ri')
    all_ok &= check(key not in pyscf_basis.USER_BASIS_ALIAS and ok_default,
                    'a threshold call leaves -ri unregistered, so a default raises',
                    msg_default.split('\n')[0][:50])

    # An element outside a set raises; CP2K has no -mini RI tiers for H and He.
    mini ='aug-SZV-MOLOPT-ae-mini'
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

    # Bad arguments raise.
    ok_name,msg_name = raises(ValueError, lambda: cb.register('aug-cc-pVDZ'),
                               'is not one of')
    ok_neg, msg_neg = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae',
                                                             max_error=-1e-4),
                             'threshold')
    ok_nan, _ = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae',
                                                       max_error=float('nan')),
                       'threshold')
    all_ok &= check(ok_name and ok_neg and ok_nan,
                    'register rejects an unknown set and a negative or nan max_error',
                    f'{msg_name[:40]}; {msg_neg[:40]}')

    # One warning names the elements outside the threshold and the Delta-I they get.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cb.register('aug-SZV-MOLOPT-ae', max_error=1e-4)
    msg = str(caught[0].message) if caught else 'no warning'
    all_ok &= check(len(caught) == 1
                    and msg.endswith('for S (tightest tier, Delta-I 1.9e-04)'),
                    'register warns once, naming the element and the Delta-I it gets',
                    msg)
    # A threshold below every tier is one line, and the -ri set.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, ri_floor = cb.register('aug-SZV-MOLOPT-ae', max_error=1e-9)
    msg = str(caught[0].message) if caught else 'no warning'
    all_ok &= check(len(caught) == 1 and ri_floor == 'aug-SZV-MOLOPT-ae-ri'
                    and 'tightest tier of every element' in msg and 'for' not in msg,
                    'a threshold below every tier warns once and gives -ri', msg)

    # Other data are refused naming the element; comments and spacing pass.
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(orbital, tmp)
        copy = os.path.join(tmp, cb.FILES['ri'])
        with open(ri) as fh:
            text = fh.read()
        spaced = re.sub(r'(?m)^(\d.*)$', lambda m: ' '.join(m.group(1).split()), text)
        with open(copy, 'w') as fh:
            fh.write(spaced.replace('\n', '   \n', 20) + '# not the pinned file\n')
        with environ(MBPT_CP2K_DATA=tmp), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, msg_pass = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae'),
                                 '')
        with open(copy, 'w') as fh:
            fh.write('\n'.join('\n'.join(lines) for el, _, lines in cb._blocks(ri)
                               if el != 'Cl') + '\n')
        with environ(MBPT_CP2K_DATA=tmp):
            refused_cl, _ = raises(ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae'),
                                   f'{copy}: the blocks of Cl differ')
        tier = cb.pick_ri_tier('aug-SZV-MOLOPT-ae', 'C', path=ri)[0]
        exponent = cb.basis_block(tier, 'C', ri)[3].split()[0]
        start = re.search(rf'^C\s+{re.escape(tier)}\b', text, re.M).end()
        pos = text.index(exponent, start)
        digit = '7' if exponent[-1] != '7' else '3'
        with open(copy, 'w') as fh:
            fh.write(text[:pos] + exponent[:-1] + digit + text[pos + len(exponent):])
        with environ(MBPT_CP2K_DATA=tmp):
            refused, msg_refused = raises(
                ValueError, lambda: cb.register('aug-SZV-MOLOPT-ae'),
                f'{copy}: the blocks of C differ from the pinned CP2K commit')
    all_ok &= check(msg_pass == 'nothing raised' and refused and refused_cl,
                    'register passes comments and spacing, refuses changed or missing '
                    'data by element',
                    f'comment copy: {msg_pass}; exponent copy: {msg_refused[:60]}')
    prov = cb.provenance('orbital', orbital)
    local = bool(os.environ.get('MBPT_CP2K_DATA'))
    all_ok &= check((prov['commit'] is not None and prov['blocks'] is not None)
                    or local, 'orbital file matches the pinned CP2K commit, file and '
                    'blocks', prov['sha256'][:12])

    print('\nALL PASSED' if all_ok else '\nFAILURES DETECTED')
    sys.exit(0 if all_ok else 1)
