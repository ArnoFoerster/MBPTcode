"""CP2K's aug-MOLOPT basis families as PySCF basis dicts, read from CP2K at run time.

The sets are the all-electron Gaussian bases for TDDFT, GW and BSE of Pasquier,
Graml and Wilhelm, J. Chem. Theory Comput. 22, 540 (2026), which CP2K ships as
`data/BASIS_AUG_MOLOPT` (orbital sets) and `data/BASIS_RI_AUG_MOLOPT` (auxiliary
RI sets, in tiers). CP2K distributes them under GPL-2.0-or-later, so this module
copies nothing: it downloads the two files from the CP2K repository at a pinned
commit into a per-user cache (or reads a local CP2K checkout), verifies them, and
prints a notice naming the source, its license and the paper to cite on first use.

    basis, aux = register('aug-SZV-MOLOPT-ae', max_error=1e-4)
    mol = gto.M(atom=..., basis=basis)
    mf = dft.RKS(mol, xc='PBE').density_fit(auxbasis=aux)

`register` makes the sets PySCF basis names: `<name>`, and `<name>-ri` with the
tightest RI tier per element, which the defaults of this tree that form
`str(mol.basis) + '-ri'` resolve, or for a threshold `<name>-ri-<Delta-I>`, one
name per set of tiers, so the name-keyed ISDF radii cache works as for any named
basis. A threshold call registers only its own RI name, so a route not handed
`auxbasis` raises rather than switching sets. `load_basis` and `load_ri_basis`
return the same sets as dicts; a dict has no name, so pass its auxiliary basis
explicitly.

`pyscf.gto.basis.parse_cp2k.parse` reads the block format; its `load` and
`search_seg` split a file on `# BASIS SET` delimiters, which these files do not
carry, so they raise, and where a file has them they take the FIRST block for an
element, never a named one. A block is therefore sliced here by exact header name.
A header is `<El> <name> [<alias>]` and the wanted name must be one of its
tokens: H's `aug-SZV-MOLOPT-ae` header also carries `aug-SZV-MOLOPT-ae-mini`. An
RI header is `RI_<basis>_N_RI_<nfunc>_s_p_d_f_g_h_i_<n_s>_.._<n_i>_error_<err>`,
where `err` is the tier's relative Delta-I of the paper's eq 29, an atomic RI-MP2
amplitude error. It predicts the (ia|jb) class of integrals; the (ij|ab) class
that the BSE direct term and the GW self-energy contract is governed by the tier's
angular momenta instead: a product of two orbital functions needs auxiliary
functions up to twice the orbital l_max. Measured on formaldehyde against the
exact four-center tensor, the Delta-I 1e-4 tiers (no d on H, no g on O) leave 27 meV
on the lowest BSE singlets and 6 meV on HOMO and LUMO; the tightest tiers, with g on
C and O, leave 1.6 meV and under 1 meV. `min_lmax` in `pick_ri_tier` asks for
that completeness explicitly.

Command line: `python -m src.Base.basis.cp2k_basis scout [ELEMENT ...]` lists,
per element, the orbital sets with their function counts and the RI tiers with
size and Delta-I; `fetch` only fills the cache, for a node without network later.
Usage, sources and the choice of RI tier: README.md beside this file.
Environment: MBPT_CP2K_DATA, a CP2K `data/` directory to read instead of
downloading; MBPT_CP2K_CACHE, the cache root (default ~/.cache/mbptcode/cp2k).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import warnings

from pyscf.gto import basis as pyscf_basis
from pyscf.gto.basis.parse_cp2k import parse

CP2K_REPO = 'https://github.com/cp2k/cp2k'
FILES = {'orbital': 'BASIS_AUG_MOLOPT', 'ri': 'BASIS_RI_AUG_MOLOPT'}
# cp2k_pin.json: the commit that last touched either file (2025-08-14), the two
# files' sha256, and per element `content_digests` of its blocks, so `register`
# accepts a copy that differs in comments and names the elements whose data differ.
with open(os.path.join(os.path.dirname(__file__), 'cp2k_pin.json')) as _fh:
    _PIN = json.load(_fh)
CP2K_COMMIT, SHA256, BLOCK_SHA256 = _PIN['commit'], _PIN['sha256'], _PIN['blocks']
BASIS_NAMES = ('aug-SZV-MOLOPT-ae', 'aug-SZV-MOLOPT-ae-mini', 'aug-SZV-MOLOPT-ae-SR',
               'aug-DZVP-MOLOPT-ae', 'aug-TZVP-MOLOPT-ae')
CITATION = ('R. Pasquier, M. Graml, J. Wilhelm, J. Chem. Theory Comput. 22, 540 '
            '(2026), doi:10.1021/acs.jctc.5c01386')
SPDF = 'spdfghi'

# Data lines start with a digit; only a header starts with an element symbol.
_HEADER = re.compile(r'^(?P<el>[A-Z][a-z]?)\s+(?P<names>\S.*)$')
_RI_NAME = re.compile(r'^RI_(?P<basis>.+)_N_RI_(?P<n>\d+)_s_p_d_f_g_h_i(?:_\d+){7}'
                      r'_error_(?P<err>[0-9.]+e[+-]?\d+)$')
_noticed = set()


def _sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _replace_with(path, write, mode='w'):
    """Write through `write(fh)` into a fresh file beside `path`, then move it there.

    No reader sees a partial file, and two processes writing at once, on one node
    or on two nodes of a shared filesystem, never share a temporary name.
    """
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                               prefix=os.path.basename(path) + '.')
    try:
        with os.fdopen(fd, mode) as fh:
            write(fh)
    except BaseException:
        os.unlink(tmp)
        raise
    os.replace(tmp, path)


def _notice(kind, path, commit):
    """Print, once per process and file, where the data came from and its license."""
    if path in _noticed:
        return
    _noticed.add(path)
    source = f'commit {commit[:10]}' if commit else 'a local checkout'
    print(f'NOTE: basis set data read from CP2K: data/{FILES[kind]} at {CP2K_REPO}, '
          f'{source}, distributed by CP2K under GPL-2.0-or-later; MBPTcode does not '
          f'redistribute this file (local copy: {path}). If you use these basis sets, '
          f'we would kindly ask you to cite {CITATION}.')


def _cache_dir(commit):
    root = os.environ.get('MBPT_CP2K_CACHE', os.path.join(os.path.expanduser('~'),
                                                          '.cache', 'mbptcode', 'cp2k'))
    return os.path.join(root, commit)


def data_file(kind, commit=CP2K_COMMIT, download=True):
    """Path of the orbital or RI file: MBPT_CP2K_DATA, the cache, or a download.

    Parameters
    ----------
    kind : {'orbital', 'ri'}
    commit : str
        CP2K commit to fetch from. The pinned default is verified by sha256; any
        other commit is read unverified and its digest is left to `provenance`.
    download : bool
        False raises FileNotFoundError instead of fetching, for offline use.
    """
    local = os.environ.get('MBPT_CP2K_DATA')
    if local:
        path = os.path.join(local, FILES[kind])
        if not os.path.exists(path):
            raise FileNotFoundError(f'MBPT_CP2K_DATA={local} has no {FILES[kind]}')
        _notice(kind, path, None)
        return path
    path = os.path.join(_cache_dir(commit), FILES[kind])
    if not os.path.exists(path):
        url = f'https://raw.githubusercontent.com/cp2k/cp2k/{commit}/data/{FILES[kind]}'
        if not download:
            raise FileNotFoundError(f'{path} is not cached and download=False')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                _replace_with(path, lambda out: out.write(resp.read()), 'wb')
        except (urllib.error.URLError, OSError) as exc:
            raise OSError(f'could not download {url} ({exc}); point MBPT_CP2K_DATA '
                          'at a CP2K data directory, or run "python -m '
                          'src.Base.basis.cp2k_basis fetch" where the network is '
                          'reachable') from exc
    if commit == CP2K_COMMIT and _sha256(path) != SHA256[kind]:
        raise ValueError(f'{path} does not match the pinned CP2K commit {commit[:10]}; '
                         'delete it and fetch again')
    _notice(kind, path, commit)
    return path


def provenance(kind, path=None):
    """{'path', 'sha256', 'commit', 'blocks'} of the file a run read, for its record.

    `commit` is the pinned commit when the file is its file, `blocks` when the
    basis data are, comments and spacing aside; None otherwise.
    """
    path = path or data_file(kind)
    digest = _sha256(path)
    commit = CP2K_COMMIT if digest == SHA256[kind] else None
    blocks = CP2K_COMMIT if content_digests(path) == BLOCK_SHA256[kind] else None
    return {'path': path, 'sha256': digest, 'commit': commit, 'blocks': blocks}


def content_digests(path):
    """{element: 16 hex digits of the sha256 over its blocks}, comments and spacing
    dropped, inside a line as well."""
    lines = {}
    for el, _, block in _blocks(path):
        lines.setdefault(el, []).extend(' '.join(line.split()) for line in block)
    return {el: hashlib.sha256('\n'.join(v).encode()).hexdigest()[:16]
            for el, v in lines.items()}


def _blocks(path):
    """(element, header names, stripped lines with the header first), per block."""
    block = None
    with open(path) as fh:
        for raw in fh:
            line = raw.split('#')[0].strip()
            if not line:
                continue
            m = _HEADER.match(line)
            if m:
                if block:
                    yield block
                block = (m['el'], m['names'].split(), [line])
            elif block:
                block[2].append(line)
    if block:
        yield block


def basis_block(name, element, path):
    """Lines of `element`'s block whose header names `name` exactly, header first."""
    hits = [lines for el, names, lines in _blocks(path)
            if el == element and name in names]
    if len(hits) != 1:
        raise ValueError(f'{len(hits)} blocks named {name} for {element} in {path}')
    # Every data token must be a number: pyscf's parser falls back to eval() on
    # anything else, which a file from an unpinned commit must never reach.
    for line in hits[0][1:]:
        try:
            [float(x) for x in line.split()]
        except ValueError:
            raise ValueError(f'non-numeric data line in block {name} for {element}: '
                             f'{line!r}') from None
    return hits[0]


def _sets(lines):
    """(lmin, lmax, contractions per l) of every set in a block, header skipped."""
    rows = iter(lines[1:])
    out = []
    for _ in range(int(next(rows))):
        comp = [int(x) for x in next(rows).split()]
        lmin, lmax, nexp, ncont = comp[1], comp[2], comp[3], comp[4:]
        if len(ncont) != lmax - lmin + 1:
            raise ValueError(f'set line {comp}: {len(ncont)} contraction counts '
                             f'for l = {lmin}..{lmax}')
        out.append((lmin, lmax, ncont))
        for _ in range(nexp):
            next(rows)
    return out


def nao_from_block(lines):
    """Function count the block implies: per set and l, contractions times 2l + 1."""
    return sum(n * (2 * l + 1) for lmin, lmax, ncont in _sets(lines)
               for l, n in zip(range(lmin, lmax + 1), ncont))


def pattern(lines):
    """Contractions per l as a string, '3s2p1d' for C aug-SZV-MOLOPT-ae."""
    per_l = {}
    for lmin, lmax, ncont in _sets(lines):
        for l, n in zip(range(lmin, lmax + 1), ncont):
            per_l[l] = per_l.get(l, 0) + n
    return ''.join(f'{per_l[l]}{SPDF[l]}' for l in sorted(per_l))


def load_basis(name, elements, path=None):
    """{element: PySCF internal basis} for one orbital set name.

    Parameters
    ----------
    name : str
        One of `BASIS_NAMES`.
    elements : iterable of str
    path : str, optional
        The orbital file; default `data_file('orbital')`.
    """
    path = path or data_file('orbital')
    return {el: parse('\n'.join(basis_block(name, el, path))) for el in elements}


def ri_tiers(basis_name, element, path=None):
    """[(ri_name, nfunc, delta_i, pattern)] of a basis' RI tiers, loosest first."""
    path = path or data_file('ri')
    tiers = []
    for el, names, lines in _blocks(path):
        if el != element:
            continue
        for n in names:
            m = _RI_NAME.match(n)
            if m and m['basis'] == basis_name:
                tiers.append((n, int(m['n']), float(m['err']), pattern(lines)))
    return sorted(tiers, key=lambda t: -t[2])


def _lmax(pat):
    return max(SPDF.index(c) for c in pat if c in SPDF)


def _threshold(max_error):
    """None for the tightest tiers, which 0 also means; negative or nan raise."""
    if max_error is not None and not max_error >= 0:
        raise ValueError(f'max_error={max_error} is not a Delta-I threshold')
    return max_error or None


def pick_ri_tier(basis_name, element, max_error=None, min_lmax=None, path=None):
    """The smallest tier within `max_error` and with l_max >= `min_lmax`.

    The default, `max_error=None` or 0, is the tightest tier: converged, and the
    most expensive. 1e-4 is the paper's recommendation (Section 3.2) and an MP2
    criterion; see the module docstring for what it leaves on BSE energies.
    `min_lmax` adds the angular condition, twice the orbital l_max for a complete
    product space. When no tier satisfies both conditions the tightest one is
    returned with a warning.

    Returns
    -------
    (ri_name, nfunc, delta_i, pattern)
    """
    max_error = _threshold(max_error)
    tiers = ri_tiers(basis_name, element, path)
    if not tiers:
        raise ValueError(f'no RI tier for {basis_name}, {element} in {path}')
    within = [t for t in tiers
              if (max_error is None or t[2] <= max_error)
              and (min_lmax is None or _lmax(t[3]) >= min_lmax)]
    if max_error is None and within:
        return min(within, key=lambda t: t[2])
    if within:
        return min(within, key=lambda t: t[1])
    tightest = min(tiers, key=lambda t: t[2])
    wanted = [f'Delta-I <= {max_error}'] if max_error is not None else []
    wanted += [f'l_max >= {min_lmax}'] if min_lmax is not None else []
    warnings.warn(f"no RI tier for {basis_name} {element} has {' and '.join(wanted)}; "
                  f'using the tightest, {tightest[0]}', stacklevel=2)
    return tightest


def load_ri_basis(basis_name, elements, max_error=None, min_lmax=None, path=None):
    """{element: PySCF internal basis}, one RI tier per element; see `pick_ri_tier`."""
    path = path or data_file('ri')
    return {el: parse('\n'.join(basis_block(
                pick_ri_tier(basis_name, el, max_error, min_lmax, path)[0], el, path)))
            for el in elements}


def _write_nwchem(fh, table):
    """{element: PySCF internal basis} as NWChem text PySCF reads back bit for bit."""
    for el, shells in table.items():
        fh.write(f'#BASIS SET: {el}\n')
        for shell in shells:
            fh.write(f'{el}    {SPDF[shell[0]].upper()}\n')
            for row in shell[1:]:
                fh.write('  '.join(repr(float(x)) for x in row) + '\n')


def register(name, max_error=None):
    """Make an orbital set and its RI tiers PySCF basis names, for this process.

    The sets are written, for every element CP2K has them for, into the cache
    beside the CP2K files and added to `pyscf.gto.basis.USER_BASIS_ALIAS`, which
    lives in the running process only: call this once per script, before the
    Mole. The RI name is `<name>-ri` for the tightest tiers, or for a threshold
    `<name>-ri-<Delta-I>` with the smallest threshold that picks the same tiers,
    so one name means one set in every process (the ISDF radii cache is keyed on
    it) and the name's Delta-I reproduces the set. A threshold call registers only
    that name; a route not handed it raises rather than switching sets.
    `min_lmax` has no name; use the dict of `load_ri_basis` for it.

    Parameters
    ----------
    name : str
        One of `BASIS_NAMES`.
    max_error : float, optional
        Delta-I threshold as in `pick_ri_tier`; None or 0, the tightest tiers.

    Returns
    -------
    (basis_name, ri_name) : the names to pass as `basis=` and `auxbasis=`. The RI
        name covers the elements that have RI tiers for `name`.
    """
    if name not in BASIS_NAMES:
        raise ValueError(f'{name} is not one of {BASIS_NAMES}')
    max_error = _threshold(max_error)
    orbital, ri = data_file('orbital'), data_file('ri')
    # Other data would put a second set under the same name, and a cached ISDF
    # grid keyed on that name would silently serve the wrong one.
    for kind, path in (('orbital', orbital), ('ri', ri)):
        got, pinned = content_digests(path), BLOCK_SHA256[kind]
        changed = sorted(el for el in set(got) | set(pinned)
                         if got.get(el) != pinned.get(el))
        if changed:
            raise ValueError(f'{path}: the blocks of {", ".join(changed)} differ from '
                             f'the pinned CP2K commit {CP2K_COMMIT[:10]}; names are '
                             'reserved for that data, use load_basis and '
                             'load_ri_basis instead')
    elements = sorted({el for el, names, _ in _blocks(orbital) if name in names})
    ri_elements = [el for el in elements if ri_tiers(name, el, ri)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # one warning below, not one per element
        picked = {el: pick_ri_tier(name, el, max_error, path=ri)
                  for el in ri_elements if max_error is not None}
        tightest = {el: pick_ri_tier(name, el, path=ri) for el in picked}
    # The smallest threshold that picks these tiers; a tightest tier is picked at
    # every lower threshold as well, so it does not count.
    error = max((t[2] for el, t in picked.items() if t != tightest[el]), default=None)
    # The file writes every Delta-I with two significant digits, so .1e is exact.
    ri_name = f'{name}-ri' if error is None else f'{name}-ri-{error:.1e}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sets = {name: load_basis(name, elements, orbital),
                ri_name: load_ri_basis(name, ri_elements, max_error, path=ri)}
    loose = [f'{el} (tightest tier, Delta-I {t[2]:.1e})' for el, t in picked.items()
             if t[2] > max_error]
    if picked and error is None:
        warnings.warn(f'{name}: max_error={max_error} picks the tightest tier of every '
                      f'element, which is {ri_name}, the set of max_error=None',
                      stacklevel=2)
    elif loose:
        warnings.warn(f'{ri_name}: no tier within {max_error} for {", ".join(loose)}',
                      stacklevel=2)
    out = os.path.join(_cache_dir(CP2K_COMMIT), 'pyscf')
    os.makedirs(out, exist_ok=True)
    for alias, table in sets.items():
        key = pyscf_basis._format_basis_name(alias)
        if key in pyscf_basis.ALIAS or key in pyscf_basis.GTH_ALIAS:
            raise ValueError(f'{alias} collides with a basis name PySCF ships')
        path = os.path.join(out, f'{alias}.dat')
        _replace_with(path, lambda fh: _write_nwchem(fh, table))
        pyscf_basis.USER_BASIS_ALIAS[key] = path
    return name, ri_name


def available(element, orbital_path=None, ri_path=None):
    """What the files offer for one element.

    Returns
    -------
    dict
        'orbital': [(name, nao, pattern)], 'ri': {basis_name: ri_tiers(...)}.
    """
    orbital_path = orbital_path or data_file('orbital')
    ri_path = ri_path or data_file('ri')
    orbital = [(n, nao_from_block(lines), pattern(lines))
               for el, names, lines in _blocks(orbital_path) if el == element
               for n in names]
    ri = {}
    for el, names, _ in _blocks(ri_path):
        if el != element:
            continue
        for n in names:
            m = _RI_NAME.match(n)
            if m:
                ri.setdefault(m['basis'], None)
    return {'orbital': orbital,
            'ri': {b: ri_tiers(b, element, ri_path) for b in sorted(ri)}}


def main(argv=None):
    """The `fetch` and `scout` subcommands of the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch', help='fill the cache with both files at the pinned commit')
    p = sub.add_parser('scout', help='list the orbital sets and RI tiers per element')
    p.add_argument('elements', nargs='*', default=['H', 'C', 'N', 'O'])
    args = ap.parse_args(argv)
    for kind in FILES:
        print(f'{kind}: {data_file(kind)}')
    if args.cmd == 'scout':
        for el in args.elements:
            info = available(el)
            print(f'\n{el}: orbital sets')
            for name, nao, pat in info['orbital']:
                print(f'  {name:28s} {nao:4d} functions  {pat}')
            for basis, tiers in info['ri'].items():
                print(f'{el}: RI tiers for {basis}')
                for name, n, err, pat in tiers:
                    print(f'  {n:4d} functions  Delta-I {err:.1e}  {pat:16s} {name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
