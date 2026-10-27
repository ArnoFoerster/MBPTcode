"""Explicit auxiliary-basis fallback for elements an augmented RI family lacks (Mg at aug-cc-pV{D,T}Z).

PySCF has no `aug-cc-pvdz-ri` or `aug-cc-pvtz-ri` for Mg, so a default
`str(mol.basis) + '-ri'` raises there and nothing in this tree substitutes by
itself. A caller opts in by name:

    aux = fallback_auxbasis('aug-cc-pvdz')   # 'aug-cc-pvdz-ri+mg-cc-pvqz-ri'
    mf = scf.RHF(mol).density_fit(auxbasis=aux)

The name is a PySCF basis registered for this process: the genuine
`aug-cc-pvdz-ri` of every element that has one and `cc-pvqz-ri` on Mg. No genuine
name is redefined. The ISDF radii table is keyed on the auxiliary set an atomic
grid was fitted in, so `element_auxbasis` maps the composite name to the genuine
key on H, O, ... and to `cc-pvqz-ri` on Mg; the Mg row key shows the substitution.
The table of substitutes, `ISDF_AUX_FALLBACK`, carries the measurement behind it.
"""
import os
import tempfile

from pyscf.gto import basis as pyscf_basis

from src.Base.basis.cp2k_basis import write_nwchem
from src.Base.numerics.constants import ISDF_AUX_FALLBACK, ISDF_MAX_AUX_L

_SYMBOLS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
            'Co Ni Cu Zn Ga Ge As Se Br Kr').split()
_registered = {}


def element_basis_name(basis, element):
    """The name of the basis one element carries, lower-cased, for a table key.

    A basis is a name for every element or a dict of per-element names (pyscf
    accepts both; a lithium-containing molecule in an augmented set needs the
    dict, since the auxiliary sets that cover Li differ from the rest). A dict
    gives this element's entry, or its 'default'. Lower-cased because the
    table's keys are and pyscf's names are case-insensitive: 'cc-pVDZ' and
    'cc-pvdz' are one basis and must be one row. An explicit basis definition
    (shells, not a name) has no row to look up and is refused by name.
    """
    name = basis.get(element, basis.get('default')) if isinstance(basis, dict) \
        else basis
    if not isinstance(name, str):
        raise TypeError(f'the basis for {element} is not a named set '
                        f'({type(name).__name__}); the radii table is keyed on '
                        f'basis names, so pass a name or explicit radii.')
    return name.strip().lower()


def paired_auxbasis(basis):
    """The auxiliary basis a route fits `basis` in where none is named:
    `<basis>-ri`. THE pairing rule, spelled once; an element whose shipped
    set lacks the partner takes its substitute (`element_auxbasis`). A
    per-element basis pairs element by element."""
    if isinstance(basis, dict):
        return {el: paired_auxbasis(element_basis_name(basis, el))
                for el in basis}
    return f'{basis}-ri'


def fallback_elements(basis):
    """Elements with a substitute auxiliary set at `basis`."""
    return [el for (el, b) in ISDF_AUX_FALLBACK if b == str(basis).lower()]


def fallback_name(basis):
    """PySCF name of the composite set, e.g. `aug-cc-pvdz-ri+mg-cc-pvqz-ri`."""
    basis = str(basis).lower()
    subs = [f'{el.lower()}-{ISDF_AUX_FALLBACK[(el, basis)]}'
            for el in fallback_elements(basis)]
    if not subs:
        raise ValueError(f'no auxiliary fallback for basis {basis!r}; defined for '
                         f'{sorted({b for _, b in ISDF_AUX_FALLBACK})}')
    return f'{basis}-ri+' + '+'.join(subs)


def element_auxbasis(element, basis, auxbasis):
    """Auxiliary set `element`'s atomic grid is keyed on; the fallback name resolves per element."""
    basis = str(basis).lower()
    if auxbasis is None or not fallback_elements(basis) \
            or str(auxbasis) != fallback_name(basis):
        return auxbasis
    return ISDF_AUX_FALLBACK.get((element, basis), paired_auxbasis(basis))


def fallback_auxbasis(basis):
    """Register the composite auxiliary set of `basis` in this process and return its name."""
    basis = str(basis).lower()
    name = fallback_name(basis)
    if name in _registered:
        return name
    key = pyscf_basis._format_basis_name(name)
    if key in pyscf_basis.ALIAS or key in pyscf_basis.GTH_ALIAS:
        raise ValueError(f'{name} collides with a basis name PySCF ships')
    check_fallback(basis)
    table = {}
    for el in _SYMBOLS:
        sub = ISDF_AUX_FALLBACK.get((el, basis))
        try:
            table[el] = pyscf_basis.load(sub or paired_auxbasis(basis), el)
        except pyscf_basis.BasisNotFoundError:
            if sub:
                raise
    out = os.environ.get('MBPT_RI_FALLBACK_CACHE', os.path.join(
        os.path.expanduser('~'), '.cache', 'mbptcode', 'ri_fallback'))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'{name}.dat')
    fd, tmp = tempfile.mkstemp(dir=out, prefix=name + '.')
    try:
        with os.fdopen(fd, 'w') as fh:
            write_nwchem(fh, table)
    except BaseException:
        os.unlink(tmp)
        raise
    os.replace(tmp, path)
    pyscf_basis.USER_BASIS_ALIAS[key] = path
    _registered[name] = path
    return name


def check_fallback(basis):
    """Refuse a substitute above the highest l the ISDF interpolation grid represents."""
    for el in fallback_elements(basis):
        sub = ISDF_AUX_FALLBACK[(el, str(basis).lower())]
        lmax = max(sh[0] for sh in pyscf_basis.load(sub, el))
        if lmax > ISDF_MAX_AUX_L:
            raise ValueError(f'{sub} on {el} reaches l={lmax} > {ISDF_MAX_AUX_L}')


def auxbasis_for(basis, elements, fallback=False):
    """`<basis>-ri` if it covers every element; else the named fallback if `fallback`; else raise."""
    basis = str(basis).lower()
    plain = paired_auxbasis(basis)
    missing = []
    for el in sorted(set(elements)):
        try:
            pyscf_basis.load(plain, el)
        except pyscf_basis.BasisNotFoundError:
            missing.append(el)
    if not missing:
        return plain
    if any((el, basis) not in ISDF_AUX_FALLBACK for el in missing):
        raise ValueError(f'{plain} has no set for {", ".join(missing)} and no '
                         f'fallback is defined for all of them')
    if not fallback:
        raise ValueError(f'{plain} has no set for {", ".join(missing)}. Pass '
                         f'fallback=True, or auxbasis=fallback_auxbasis({basis!r}), '
                         f'to use {fallback_name(basis)}.')
    return fallback_auxbasis(basis)


def missing_row_hint(element, basis, auxbasis, counts):
    """What a missing radii row of a fallback element needs, else ''."""
    basis = str(basis).lower()
    if (element, basis) not in ISDF_AUX_FALLBACK:
        return ''
    return (f' {element} at {basis} uses the explicit auxiliary fallback '
            f'{ISDF_AUX_FALLBACK[(element, basis)]}; its row is keyed '
            f'{element}|{basis}|{ISDF_AUX_FALLBACK[(element, basis)]}|<counts> '
            f'and has to be optimized in that auxiliary set.')
