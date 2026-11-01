"""One orbital basis for a molecule: a name, a name per element, or a name per
atom.

A run asks for its basis as one name and, optionally, other names for some
elements or some atoms: aug-cc-pVDZ on the heavy atoms and cc-pVDZ on
hydrogen, or aug-cc-pVDZ on an emitter (its atoms 1-41) and cc-pVDZ on the
host molecules around it:

    atom, basis = molecule_basis(atom, 'cc-pvdz', {}, {range(0, 41): 'aug-cc-pvdz'})
    mol = gto.M(atom=atom, basis=basis)

The result is the canonical form every stamp and record carries, decided by
the basis each atom ends up with and by nothing else:

    one set on every atom           the plain name, the atoms as given
    one set per element             {element: name} naming every element of
                                    the molecule, the atoms as given
    otherwise                       each element's atoms grouped by their set;
                                    the group of the element's first atom
                                    keeps the element's symbol, every other
                                    group is labelled 'C@1', 'C@2', ... in the
                                    order its first atom comes, and the dict
                                    names every symbol and label

Names are lower-cased. pyscf reads a labelled atom's basis under its label;
the auxiliary sets and the interpolation grid are looked up under the same
key (`geometry.grid_key`), so a per-atom basis pairs its -ri and -jkfit sets
and takes its grid rows atom by atom.
"""
import re

from src.Base.basis.ri_fallback import element_basis_name, pure_element

#: The selector of an atom-index item of --basis-for.
ATOMS = 'atoms'


def parse_basis_for(items):
    """({element: name}, {atom indices (0-based, a tuple): name}) of
    --basis-for items: 'El=NAME' for an element, 'atoms=I-J,K:NAME' for
    atoms counted from 1 as in an xyz file. An element named twice, an atom
    named twice, a malformed item or an unknown symbol is refused."""
    from pyscf.data.elements import ELEMENTS
    per_element, per_atom, seen = {}, {}, {}
    for item in items or ():
        key, sep, rest = str(item).partition('=')
        key, rest = key.strip(), rest.strip()
        if key.lower() == ATOMS:
            spec, colon, name = rest.rpartition(':')
            name = name.strip()
            if not colon or not name or not spec.strip():
                raise ValueError(f'{item!r}: an atom range is atoms=I-J[,K...]:NAME, '
                                 f'e.g. atoms=1-41:aug-cc-pvdz')
            atoms = _atom_indices(spec, item)
            twice = sorted(i + 1 for i in atoms if i in seen)
            if twice:
                raise ValueError(f'atom(s) {twice} are given a basis twice')
            seen.update(dict.fromkeys(atoms))
            per_atom[tuple(atoms)] = name
            continue
        if not sep or not key or not rest:
            raise ValueError(f'{item!r}: a per-element basis is El=NAME, '
                             f'e.g. H=cc-pvdz')
        el = key[0].upper() + key[1:].lower()
        if el not in ELEMENTS[1:]:
            raise ValueError(f'{item!r}: {el!r} is not an element symbol '
                             f'(atoms are named by atoms=I-J:NAME)')
        if el in per_element:
            raise ValueError(f'{el} is given a basis twice '
                             f'({per_element[el]!r} and {rest!r})')
        per_element[el] = rest
    return per_element, per_atom


def _atom_indices(spec, item):
    """0-based indices of '1-41,45' (counted from 1), in order."""
    out = []
    for part in spec.split(','):
        m = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', part)
        if m is None:
            raise ValueError(f'{item!r}: {part!r} is not an atom number or range')
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if lo < 1 or hi < lo:
            raise ValueError(f'{item!r}: {part!r} is not a range of atoms counted from 1')
        out.extend(range(lo - 1, hi))
    return out


def atom_items(atom):
    """The items of a pyscf atom string, one per atom, as written."""
    return [s.strip() for s in re.split(r'[;\n]', atom) if s.strip()]


def atom_elements(atom):
    """The sorted element symbols of a pyscf atom string or list (ghosts
    and labels reduced to their element)."""
    from pyscf.gto.mole import _std_symbol_without_ghost, format_atom
    return sorted({_std_symbol_without_ghost(a[0]) for a in format_atom(atom)})


def molecule_basis(atom, name, per_element=None, per_atom=None):
    """(atom, basis): the canonical form (module docstring) of `name` on
    every atom of the pyscf atom string `atom`, `per_element` {element: name}
    on an element's atoms and `per_atom` {indices: name} on those atoms over
    both. An element or an atom the molecule does not have is refused, since
    it would make two stamps for one basis or hide a typo; so is an atom
    already labelled where a per-atom basis needs labels of its own."""
    items = atom_items(atom)
    symbols = [item.split()[0] for item in items]
    elements = [pure_element(s) for s in symbols]
    per_element = dict(per_element or {})
    absent = sorted(set(per_element) - set(elements))
    if absent:
        raise ValueError(f'a basis is given for {", ".join(absent)}, which the '
                         f'molecule does not have (it has '
                         f'{", ".join(sorted(set(elements)))})')
    names = [element_basis_name(per_element.get(el, name), el) for el in elements]
    for atoms, atom_name in (per_atom or {}).items():
        outside = [i + 1 for i in atoms if i >= len(items)]
        if outside:
            raise ValueError(f'atom(s) {outside} are given a basis, and the '
                             f'molecule has {len(items)} atoms')
        for i in atoms:
            names[i] = element_basis_name(atom_name, elements[i])
    if len(set(names)) == 1:
        return atom, names[0]
    by_element = {}
    for el, n in zip(elements, names):
        by_element.setdefault(el, []).append(n)
    if all(len(set(v)) == 1 for v in by_element.values()):
        return atom, {el: v[0] for el, v in sorted(by_element.items())}
    labelled = [s for s, el in zip(symbols, elements) if s != el]
    if labelled:
        raise ValueError(f'a per-atom basis labels the atoms itself, and these '
                         f'are labelled already: {sorted(set(labelled))}')
    basis, groups, out = {}, {}, []
    for item, el, n in zip(items, elements, names):
        kinds = groups.setdefault(el, {})
        if n not in kinds:
            kinds[n] = el if not kinds else f'{el}@{len(kinds)}'
            basis[kinds[n]] = n
        out.append(' '.join([kinds[n]] + item.split()[1:]))
    return '; '.join(out), dict(sorted(basis.items()))


def basis_keys(atom):
    """The keys a canonical basis of `atom` names: each atom's symbol or
    label, sorted, each once."""
    return sorted({item.split()[0] for item in atom_items(atom)})


def atom_basis_names(atom, basis):
    """The set each atom of `atom` carries, in atom order: what a record
    keys a per-atom basis on where it keeps the geometry apart from its
    labels."""
    return [element_basis_name(basis, item.split()[0]) for item in atom_items(atom)]


def basis_label(basis):
    """A one-line name of a basis for a message or a table: the name, or
    'key=name,...' sorted by key."""
    if isinstance(basis, dict):
        return ','.join(f'{el}={basis[el]}' for el in sorted(basis))
    return str(basis)
