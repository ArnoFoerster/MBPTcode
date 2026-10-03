"""A protein (or any molecular) environment from a charged structure file.

Turns a PQR file -- a structure with one partial charge and one radius per
atom, as pdb2pqr or AmberTools' ambpdb write it from a force field -- into
what a chain needs: the QM molecule, its fragment partition, and the
classical environment around it (`PointCharges` for the permanent charges,
optionally `PolarizableSites` for the induced dipoles, combined by
`CompositeEnvironment` so neither channel leaks into the other).

THE QM REGION is a list of residues, each one fragment of the partition (a
pigment, a carotenoid, ...). Atoms of a selected residue can be excluded --
a chlorophyll's phytyl tail, typically -- and then belong to the environment.

COVALENT CUTS. A bond from a QM atom to an environment atom is cut and capped
with a hydrogen LINK ATOM on the bond, at the standard X-H distance of the QM
atom's element. The environment atom on the other side (the boundary atom)
would put a bare charge next to the link atom; its charge is set to zero and
redistributed in equal parts onto the environment atoms bonded to it, which
conserves the total charge (the redistributed-charge idea of Lin and Truhlar,
J. Phys. Chem. A 109, 3991 (2005), in its simplest form). Bonds are found from
covalent radii; bonds to the metals in `COORDINATION_ELEMENTS` are coordination,
not covalent, and are never cut: a chlorophyll's magnesium keeps its axial
ligand in the environment without a link atom.

WHAT IS KEPT. Every environment residue with an atom within `cutoff` Angstrom
of a QM atom, WHOLE, so that a neutral residue enters neutral; nothing else.
Polarizable sites sit on the kept atoms, with isotropic polarizabilities from a
per-element table the caller supplies, and sites of one residue do not
polarize each other (`exclude_within='residue'`, the exclusion convention of
polarizable embedding with atom-centred sites; bonded atoms are otherwise
inside the polarization catastrophe of any realistic pair) -- there is no default table, because
the numbers belong to the force field the charges came from, and a site
closer to a QM atom than MIN_SITE_TO_QM_DISTANCE is dropped (and reported),
since the folding of the screened interaction onto the QM region assumes the
QM density does not reach the sites (`polarizable_sites`).

`report` records what was done: counts, the environment's total charge, the
QM charge read off the file, link atoms, dropped sites, and the closest
charge to any QM atom, which is where an over-polarized boundary would show.
"""
from dataclasses import dataclass, field

import numpy as np
from pyscf import gto

from src.Base.composite_environment import CompositeEnvironment
from src.Base.constants import BOHR_TO_ANGSTROM, MIN_SITE_TO_QM_DISTANCE
from src.Base.environment import PointCharges
from src.Base.polarizable_sites import PolarizableSites

#: Covalent radii in Angstrom (Cordero et al., Dalton Trans. 2832 (2008)),
#: for bond detection only.
COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05,
                  'P': 1.07, 'F': 0.57, 'Cl': 1.02, 'Mg': 1.41, 'Zn': 1.22,
                  'Fe': 1.32, 'Mn': 1.39, 'Ca': 1.76, 'Na': 1.66, 'K': 2.03}
#: Bonded when closer than this factor times the sum of covalent radii.
BOND_TOLERANCE = 1.2
#: Bonds to these elements are coordination, never cut, never capped.
COORDINATION_ELEMENTS = ('Mg', 'Zn', 'Fe', 'Mn', 'Ca', 'Na', 'K')
#: Link-atom X-H distances in Angstrom by the element of the QM atom capped.
LINK_BOND_LENGTH = {'C': 1.09, 'N': 1.01, 'O': 0.96, 'S': 1.34}
#: Atom names that are two-letter elements when they stand alone.
_TWO_LETTER = {'MG': 'Mg', 'ZN': 'Zn', 'FE': 'Fe', 'MN': 'Mn', 'CL': 'Cl',
               'NA': 'Na'}


@dataclass
class PQRAtom:
    name: str
    resname: str
    chain: str
    resseq: int
    xyz: np.ndarray
    charge: float
    radius: float
    element: str

    @property
    def residue(self):
        return (self.chain, self.resseq, self.resname)


def _element(name, resname):
    key = name.strip().upper()
    if key in _TWO_LETTER and resname.strip().upper() != 'HOH':
        return _TWO_LETTER[key]
    letters = ''.join(c for c in key if c.isalpha())
    if not letters:
        raise ValueError(f'cannot read an element from atom name {name!r}')
    return letters[0]


def read_pqr(path):
    """[PQRAtom] from a PQR file (ATOM / HETATM records, whitespace-separated:
    serial, name, residue name, optional chain, residue number, x, y, z,
    charge, radius)."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            f = line.split()
            if len(f) == 11:
                _, _, name, resname, chain, resseq, x, y, z, q, r = f
            elif len(f) == 10:
                _, _, name, resname, resseq, x, y, z, q, r = f
                chain = ''
            else:
                raise ValueError(f'unreadable PQR record: {line.rstrip()}')
            atoms.append(PQRAtom(name, resname, chain, int(resseq),
                                 np.array([float(x), float(y), float(z)]),
                                 float(q), float(r), _element(name, resname)))
    if not atoms:
        raise ValueError(f'{path}: no ATOM/HETATM records')
    return atoms


def _bonded(a, b):
    if a.element in COORDINATION_ELEMENTS or b.element in COORDINATION_ELEMENTS:
        return False
    ra = COVALENT_RADII.get(a.element)
    rb = COVALENT_RADII.get(b.element)
    if ra is None or rb is None:
        raise ValueError(f'no covalent radius for {a.element} or {b.element}')
    return np.linalg.norm(a.xyz - b.xyz) < BOND_TOLERANCE * (ra + rb)


@dataclass
class ProteinEmbedding:
    """The QM molecule, its fragments, and the classical environment."""
    mol: object
    fragments: list
    environment: object
    charges: object
    sites: object
    report: dict = field(default_factory=dict)


def embed(atoms, qm_residues, basis, cutoff=12.0, exclude=None,
          polarizabilities=None, pol_cutoff=None, exclude_within='residue'):
    """ProteinEmbedding from PQR atoms.

    qm_residues: residue keys (chain, resseq, resname), one fragment each, in
    the order the fragments are wanted. exclude: {residue key: [atom names]}
    taken out of the QM region into the environment. polarizabilities:
    {element: alpha in Bohr^3}, or None for charges only. pol_cutoff: a
    tighter cutoff (Angstrom) for the polarizable sites than for the charges.
    """
    exclude = {} if exclude is None else exclude
    qm_residues = [tuple(r) for r in qm_residues]
    present = {a.residue for a in atoms}
    missing = [r for r in qm_residues if r not in present]
    if missing:
        raise ValueError(f'QM residues not in the structure: {missing}')
    is_qm = np.array([a.residue in qm_residues
                      and a.name not in exclude.get(a.residue, ())
                      for a in atoms])
    qm_idx = np.flatnonzero(is_qm)
    mm_idx = np.flatnonzero(~is_qm)
    xyz = np.array([a.xyz for a in atoms])

    # covalent cuts: QM atom -- environment atom
    cuts = [(i, j) for i in qm_idx for j in mm_idx
            if np.linalg.norm(xyz[i] - xyz[j]) < 3.0 and _bonded(atoms[i], atoms[j])]
    charges = np.array([a.charge for a in atoms])
    links = []
    for i, j in cuts:
        d = LINK_BOND_LENGTH.get(atoms[i].element)
        if d is None:
            raise ValueError(f'no link-atom bond length for a cut at '
                             f'{atoms[i].element}')
        bond = xyz[j] - xyz[i]
        links.append((i, xyz[i] + d * bond / np.linalg.norm(bond)))
        neigh = [k for k in mm_idx if k != j
                 and np.linalg.norm(xyz[k] - xyz[j]) < 3.0
                 and _bonded(atoms[j], atoms[k])]
        if not neigh:
            raise ValueError(f'boundary atom {atoms[j].name} of '
                             f'{atoms[j].residue} has no environment '
                             f'neighbour to take its charge')
        charges[neigh] += charges[j] / len(neigh)
        charges[j] = 0.0

    # QM molecule: fragment by fragment, link atoms with their QM partner
    qm_atoms, fragments = [], []
    for res in qm_residues:
        frag = []
        for i in qm_idx:
            if atoms[i].residue == res:
                frag.append(len(qm_atoms))
                qm_atoms.append((atoms[i].element, tuple(xyz[i])))
        for i, pos in links:
            if atoms[i].residue == res:
                frag.append(len(qm_atoms))
                qm_atoms.append(('H', tuple(pos)))
        fragments.append(frag)
    qm_charge = float(sum(atoms[i].charge for i in qm_idx))
    mol = gto.M(atom=qm_atoms, basis=basis, charge=int(round(qm_charge)),
                unit='Angstrom', verbose=0)
    qm_xyz = np.array([p for _, p in qm_atoms])

    # environment: whole residues within the cutoff
    def within(r):
        dmin = np.full(len(mm_idx), np.inf)
        for q in qm_xyz:
            dmin = np.minimum(dmin, np.linalg.norm(xyz[mm_idx] - q, axis=1))
        keep_res = {atoms[mm_idx[k]].residue for k in np.flatnonzero(dmin < r)}
        return np.array([k for k in mm_idx if atoms[k].residue in keep_res],
                        int)
    kept = within(cutoff)
    kept_q = kept[np.abs(charges[kept]) > 0.0]
    point = PointCharges(xyz[kept_q], charges[kept_q], unit='Angstrom')
    dist = (np.linalg.norm(xyz[kept_q][:, None] - qm_xyz[None], axis=2).min()
            if kept_q.size else np.inf)

    sites, dropped = None, []
    if polarizabilities is not None:
        pol = within(cutoff if pol_cutoff is None else pol_cutoff)
        clear = MIN_SITE_TO_QM_DISTANCE * BOHR_TO_ANGSTROM
        ok = []
        for k in pol:
            d = np.linalg.norm(qm_xyz - xyz[k], axis=1).min()
            if d < clear:
                dropped.append((atoms[k].residue, atoms[k].name, float(d)))
            elif atoms[k].element not in polarizabilities:
                raise ValueError(f'no polarizability for element '
                                 f'{atoms[k].element}')
            else:
                ok.append(k)
        ok = np.array(ok, int)
        if exclude_within == 'residue':
            groups = [hash(atoms[k].residue) for k in ok]
        elif exclude_within is None:
            groups = None
        else:
            raise ValueError(f"exclude_within must be 'residue' or None, got "
                             f"{exclude_within!r}")
        sites = PolarizableSites(xyz[ok], [polarizabilities[atoms[k].element]
                                           for k in ok], unit='Angstrom',
                                 mol=mol, groups=groups)
    env = point if sites is None else CompositeEnvironment(point, sites)
    report = dict(n_qm=len(qm_atoms), n_links=len(links), n_charges=len(kept_q),
                  n_sites=0 if sites is None else sites.nsites,
                  environment_charge=float(charges[kept].sum()),
                  qm_charge_in_file=qm_charge, dropped_sites=dropped,
                  closest_charge=float(dist),
                  cut_bonds=[(atoms[i].residue, atoms[i].name, atoms[j].name)
                             for i, j in cuts])
    return ProteinEmbedding(mol=mol, fragments=fragments, environment=env,
                            charges=point, sites=sites, report=report)
