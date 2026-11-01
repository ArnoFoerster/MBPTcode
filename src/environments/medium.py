"""The dielectric medium of a run: what it is solvated in, declared once.

A continuum is two dielectric constants and a cavity. The static constant
eps_s relaxes the ground state and every equilibrium state; the optical
constant eps_inf = n^2 screens what is too fast for the solvent nuclei to
follow. A solvent of pyscf's SMD table supplies both by name (n from its
refractive index, eps_s from its dielectric constant). A medium that is not in
that table, a host film for instance, gives both numbers: a medium with only
one of them is refused, because the other is not a default but a property of
the material.

THE CAVITY is one named convention for every method and every stage of a run:
a radius set and the scale it is multiplied by, chosen together
(`CAVITY_CONVENTIONS`). The ground-state SCF, the linear-response term, the
state-specific loop of every method and their forces all read the cavity of
the one `SolventScreening` the medium builds, so two parts of a run cannot sit
on different solute boundaries.

A `Medium` is a declaration. `screening(mol)` turns it into the environment
the run carries (`SolventScreening`, which moves to a new geometry with
`for_geometry`), and `record()` is what a record and a resume key carry of
it. A spec string (`parse`) names it on a command line:

    'gas'                       no medium (parse returns None)
    'toluene'                   a solvent of pyscf's SMD table
    'eps=5.0,n2=2.89'           both constants, eps_s and n^2
    'eps=5.0,n2=2.89,label=film' the same, with a name for the records
    'N2:EPS'                    both constants as n^2, then eps_s
    'toluene,cavity=uff-1.1'    any of these on a named cavity convention
                                (else DEFAULT_CAVITY)
"""
from dataclasses import dataclass

from pyscf.solvent.smd import solvent_db

from src.environments.solvent_vocabulary import GAS
from src.environments.solvent_screening import (RADII_SETS, SolventScreening,
                                                solvent_dielectrics)


# A cavity convention: the radius set and the scale its radii are multiplied
# by, named together. 'bondi-1.2' is pyscf's PCM default (modified Bondi radii
# x 1.2, also Q-Chem's), 'uff-1.1' Gaussian's (UFF radii x 1.1), 'cosmo' the
# COSMO-optimized radii unscaled.
CAVITY_CONVENTIONS = {
    'bondi-1.2': ('bondi', 1.2),
    'uff-1.1': ('uff', 1.1),
    'cosmo': ('cosmo', 1.0),
}
# The cavity of a medium that names none.
DEFAULT_CAVITY = 'bondi-1.2'


def cavity_settings(cavity):
    """{'radius_set', 'vdw_scale'} of the convention named `cavity`."""
    if cavity not in CAVITY_CONVENTIONS:
        raise ValueError(f'cavity {cavity!r}: one of {sorted(CAVITY_CONVENTIONS)}')
    radius_set, scale = CAVITY_CONVENTIONS[cavity]
    assert radius_set in RADII_SETS
    return {'radius_set': radius_set, 'vdw_scale': scale}


@dataclass(frozen=True)
class Medium:
    """A continuum by its two dielectric constants and its cavity convention.

    eps_static: the static constant eps_s; eps_optical: n^2. solvent: the SMD
    name both came from, or None for constants given as numbers; label: a
    name for the records of a medium given by numbers. Build one with
    `named` or `constants` (or `parse`).
    """

    eps_static: float
    eps_optical: float
    cavity: str = DEFAULT_CAVITY
    solvent: str = None
    label: str = None

    def __post_init__(self):
        cavity_settings(self.cavity)
        if not 1.0 <= self.eps_optical <= self.eps_static:
            raise ValueError(
                f'n^2 = {self.eps_optical} and eps_s = {self.eps_static}: a '
                f'medium needs 1 <= n^2 <= eps_s (the static constant contains '
                f'the electronic response and the nuclear one on top)')

    @classmethod
    def named(cls, solvent, cavity=DEFAULT_CAVITY):
        """A solvent of pyscf's SMD table: n^2 and eps_s from its entry."""
        if solvent.lower() not in solvent_db:
            raise ValueError(
                f'unknown solvent {solvent!r} (pyscf SMD table): give the medium '
                f'as eps=EPS_S,n2=N2 instead')
        eps_optical, eps_static = solvent_dielectrics(solvent)
        return cls(float(eps_static), float(eps_optical), cavity, solvent, None)

    @classmethod
    def constants(cls, eps_static, n2, cavity=DEFAULT_CAVITY, label=None):
        """A medium by its static constant and n^2, both required."""
        if eps_static is None or n2 is None:
            raise ValueError(
                f'a medium needs both its static constant and n^2 (got eps_s '
                f'{eps_static}, n^2 {n2}): neither has a default')
        return cls(float(eps_static), float(n2), cavity, None, label)

    @property
    def name(self):
        """The solvent's name, the label, or 'eps=..,n2=..'."""
        return (self.solvent or self.label
                or f'eps={self.eps_static:g},n2={self.eps_optical:g}')

    def screening(self, mol, **kwargs):
        """The `SolventScreening` of this medium around `mol`: n^2 screens
        the response, eps_s relaxes the ground state, on this cavity. Keyword
        arguments (method, lebedev_order, equilibrium, ...) go to it."""
        cavity = cavity_settings(self.cavity)
        if self.solvent is not None:
            return SolventScreening(mol, solvent=self.solvent, **cavity, **kwargs)
        return SolventScreening(mol, eps=self.eps_optical, eps_static=self.eps_static,
                                allow_static_eps=True, **cavity, **kwargs)

    def record(self):
        """What a record and a resume key carry of the medium."""
        radius_set, scale = CAVITY_CONVENTIONS[self.cavity]
        return {'name': self.name, 'solvent': self.solvent, 'label': self.label,
                'eps_static': self.eps_static, 'n2': self.eps_optical,
                'cavity': self.cavity, 'radius_set': radius_set, 'vdw_scale': scale}


def parse(spec, cavity=None):
    """The `Medium` a spec string names (module docstring), None for 'gas'.

    A spec may name its cavity itself (',cavity=NAME'); `cavity` is the one a
    spec without it takes (DEFAULT_CAVITY when None), and a spec naming
    another than a given `cavity` is refused."""
    spec = str(spec).strip()
    if spec.lower() == GAS:
        return None
    head, *items = [item.strip() for item in spec.split(',')]
    fields = {}
    if '=' in head:
        items = [head] + items
        head = None
    for item in items:
        key, sep, value = item.partition('=')
        key = key.strip().lower()
        if not sep or key not in ('eps', 'n2', 'label', 'cavity') or key in fields:
            raise ValueError(f'medium {spec!r}: expected NAME or eps=EPS_S,n2=N2'
                             f'[,label=NAME], each optionally with ,cavity=NAME')
        fields[key] = value.strip()
    named = fields.pop('cavity', None)
    if named is not None and cavity is not None and named != cavity:
        raise ValueError(f'medium {spec!r} names the cavity {named!r} and '
                         f'{cavity!r} was given as well')
    cavity = named or cavity or DEFAULT_CAVITY
    if head is not None and ':' in head:
        if fields:
            raise ValueError(f'medium {spec!r}: N2:EPS takes no other constants')
        n2, eps_static = (float(x) for x in head.split(':', 1))
        return Medium.constants(eps_static, n2, cavity)
    if head is not None:
        if fields:
            raise ValueError(f'medium {spec!r}: a named solvent takes its constants '
                             f'from the table; give eps=,n2= without a name')
        return Medium.named(head, cavity)
    return Medium.constants(
        None if 'eps' not in fields else float(fields['eps']),
        None if 'n2' not in fields else float(fields['n2']),
        cavity, fields.get('label'))


def describe(env):
    """What a record carries of the continuum `env` (a `SolventScreening`),
    None in the gas phase: its constants, model and cavity, the convention by
    name when the radius set and scale are one of `CAVITY_CONVENTIONS`."""
    if env is None:
        return None
    cavity = env.cavity_settings
    named = [name for name, (radius_set, scale) in CAVITY_CONVENTIONS.items()
             if cavity['radius_set'] == radius_set and cavity['vdw_scale'] == scale
             and cavity['radii_table'] is None and cavity['r_probe'] == 0.0]
    return {'solvent': env.solvent, 'eps_static': env.eps_static,
            'n2': env.eps_optical, 'method': env.method,
            'cavity': named[0] if named else None, 'radius_set': cavity['radius_set'],
            'vdw_scale': cavity['vdw_scale'], 'r_probe': cavity['r_probe'],
            'lebedev_order': cavity['lebedev_order'],
            'custom_radii_table': cavity['radii_table'] is not None,
            'omega_p': env.omega_p, 'ngrids': env.ngrids}
