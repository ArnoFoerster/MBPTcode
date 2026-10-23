"""Declarations, runs and stand-ins the photophysics tests share.

The gas-phase production chain on formaldehyde/cc-pVDZ, every policy named:
`declaration(**changes)` with the command line's production numerics
(`cli.production_numerics`) and the tests' own choices (G3 factors, a tight
SCF gradient, no warm start, the Cartesian walk), and `run_at_r0`, a `Run` in
a fresh store whose s0 record holds the input geometry.
"""
import ast
from dataclasses import replace

import numpy as np
from pyscf import gto

from src.Base.numerics.constants import AMU_TO_ME
from src.Base.declaration import QPStates
from src.photophysics.cli import declaration_of, parser
from src.photophysics.minimum import GroundModes
from src.photophysics.stages import Run, ground_minimum
from src.photophysics.store import Store
from src.gradients.gwbse.surface_factory import reference_mean_field

#: The formaldehyde geometry of the production gate, Angstrom.
FORMALDEHYDE = ('C 0.0 0.0 -0.5296; O 0.0 0.0 0.6742; '
                'H 0.0 0.9429 -1.1123; H 0.0 -0.9429 -1.1123')

#: Formaldehyde as the toy-spectrum tests place it, Angstrom: planar, C-O
#: 1.2, the H atoms 0.94 off the axis.
FORMALDEHYDE_TOY = 'C 0 0 0; O 0 0 1.2; H 0.94 0 -0.6; H -0.94 0 -0.6'

#: Every required option of the command line, at the production values.
REQUIRED = ['--temperature', '300', '--lambda-env', '0.05', '--nudge',
            '0.02', '--no-far-start', '--refreeze', '1', '--saddle-check',
            '0', '--df-check', '0', '--hessian', 'analytic-df',
            '--relaxed-solvation', 'equilibrium', '--span', 'full',
            '--manifold-gaps', 'stacked']


def argv(*extra, out='unused', atom=FORMALDEHYDE):
    """The command line of the tests' formaldehyde run, `extra` appended
    (argparse takes the last of a repeated option)."""
    return (['--system', 'formaldehyde', '--atom', atom, '--basis', 'cc-pvdz',
             '--environment', 'gas', '--grid-accuracy', 'G3',
             '--scf-grad-tol', '1e-11', '--no-scf-warm-start', '--engine',
             'cartesian', '--out', str(out)] + REQUIRED + list(extra))


def declaration(*extra, qp_states=None, **changes):
    """The tests' declaration: `extra` command-line options, then
    `changes` to its fields; `qp_states` a `QPStates` kind ('admitted' is
    the walk tests' set)."""
    d = declaration_of(parser().parse_args(argv(*extra)))
    if qp_states is not None:
        changes['qp_states'] = (QPStates(kind=qp_states)
                                if isinstance(qp_states, str) else qp_states)
    return replace(d, **changes) if changes else d


def run_in(store_root, decl, max_memory=4000):
    """A `Run` of `decl` on a store in `store_root`."""
    return Run(decl, Store(store_root, decl), max_memory)


def run_of(store_root, *extra, max_memory=4000, **changes):
    """A `Run` on a store in `store_root` of `declaration(*extra,
    **changes)`."""
    return run_in(store_root, declaration(*extra, **changes), max_memory)


def production(store_root, environment, solvation, *extra):
    """(the photophysics Run, the S1 surface at R0 it would walk): the walk
    gate's configuration, the run's own factory and declaration; `extra`
    command-line options appended."""
    run = run_of(store_root, '--environment', environment,
                 '--relaxed-solvation', solvation, *extra,
                 qp_states='admitted')
    mol0 = run.mol_in
    mf0 = reference_mean_field(mol0, run.factory, run.relaxed_spec.environment)
    return run, run.excited_at(run.relaxed_spec, mol0, mf0)


def run_at_r0(store_root, decl, max_memory=4000):
    """A `Run` of `decl` on a store in `store_root` whose s0 record is the
    input geometry, read back as R0 (no ground-state walk)."""
    run = run_in(store_root, decl, max_memory)
    if not run.store.has('s0'):
        run.write('s0', {'coords_bohr': run.mol_in.atom_coords()})
    ground_minimum(run)
    return run


class Stop(Exception):
    """Raised by the stub that stands at the point under test."""


def refusing(message):
    """A stub that fails the test with `message` when it is called."""
    def refused(*_, **__):
        raise AssertionError(message)
    return refused


def gw_bse_props(mol, energy, omega, socme, emitter='S1', reservoir='T1'):
    """A minimum's GW-BSE `props` on `mol`: fixed oscillator strengths and
    dipoles, zero ground-state and spin-orbit derivatives."""
    return {'coords_bohr': mol.atom_coords().tolist(), 'e0': -1.0,
            'energy': energy, 'omega': omega,
            'emitter': emitter, 'reservoir': reservoir,
            'f_osc': [0.1, 0.2, 0.3], 'dipole': [[1, 0, 0], [0, 2, 0],
                                                  [0, 0, 3]],
            'socme': socme, 'd_ground': np.zeros((mol.natm, 3)).tolist(),
            'soc_dv_cart': np.zeros((3, mol.natm, 3)).tolist(),
            'soc_dv_paths': {}}


def unit_modes(mol, omega):
    """`GroundModes` of `mol` at frequencies `omega`: the first Cartesian
    unit vectors as modes, unit masses."""
    return GroundModes(omega, np.eye(3 * mol.natm)[:, :len(omega)],
                       np.ones(mol.natm), 'df')


def internal_modes_of(mol):
    """(modes (3 natm, 6), masses): orthonormal internal modes of `mol`,
    mass-weighted, orthogonal to translations and rotations."""
    masses = np.asarray(mol.atom_mass_list(isotope_avg=True)) * AMU_TO_ME
    x = mol.atom_coords()
    com = (masses[:, None] * x).sum(0) / masses.sum()
    sq = np.sqrt(np.repeat(masses, 3)).reshape(-1, 3)
    rigid = [(sq * np.eye(3)[a]).ravel() for a in range(3)]
    rigid += [(sq * np.cross(np.eye(3)[a], x - com)).ravel()
              for a in range(3)]
    q, _ = np.linalg.qr(np.array(rigid).T, mode='complete')
    return q[:, 6:12], masses


# ------------------------------------------- the double-well toy surface
# planar NH3: N at the centre, three H at 2 Bohr, 120 degrees apart
PLANAR = np.array([[0.0, 0.0, 0.0]]
                  + [[2.0 * np.cos(t), 2.0 * np.sin(t), 0.0]
                     for t in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)])
SPRING = 0.5          # Ha/Bohr^2, every pair distance
WELL_A = 0.05         # Ha/Bohr^2: curvature -2 WELL_A at the plane
WELL_C = 0.15625      # Ha/Bohr^4: |h| = 0.4 at the minima without the springs
FD = 1e-5


def pyramidalization(x):
    """Signed distance of atom 0 from the plane of atoms 1-3."""
    n = np.cross(x[2] - x[1], x[3] - x[1])
    return float(n @ (x[0] - x[1]) / np.linalg.norm(n))


def well_energy(x):
    """Spring network at the planar rest distances plus the double well."""
    rest = np.linalg.norm(PLANAR[:, None] - PLANAR[None], axis=-1)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    h = pyramidalization(x)
    return (0.25 * SPRING * ((d - rest) ** 2).sum()
            - WELL_A * h ** 2 + WELL_C * h ** 4)


class GroundWell:
    """The double well as a ground-state surface, its central-difference
    force."""

    def __init__(self, mol):
        self.mol0 = mol

    def total_energy(self, mol=None, mf=None):
        return well_energy((mol or self.mol0).atom_coords())

    def total_gradient(self, mol=None, mf=None):
        x = (mol or self.mol0).atom_coords()
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            for k in range(3):
                xp, xm = x.copy(), x.copy()
                xp[i, k] += FD
                xm[i, k] -= FD
                g[i, k] = (well_energy(xp) - well_energy(xm)) / (2 * FD)
        return g, well_energy(x), {'omega': 0.1}

    def refreeze(self, mol):
        return self

    def label(self):
        return 'double well'


class DoubleWell(GroundWell):
    """The double well with an `excitation`: an excited-state surface to
    `start_nudge`."""

    def excitation(self, mol=None, mf=None):
        return 0.1


def molecule(coords):
    return gto.M(atom=[('N' if i == 0 else 'H', tuple(c))
                       for i, c in enumerate(coords)],
                 unit='Bohr', basis='sto-3g', verbose=0)


# ------------------------------------------- what the source scans look for
def calls(tree, attr):
    """Line numbers of the calls in `tree` to a function or method `attr`."""
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, 'attr', getattr(n.func, 'id', None)) == attr]


def occupied_counts_by_hand(tree):
    """Line numbers of `mo_occ > 0` in `tree`, the mask itself or wrapped in
    one call (`np.asarray(mf.mo_occ) > 0`), on any object."""
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Compare) and len(n.ops) == 1
                and isinstance(n.ops[0], ast.Gt)
                and isinstance(n.comparators[0], ast.Constant)
                and n.comparators[0].value == 0):
            continue
        left = n.left
        if isinstance(left, ast.Call) and len(left.args) == 1:
            left = left.args[0]
        if isinstance(left, ast.Attribute) and left.attr == 'mo_occ':
            out.append(n.lineno)
    return out


def halves_electrons(tree):
    """Line numbers of `<x>.nelectron // 2` in `tree`."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv)
            and isinstance(n.left, ast.Attribute)
            and n.left.attr == 'nelectron'
            and isinstance(n.right, ast.Constant) and n.right.value == 2]
