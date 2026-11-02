"""Relaxed torsion scans: barriers between conformers, and following one
state through a large twist.

A conformer population means something only if the conformers interconvert
on the time scale being modelled, and a twisted-ICT state relaxes along the
very torsion whose profile decides that. Both are a RELAXED SCAN: the
dihedral held at a sequence of values while everything else relaxes.

The hold is a harmonic restraint on the dihedral, added to any surface by
`TorsionRestrained`, which is itself a `PotentialEnergySurface`: the scan runs
through the existing optimizers (`optimize.relax`) and the surface underneath
is never told. The energy each point reports is the surface's own, the
restraint's share removed, and the residual |phi - target| is recorded beside
it (a stiffer `force_constant` shrinks it).

FOLLOWING A STATE THROUGH A SWEEP. Every point starts at the previous point's
minimum with the torsion advanced rigidly (`conformers.rotate_torsion`) and on
the surface the previous point ended on (refrozen at its minimum when
`refreeze` asks), so a surface that follows its root by overlap
(`ExcitedStateChain`) matches the new point's state against the previous
point's committed eigenvector -- the hand-over that `refreeze` already makes.
Across a large increment that overlap can fall below what the follower
accepts (`RootFollowingLost`, a `StepFailed`), the walk can fail its first
step, or the restrained relaxation can end unconverged: the increment is then
HALVED, up to `max_halvings` times, and the scan
stops with the reason recorded when even that does not hold. A 60 degree
sweep in 10 degree steps that the follower cannot hold becomes 5 and then
2.5 degree steps exactly where the state changes character, and nowhere
else.
"""
from dataclasses import dataclass, field

import numpy as np

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import SCAN_MAX_HALVINGS, TORSION_RESTRAINT_K
from src.Base.step_failed import StepFailed
from src.gradients.surface import PotentialEnergySurface
from src.properties.conformers import dihedral, rotate_torsion
from src.properties.curvilinear import wilson_b
from src.properties.optimize import relax



def wrapped(delta_deg):
    """An angle difference wrapped into [-180, 180) degrees."""
    return (np.asarray(delta_deg, float) + 180.0) % 360.0 - 180.0


def dihedral_gradient(coords, tor):
    """d phi / dx (natm, 3) in rad/Bohr of `conformers.dihedral`: the
    analytic B-matrix row of `curvilinear`'s dihedral, which is the same
    angle with the IUPAC sign, negated."""
    x = np.asarray(coords, float)
    return -wilson_b(x, ([], [], [(tor.i, tor.j, tor.k, tor.l)]))[0].reshape(-1, 3)


class TorsionRestrained(PotentialEnergySurface):
    """`surface` plus k (phi - target)^2 on one torsion.

    Everything the protocol asks is the inner surface's with the restraint
    added; `refreeze` refreezes the inner surface and keeps the restraint, and
    `surface` is the attribute a walk's `commit` reaches the state-following
    half through. Any other attribute is the inner surface's.
    """

    def __init__(self, surface, torsion, target_deg,
                 force_constant=TORSION_RESTRAINT_K):
        self.surface = surface
        self.torsion = torsion
        self.target = float(target_deg)
        self.k = float(force_constant)

    def __getattr__(self, name):
        if name == 'surface':
            raise AttributeError(name)
        return getattr(self.surface, name)

    def restraint(self, mol):
        """(energy, gradient (natm, 3)) of the restraint at `mol`."""
        x = np.asarray(mol.atom_coords(), float)
        t = self.torsion
        d = np.radians(wrapped(dihedral(x, t.i, t.j, t.k, t.l) - self.target))
        return (float(self.k * d * d),
                2.0 * self.k * d * dihedral_gradient(x, t))

    def total_energy(self, mol=None, mf=None):
        mol = self.surface.mol0 if mol is None else mol
        return self.surface.total_energy(mol, mf) + self.restraint(mol)[0]

    def total_gradient(self, mol=None, mf=None):
        mol = self.surface.mol0 if mol is None else mol
        grad, e, diags = self.surface.total_gradient(mol, mf)
        er, gr = self.restraint(mol)
        return np.asarray(grad) + gr, e + er, diags

    def refreeze(self, mol, mf=None):
        return TorsionRestrained(self.surface.refreeze(mol, mf), self.torsion,
                                 self.target, self.k)

    def label(self):
        return (f'{self.surface.label()} + torsion {self.torsion.i}-'
                f'{self.torsion.j}-{self.torsion.k}-{self.torsion.l} held at '
                f'{self.target:.1f} deg')

    def identity(self):
        t = self.torsion
        return {'surface': self.surface.identity(),
                'restraint': {'torsion': (t.i, t.j, t.k, t.l),
                              'target_deg': self.target, 'k': self.k}}

    def mean_field(self, mol=None, mf=None):
        return self.surface.mean_field(mol, mf)


@dataclass
class ScanPoint:
    """One relaxed point of a torsion scan.

    target, phi: the held and the reached dihedral, degrees. energy: the
    surface's total energy at the point (restraint removed), Hartree.
    restraint: what the restraint added there. surface: the surface the
    point ended on (refrozen at it when asked), which the next point starts
    on. halvings: how often the increment into this point was halved.
    """

    target: float
    phi: float
    energy: float
    restraint: float
    mol: object
    surface: object
    converged: bool
    halvings: int = 0
    info: dict = field(default_factory=dict)


def _relax_point(surface, mol, torsion, target, k, engine, refreeze,
                 relax_kw):
    """One restrained relaxation; (ScanPoint, None) or (None, reason)."""
    held = TorsionRestrained(surface, torsion, target, k)
    kw = dict(relax_kw)
    kw.setdefault('verbose', False)
    kw.setdefault('nudge', 0.0)        # the rotated start is already displaced
    try:
        mol_opt, info = relax(held, mol, engine=engine, refreeze=refreeze,
                              **kw)
    except StepFailed as exc:
        return None, f'{type(exc).__name__}: {exc}'
    if info.get('energy') is None:
        return None, info.get('status', 'no energy')
    final = info.get('final_surface', held)
    inner = getattr(final, 'surface', final)
    er = held.restraint(mol_opt)[0]
    t = torsion
    return ScanPoint(target=float(target),
                     phi=dihedral(mol_opt.atom_coords(), t.i, t.j, t.k, t.l),
                     energy=float(info['energy']) - er, restraint=er,
                     mol=mol_opt, surface=inner,
                     converged=bool(info.get('converged')), info=info), None


def torsion_scan(surface, mol, torsion, targets, force_constant=TORSION_RESTRAINT_K,
                 engine='auto', refreeze=0, max_halvings=SCAN_MAX_HALVINGS,
                 verbose=True, **relax_kw):
    """A relaxed scan of `torsion` through `targets` (degrees, in order).

    surface: any `PotentialEnergySurface`; mol: the start, whose own dihedral
        is where the sweep begins. Each point starts from the previous one's
        minimum turned rigidly to the next target, on the surface the
        previous point ended on (module docstring); refreeze: the outer
        refreeze loop of every point (0: one frozen surface for the whole
        scan, comparable energies; 1: conventions rebuilt at each point, the
        followed state handed over).

    An increment whose point fails (a lost root, a failed first step, a
    restrained relaxation that does not converge) is halved up to
    `max_halvings` times; then the scan stops. No point is passed on
    unconverged. Returns
    (points, status): the ScanPoints reached, in order, and 'complete' or the
    reason it stopped.
    """
    t = torsion
    here_mol, here_surface = mol, surface
    # where the last point stands (rotated from) and what it was held at
    # (halved from, so the targets stay the ones asked and their halves)
    here_phi = dihedral(mol.atom_coords(), t.i, t.j, t.k, t.l)
    here_target = here_phi
    points = []
    for target in targets:
        goal = float(target)
        halvings, attempt = 0, goal
        while True:
            step = float(wrapped(attempt - here_phi))
            start = molecule_at(here_mol, rotate_torsion(
                here_mol.atom_coords(), t, step))
            point, why = _relax_point(here_surface, start, t, attempt,
                                      force_constant, engine, refreeze,
                                      relax_kw)
            if point is not None and not point.converged:
                # not a point: the next one would start from a geometry
                # that is not the minimum at this torsion
                why = (f'the restrained relaxation at {attempt:.2f} deg did '
                       f'not converge ({point.info.get("status", "no status")})')
                point = None
            if point is not None:
                point.halvings = halvings
                points.append(point)
                if verbose:
                    print(f'  [scan] {point.target:8.2f} deg: phi '
                          f'{point.phi:8.2f}, E {point.energy:.8f} Ha'
                          + (f' (increment halved {halvings}x)'
                             if halvings else ''))
                here_mol, here_surface = point.mol, point.surface
                here_phi, here_target = point.phi, point.target
                if attempt == goal:
                    break
                attempt, halvings = goal, 0
                continue
            if halvings >= max_halvings:
                status = (f'stopped before {attempt:.2f} deg after '
                          f'{halvings} halvings: {why}')
                if verbose:
                    print(f'  [scan] {status}')
                return points, status
            halvings += 1
            attempt = here_target + 0.5 * float(wrapped(attempt - here_target))
    return points, 'complete'


def barrier(points, reference=None):
    """(height, point) of the highest scan point above `reference` (a total
    energy, Hartree; the lowest scan point when None): the relaxed-scan
    estimate of the barrier, an upper bound to the true saddle along a path
    that the scan coordinate describes, and nothing more."""
    if not points:
        raise ValueError('an empty scan has no barrier')
    e = np.array([p.energy for p in points])
    ref = e.min() if reference is None else float(reference)
    top = int(np.argmax(e))
    return float(e[top] - ref), points[top]
