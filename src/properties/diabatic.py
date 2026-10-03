"""Finite-difference nuclear derivatives of the fragment-diabatic BSE matrix.

The reference the analytic diabatic gradient (`src.gradients.fragment_diabatic`)
has to reproduce, in the role `nonadiabatic.derivative_couplings` plays for the
derivative coupling and `FiniteDifferenceGradient` for forces. It is also the
cheap route when only a few nuclear directions matter: one displaced pair of
calculations gives the derivative of EVERY diabatic element along one
direction, where the analytic route spends one reverse pass per element.

AT EVERY DISPLACED GEOMETRY EVERYTHING IS REDONE: the mean field, the
quasiparticle energies and screening, the fragment localization, the diabats
and the resolvent. Nothing is frozen, so the difference is the derivative of
the quantity as defined, localization response included. What is carried over
from the reference geometry is IDENTITY, not values:

* the localization starts from the reference's local orbitals transported to
  the displaced geometry and is matched to them orbital by orbital, so local
  orbital k is the same orbital on both sides (`FragmentOrbitals.from_mf`
  with `reference=`);
* each diabat's sign is fixed by its overlap with the reference diabat in that
  matched local basis, and a diabat that overlaps its reference by less than
  `DIABAT_OVERLAP_FLOOR` is refused rather than differenced;
* Omega_0 is the reference's, held fixed.

`step_ladder` repeats the central difference over a ladder of steps and
Richardson-extrapolates; a localization that jumps between local maxima, or a
diabat that swaps, shows up there as a ladder that does not converge as h^2.
"""
import numpy as np

from src.Base.constants import DIABAT_OVERLAP_FLOOR, NUCLEAR_FD_STEP
from src.Base.fragment_localization import FragmentOrbitals
from src.properties.fragment_bse import BSEOperator, FragmentPartition
from src.properties.nonadiabatic import mo_overlap, state_overlap


def diabatic_matrix(chain, fragments, sites, ct=None, omega0=None, mol=None,
                    mf=None, scheme='lowdin', reference=None, route='auto'):
    """FragmentPartition of `chain`'s TDA-BSE at one geometry.

    `reference`, a FragmentPartition at another geometry, fixes the local
    orbitals' identity, the diabats' signs and Omega_0 (see module docstring).
    """
    mol, mf = chain.mean_field(mol, mf)
    op, _ = BSEOperator.from_chain(chain, mol, mf, route=route)
    orbitals = FragmentOrbitals.from_mf(
        mf, fragments, scheme=scheme,
        reference=None if reference is None else reference.orbitals)
    part = FragmentPartition.build(
        op, orbitals, sites, ct,
        omega0=omega0 if reference is None else reference.omega0)
    if reference is not None:
        _align(part, reference)
    return part


def _align(part, reference):
    """Flip the signs of `part`'s diabats to overlap the reference's positively.

    The overlap is the transported state overlap of the CANONICAL amplitudes
    (`nonadiabatic.state_overlap`), not a dot product in the local basis:
    the fragment functional is nearly flat under rotations inside one
    fragment, so local orbitals of one fragment can rotate among themselves
    between geometries while the diabat, which depends only on the fragment
    subspaces, does not move.
    """
    ref_o, o = reference.orbitals, part.orbitals
    t = mo_overlap(ref_o.mol, ref_o.mo_coeff, o.mol, o.mo_coeff)
    pr, pc = reference.p_canonical(), part.p_canonical()
    ov = np.diag(state_overlap(t, o.nocc, pr, np.zeros_like(pr), pc,
                               np.zeros_like(pc))[1:, 1:])
    if np.abs(ov).min() < DIABAT_OVERLAP_FLOOR:
        k = int(np.abs(ov).argmin())
        raise ValueError(
            f'diabat {part.labels[k]} overlaps its reference by only '
            f'{abs(ov[k]):.3f}: it has mixed with another state of its block '
            f'over this displacement. Shorten the step.')
    s = np.sign(ov)
    part.p_local = part.p_local * s
    part.y_local = part.y_local * s
    for name in ('a_pp', 'sigma', 'dsigma'):
        setattr(part, name, getattr(part, name) * np.outer(s, s))


def displaced_along(mol, direction, h):
    out = mol.copy()
    out.set_geom_(np.asarray(mol.atom_coords()) + h * direction, unit='Bohr')
    out.build(False, False)
    return out


def diabatic_derivative(chain, fragments, sites, direction, ct=None,
                        omega0=None, step=NUCLEAR_FD_STEP, scheme='lowdin',
                        reference=None, route='auto'):
    """Central difference of A_eff(Omega_0), A_PP and Sigma along `direction`.

    `direction` is a (natm, 3) Cartesian displacement in Bohr per unit step;
    the result is d/dlambda at lambda = 0 of the matrices at R0 + lambda *
    direction. With direction = e_(atom, axis) it is one column of the
    Cartesian gradient of every element.
    """
    direction = np.asarray(direction, float)
    mol0 = chain.mol0
    if reference is None:
        reference = diabatic_matrix(chain, fragments, sites, ct, omega0,
                                    scheme=scheme, route=route)
    plus, minus = (diabatic_matrix(chain, fragments, sites, ct,
                                   mol=displaced_along(mol0, direction, s * step),
                                   scheme=scheme, reference=reference,
                                   route=route)
                   for s in (1.0, -1.0))
    out = {name: (getattr(plus, name) - getattr(minus, name)) / (2.0 * step)
           for name in ('a_pp', 'sigma', 'dsigma')}
    out['a_eff'] = out['a_pp'] + out['sigma']
    out['step'] = step
    out['labels'] = reference.labels
    return out, reference


def step_ladder(chain, fragments, sites, direction, ct=None, omega0=None,
                steps=None, scheme='lowdin', route='auto'):
    """Derivatives over a ladder of steps, and their Richardson extrapolation.

    steps default to (4, 2, 1, 0.5) x NUCLEAR_FD_STEP. Successive halvings of
    a central difference converge as h^2, so (4 D(h/2) - D(h)) / 3 removes the
    leading error; `ratio` is |D(h) - D(h/2)| / |D(h/2) - D(h/4)|, which is 4
    for a clean h^2 ladder and wanders when the localization or a diabat jumps.
    """
    steps = (tuple(NUCLEAR_FD_STEP * f for f in (4, 2, 1, 0.5))
             if steps is None else tuple(steps))
    reference = None
    ladder = []
    for h in steps:
        d, reference = diabatic_derivative(
            chain, fragments, sites, direction, ct, omega0, step=h,
            scheme=scheme, reference=reference, route=route)
        ladder.append(d['a_eff'])
    ladder = np.array(ladder)
    rich = (4.0 * ladder[1:] - ladder[:-1]) / 3.0
    diffs = np.abs(np.diff(ladder, axis=0)).max(axis=(1, 2))
    ratio = diffs[:-1] / np.maximum(diffs[1:], 1e-300)
    return dict(steps=steps, a_eff=ladder, richardson=rich, ratio=ratio,
                labels=reference.labels, reference=reference)
