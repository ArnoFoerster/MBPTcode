"""Herzberg-Teller intensity: the transition dipole's linear dependence on the
nuclei.

The Condon band weights every vibronic line by |mu0|^2 and the Franck-Condon
overlap. A weakly allowed or symmetry-forbidden transition -- an n -> pi*
band, a charge-transfer emitter with little HOMO/LUMO overlap -- takes much
or all of its intensity from the change of the dipole along the modes,

    mu(q) = mu0 + sum_k dmu/dq_k q_k,

which lends intensity to lines with one more or one fewer quantum in a
non-totally-symmetric mode (false origins) and interferes with the Condon
term along the totally symmetric ones. `TransitionDipole` holds mu0 and
dmu/dq at the minimum of one state on that state's mass-weighted normal
coordinates; `duschinsky.DuschinskyOscillators(model, dipole)` puts it into
the correlation function, Condon and Herzberg-Teller cross terms included,
for any Duschinsky model (the displaced oscillator is
`DuschinskyModel.from_displacement`, with the SIGNED displacements, because the cross
term knows the direction of the displacement and S_k does not).

dmu/dq comes from the caller -- an analytic derivative projected with
`dipole_from_cartesian` -- or from central differences of the transition
dipole along the normal modes (`dipole_derivative`), with any routine that
returns the dipole at a geometry. That routine must return dipoles of ONE
phase: the sign of an excited state's vector is the eigensolver's, and two
displaced solves that come back with opposite signs turn the slope into a
curvature. `dipole_derivative` cannot fix a phase it is not told, so it
warns (`TransitionDipolePhase`) where the even part of the difference
dominates the odd part.

`PhaseTrackedDipole` is such a routine for an excited state from any
Casida-type solve (BSE@GW, RPA): every displaced solve is aligned to the
reference one by the transported overlap of their eigenvectors, in the RPA
metric <X|X> - <Y|Y> the solve normalizes them in and with the orbitals
matched by `body_overlap` (`gradients.state_identity`, the overlap root
following reads). A group of near-degenerate roots is matched as a subspace:
the orthogonal rotation of the displaced roots closest to the reference ones
(orthogonal Procrustes, the polar factor of the overlap block), which is the
per-root sign when the group is one root. Where the group's smallest overlap
singular value falls below ROOT_FOLLOW_WEIGHT_MIN the displaced solve does
not contain the reference roots -- a step too large, or too few roots solved
-- and the dipole is refused.

Atomic units: dipoles in e Bohr, q in Bohr sqrt(m_e).
"""
import warnings
from dataclasses import dataclass

import numpy as np

from src.Base.numerics.constants import ROOT_FOLLOW_WEIGHT_MIN
from src.gradients.state_identity import transported_overlap
from src.properties.duschinsky import log_correlation
from src.properties.vibronic import displace_along_mode

#: Dimensionless normal-coordinate step of the central difference of the
#: transition dipole: Q_k = q_k sqrt(omega_k), so 0.05 is a twentieth of the
#: zero-point amplitude 1/sqrt(2) of every mode alike.
HT_FD_STEP = 0.05
#: Floor, relative to the largest |mu| at the stencil, below which the
#: even/odd test of `dipole_derivative` does not judge a mode: a mode along
#: which the dipole does not change at all has nothing to get wrong.
HT_PHASE_FLOOR = 1e-6
#: Hartree. Reference roots this close to the followed one are matched with
#: it as one subspace by `PhaseTrackedDipole`: over an HT_FD_STEP displacement
#: two roots closer than ~27 meV can exchange character, and only their span
#: is the same at both geometries.
HT_DEGENERACY_WINDOW = 1e-3

REFERENCES = ('ground', 'excited')


class TransitionDipolePhase(RuntimeWarning):
    """Along a mode the dipole's change is mostly EVEN in the step: a phase
    flip between the two displaced solves, or a dipole far from linear over
    HT_FD_STEP. The slope along that mode is not a derivative."""


@dataclass(frozen=True)
class TransitionDipole:
    """mu(q) = mu0 + dmu_dq^T q about the minimum of `reference`.

    mu0: (3,) at that minimum; dmu_dq: (N, 3), row k the derivative along
    that state's mass-weighted normal coordinate q_k (`DuschinskyModel`'s
    modes of that state); reference: 'ground' or 'excited'; suspect_modes:
    the modes `dipole_derivative` warned about (`TransitionDipolePhase`).
    """

    mu0: np.ndarray
    dmu_dq: np.ndarray
    reference: str
    suspect_modes: tuple = ()

    def __post_init__(self):
        if self.reference not in REFERENCES:
            raise ValueError(f'reference is one of {REFERENCES}, got '
                             f'{self.reference!r}')
        if np.shape(self.mu0) != (3,) or np.ndim(self.dmu_dq) != 2 or \
                np.shape(self.dmu_dq)[1] != 3:
            raise ValueError(f'mu0 is (3,) and dmu_dq (N, 3); got '
                             f'{np.shape(self.mu0)}, {np.shape(self.dmu_dq)}')

    @classmethod
    def condon(cls, mu0, n_modes, reference='ground'):
        """The Condon dipole: mu0 everywhere."""
        return cls(np.asarray(mu0, float), np.zeros((n_modes, 3)), reference)

    def in_initial(self, model, direction):
        """(mu0', m') with mu = mu0' + m'^T q_i in the initial state's
        coordinates of `model.oriented(direction)`.

        About the other state's minimum, q_f = J q_i + K turns the expansion
        into mu0' = mu0 + m^T K, m' = J^T m: the same linear function of the
        nuclei, so the band does not depend on which minimum the dipole was
        expanded about beyond what the linear model itself assumes.
        """
        _, _, J, K = model.oriented(direction)
        mu0 = np.asarray(self.mu0, float)
        m = np.asarray(self.dmu_dq, float)
        if m.shape[0] != len(K):
            raise ValueError(f'{m.shape[0]} dipole derivatives for a model of '
                             f'{len(K)} modes')
        initial = 'ground' if direction == 'absorption' else 'excited'
        if self.reference == initial:
            return mu0, m
        return mu0 + m.T @ K, J.T @ m

    def strength(self, model, temperature, direction):
        """<|mu(q)|^2> over the initial state's thermal ensemble: the
        integrated intensity the band carries, |mu0|^2 in the Condon case."""
        return float(np.exp(log_correlation(
            model, np.array([0.0]), temperature, direction,
            dipole=self.in_initial(model, direction))[0]).real)


def dipole_from_cartesian(mu0, dmu_dx, state, reference):
    """A `TransitionDipole` from a Cartesian derivative dmu_dx, (natm, 3, 3)
    [atom, x/y/z of the atom, dipole component] in `state.mol`'s frame,
    projected on `state`'s mass-weighted modes: dmu/dq = L^T M^-1/2 dmu/dx."""
    d = np.asarray(dmu_dx, float).reshape(-1, 3)
    if d.shape[0] != state.modes.shape[0]:
        raise ValueError(f'dmu_dx has {d.shape[0]} Cartesian rows for '
                         f'{state.modes.shape[0] // 3} atoms')
    w = np.repeat(np.asarray(state.masses, float), 3) ** -0.5
    return TransitionDipole(np.asarray(mu0, float), state.modes.T @ (w[:, None] * d),
                            reference)


def dipole_derivative(dipole_at, state, reference, step=HT_FD_STEP,
                      map_fn=None):
    """dmu/dq by central differences along `state`'s normal modes.

    dipole_at(mol) -> (3,) transition dipole at that geometry, ONE phase at
        every geometry (see the module docstring); `state` a
        `duschinsky.HarmonicState` (its minimum, modes, masses, omega);
        reference: which state's minimum `state` is ('ground' or 'excited').
    step: the dimensionless step Q_k; the derivative is taken per mass-
        weighted q_k = Q_k / sqrt(omega_k).
    map_fn: map's signature, default the builtin; where a pool spreads the
        2 N + 1 dipole evaluations.

    Returns a `TransitionDipole` (its `suspect_modes` the ones below);
    `TransitionDipolePhase` warns for every mode
    whose even part |mu+ + mu- - 2 mu0| exceeds its odd part |mu+ - mu-|
    (the test that sees a flip of a forbidden transition's dipole), or along
    which a displaced dipole points against a mu0 longer than half the
    linear change, |mu+ - mu-| / 4 (an allowed one's; a forbidden one's mu0
    is roundoff, far shorter than its change, and is not tested). A routine that fixes the phase itself
    (`PhaseTrackedDipole`, `phase_tracked = True`) is not tested: there an
    even part is the dipole's curvature -- along a mode whose slope vanishes
    by symmetry it is all there is -- and not a flip.
    """
    runner = map if map_fn is None else map_fn
    n = len(state.omega)
    mols = [state.mol]
    for k in range(n):
        for sign in (1.0, -1.0):
            mols.append(displace_along_mode(state.mol, state.modes,
                                            state.masses, state.omega, k,
                                            sign * step))
    mus = np.array([np.asarray(m, float).reshape(3)
                    for m in runner(dipole_at, mols)])
    mu0, plus, minus = mus[0], mus[1::2], mus[2::2]
    dq = step / np.sqrt(np.asarray(state.omega, float))
    slope = (plus - minus) / (2.0 * dq[:, None])
    odd = np.linalg.norm(plus - minus, axis=1)
    even = np.linalg.norm(plus + minus - 2.0 * mu0, axis=1)
    floor = HT_PHASE_FLOOR * float(np.abs(mus).max())
    # an allowed dipole keeps its direction over the step: a displaced one
    # pointing against a mu0 longer than half the linear change has flipped
    norm0 = np.linalg.norm(mu0)
    turned = (norm0 > floor) & (norm0 > 0.25 * odd) & (
        np.minimum(plus @ mu0, minus @ mu0) < 0.0)
    bad = np.flatnonzero(((even > odd) & (even > floor)) | turned)
    if getattr(dipole_at, 'phase_tracked', False):
        bad = np.array([], int)
    if bad.size:
        warnings.warn(f'dipole_derivative: along modes {bad.tolist()} the '
                      f'dipole changes mostly evenly in the step (|even| '
                      f'{np.round(even[bad], 8).tolist()} > |odd| '
                      f'{np.round(odd[bad], 8).tolist()} e Bohr), or a '
                      f'displaced dipole points against mu0: a phase flip '
                      f'between the displaced solves, or a dipole far from '
                      f'linear over the step', TransitionDipolePhase)
    return TransitionDipole(mu0, slope, reference,
                            tuple(int(k) for k in bad))


@dataclass(frozen=True)
class ExcitedRoots:
    """The roots of one Casida-type solve at one geometry, as the phase
    tracking reads them: mol and mo_coeff of the reference determinant, nocc,
    omega (n,), X and Y (nocc * nvir, n) normalized <X|X> - <Y|Y> = 1 (Y None
    under Tamm-Dancoff), dipole (n, 3) the transition dipoles <0|r|n>."""

    mol: object
    mo_coeff: np.ndarray
    nocc: int
    omega: np.ndarray
    X: np.ndarray
    Y: object
    dipole: np.ndarray

    @classmethod
    def from_solve(cls, mf, omega, X, Y, info, mol=None):
        """From `solve_bse`'s (omega, X, Y, info) on mean field `mf`."""
        mol = mf.mol if mol is None else mol
        if info.get('transition_dipole') is None:
            raise ValueError('the solve returned no transition dipoles (a '
                             'triplet has none to differentiate)')
        return cls(mol, np.asarray(mf.mo_coeff, float), mol.nelectron // 2,
                   np.asarray(omega, float), np.asarray(X, float),
                   None if Y is None else np.asarray(Y, float),
                   np.asarray(info['transition_dipole'], float))


def degenerate_group(omega, root, window=HT_DEGENERACY_WINDOW):
    """The roots chained to `root` by gaps below `window` (Hartree)."""
    omega = np.asarray(omega, float)
    order = np.argsort(omega)
    at = int(np.flatnonzero(order == root)[0])
    lo, hi = at, at
    while lo > 0 and omega[order[lo]] - omega[order[lo - 1]] < window:
        lo -= 1
    while hi < len(order) - 1 and omega[order[hi + 1]] - omega[order[hi]] < window:
        hi += 1
    return tuple(int(k) for k in sorted(order[lo:hi + 1]))


def subspace_alignment(s, nstates):
    """(rotation, columns, sigma_min) carrying displaced roots onto reference
    ones.

    s: (n_ref, n_disp) overlaps <ref_i|disp_j> of the reference group with
    every displaced root. The n_ref displaced roots of largest weight
    sum_i s_ij^2 span the displaced group; on their block M = U sigma V^T the
    orthogonal R = U V^T maximizes tr(M R^T), and the aligned displaced roots
    sum_j R_ij |disp_j> overlap the reference ones by the symmetric positive
    U sigma U^T. For one root R is the sign of s. sigma_min, the smallest
    singular value, is how much of the reference group the displaced solve
    holds.
    """
    s = np.asarray(s, float)
    if s.shape[1] < nstates:
        raise ValueError(f'{s.shape[1]} displaced roots for a group of '
                         f'{nstates}')
    weight = (s ** 2).sum(axis=0)
    columns = np.sort(np.argsort(weight)[::-1][:nstates])
    u, sigma, vt = np.linalg.svd(s[:, columns])
    return u @ vt, columns, float(sigma.min())


def aligned_dipoles(reference, displaced, group):
    """(len(group), 3) transition dipoles of the displaced solve in the
    reference roots' phase and rotation (`subspace_alignment` on the
    transported overlap). Refuses (ValueError) a displaced solve that holds
    less than ROOT_FOLLOW_WEIGHT_MIN of the group."""
    group = list(group)
    y_ref = None if reference.Y is None else reference.Y[:, group]
    s = transported_overlap((reference.mol, reference.mo_coeff),
                            (displaced.mol, displaced.mo_coeff),
                            reference.nocc, reference.X[:, group], y_ref,
                            displaced.X, displaced.Y)[1:, 1:]
    rot, columns, sigma_min = subspace_alignment(s, len(group))
    if sigma_min < ROOT_FOLLOW_WEIGHT_MIN:
        raise ValueError(
            f'the displaced solve holds the reference roots {group} only to '
            f'an overlap singular value {sigma_min:.3f} (< '
            f'{ROOT_FOLLOW_WEIGHT_MIN}): the step is too large or too few '
            f'roots were solved; no phase can be assigned')
    return rot @ displaced.dipole[columns]


class PhaseTrackedDipole:
    """dipole_at(mol) for `dipole_derivative`: root `root` of `solve_at`, in
    the phase of the solve at `mol` (the reference).

    solve_at(mol) -> `ExcitedRoots` at that geometry, enough roots to hold the
    followed one's group at every displaced geometry. The group is the
    reference roots within `window` (Hartree) of the followed one
    (`degenerate_group`); each displaced solve is aligned to it as a subspace
    (`aligned_dipoles`) and the followed root's row returned. At the reference
    geometry the reference dipole is returned without a second solve.
    """

    phase_tracked = True

    def __init__(self, solve_at, mol, root, window=HT_DEGENERACY_WINDOW):
        self.solve_at = solve_at
        self.reference = solve_at(mol)
        self.root = int(root)
        self.group = degenerate_group(self.reference.omega, self.root, window)
        self.position = self.group.index(self.root)

    def __call__(self, mol):
        if np.array_equal(mol.atom_coords(), self.reference.mol.atom_coords()):
            return self.reference.dipole[self.root]
        return aligned_dipoles(self.reference, self.solve_at(mol),
                               self.group)[self.position]
