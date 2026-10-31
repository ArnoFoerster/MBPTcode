"""Franck-Condon band shapes with Duschinsky rotation and frequency change.

The displaced oscillator of `franck_condon` gives both electronic states the
ground state's normal modes and frequencies. Here each state carries its own:
the final state's mass-weighted normal coordinates are a rotated and shifted
copy of the initial state's,

    q_f = J q_i + K,

J the Duschinsky matrix (the overlap of the two mode sets) and K the
displacement between the minima, each state with its own frequencies. The
Franck-Condon density is the Fourier transform of the thermal correlation
function

    G(t) = Tr[ rho_i e^{i H_i t} e^{-i H_f t} ],    H_s measured from its minimum,

so rho(dE) = (1/2 pi) int dt e^{-i dE t} G(t) is the density of the
vibrational energy x = -dE the transition leaves in the nuclei (the convention
of `fc_weighted_dos`, which this reduces to at J = 1 and equal frequencies).
With unequal frequencies the 0-0 line sits at the adiabatic gap plus the
zero-point energy difference ZPE_f - ZPE_i, which G carries as its phase
e^{i sum (omega_i - omega_f) t / 2}.

G is the Gaussian integral of the two Mehler kernels (the path-integral form
of Peng, Niu, Shuai et al. and of Baiardi, Bloino and Barone). Written with
the half-angle sines and cosines of the final state and the thermal factor
w = e^{i omega_i t - beta omega_i} of the initial one, every matrix in it is
finite at every real time and at T -> 0:

    G(t) = prod_k 2 sqrt(omega_ik omega_fk) (1 - e^{-beta omega_ik})
           e^{i (omega_ik - omega_fk) t / 2} / sqrt(det N1 det N2)
           x exp( -i K^T Omega_f S F^{-1} R K )

    P0 = J Omega_i + Omega_f J,   P1 = J Omega_i - Omega_f J,   E2 = e^{-i Omega_f t}
    N1 = P0 + E2 P1 - P1 w - E2 P0 w,   N2 = P0 - E2 P1 + P1 w - E2 P0 w
    R  = J Omega_i T_i J^T,   T_i = -i (1 - w) / (1 + w),   F = R C + Omega_f S

with C, S = cos, sin(Omega_f t / 2). The square root is continued along the
time grid from G(0) = 1: each step's phase increment of det N1 N2 is the
branch nearest the trapezoid of d/dt ln det N = tr(N^-1 dN/dt), and a step on
which the two disagree is halved, so the branch cannot jump however many
modes add up their phases.

The excited-state Hessian is INPUT: any object with `.mol` and `.hessian`
(`CartesianHessian`; `read_orca_hessian` reads an ORCA .hess file as data).
Nothing here computes one. The mode sets and the displacement may come from
different methods: `mode_sets` takes the two Hessians (a TD-DFT pair at the
TD-DFT minima, say), and `displacement_from_minima` or
`displacement_from_gradient` gives K from another method's own minima or
gradient.

A rectilinear model is only as good as the straight lines it is built on.
Between two minima a large torsion apart, the overlap of the two Cartesian
mode sets turns the torsion into the stretches, and the thermal amplitude of
a 10 cm^-1 torsion projected on a 3000 cm^-1 stretch puts tens of eV into
the band. `duschinsky_guard` refuses such a model by three numbers (the J^2
weight from soft onto stiff modes, the sum-rule lambda against the energies,
the band's first moment against lambda), each against a bar in
`Base.numerics.constants`.

CURVILINEAR COORDINATES. `mode_sets(..., coordinates='curvilinear')` takes J
from the valence internals instead (`curvilinear.curvilinear_duschinsky`,
after Reimers, J. Chem. Phys. 115, 9103 (2001), and Cerezo, Zuniga, Requena,
Avila Ferrer and Santoro, J. Chem. Theory Comput. 9, 4947 (2013)): a torsion
of one state maps onto the torsions of the other, not onto its stretches.
That J is not orthogonal, because each state carries its own kinetic metric,
and it is used as it is: the trace over the nuclear positions picks up the
Jacobian |det J|, and the correlation function above holds with J^-T in
place of J wherever J multiplies the initial frequencies,

    P0 = J^-T Omega_i + Omega_f J,   P1 = J^-T Omega_i - Omega_f J,
    R  = J^-T Omega_i T_i J^-1,

which is the same at an orthogonal J; G(0) = 1 still. The Herzberg-Teller
factor takes the same substitution, and J^-1 for J^T. Emission reads the
inverse map, q_g = J^-1 (q_e - K). At one geometry the curvilinear J is the
Cartesian one.

SOFT MODES. A harmonic mode at 10 cm^-1 is not harmonic at room temperature,
and its frequency change between the two states (10 -> 90 cm^-1, say) puts
kT (w_f^2/w_i^2 - 1)/2 of thermal energy into the band. `split_soft_modes`
takes the 'ground' modes below a cutoff and their image in the
'excited' coordinates out of the model as one classical, frequency-unchanged
displaced block: in each direction its reorganization lambda_soft (the
initial state's own along the block) is a classical broadening, a Gaussian
of mean lambda_soft and variance 2 lambda_soft kT (Ulstrup and Jortner, J.
Chem. Phys. 63, 4358 (1975)), its zero-point difference a shift of the gap,
and the rest stays a Duschinsky model.

Energies in Hartree, time in atomic units, temperature in Kelvin.
"""
import re
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         DUSCHINSKY_CUMULANT_MAX_HALVINGS,
                                         DUSCHINSKY_CUMULANT_TOL,
                                         DUSCHINSKY_CURVILINEAR_SV_MIN,
                                         DUSCHINSKY_FIRST_MOMENT_RATIO_MAX,
                                         DUSCHINSKY_SOFT_MODE_CM,
                                         DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX,
                                         DUSCHINSKY_STIFF_MODE_CM,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         RATE_FC_LAMBDA_RATIO_MAX)
from src.properties.franck_condon import (envelope_time, real_axis_density,
                                          require_vibrational_modes,
                                          thermal_energy)
from src.properties.curvilinear import (curvilinear_displacement,
                                        curvilinear_duschinsky)
from src.properties.optimize import cartesian_hessian
from src.properties.vibronic import align_to, normal_modes

#: Largest disagreement (radians) between the phase increment of det N1 N2
#: over one step of the time grid and its trapezoid estimate from
#: tr(N^-1 dN/dt), and largest turn of that rate across the step, before the
#: step is halved: the square root's branch follows the estimate, which no
#: number of modes adding up their phases can fool.
BRANCH_MAX_TURN = np.pi / 4
#: Halvings of one time step the branch continuation may make.
BRANCH_MAX_HALVINGS = 30
#: |s - 1| over the singular values s of the Duschinsky matrix above which the
#: two mode sets do not span one space (axis switching beyond what the Eckart
#: alignment removes, an alignment that failed): J is always replaced by its
#: nearest orthogonal matrix, and above this `DuschinskyNotOrthogonal` warns.
#: Water's HF/STO-3G and HF/6-31G minima, 0.04 A and 11 degrees apart, leave
#: 4.9e-3.
DUSCHINSKY_ORTHOGONALITY_TOL = 2e-2
#: The two ways of measuring J: the overlap of the Cartesian mode sets
#: (orthogonal), or the map between them through the valence internals.
COORDINATES = ('rectilinear', 'curvilinear')
#: First step, in units of the slowest period, at which the first two
#: cumulants of a band are read off ln G (they set the grid of
#: `vibronic_band_grid` and the Lorentzian kink correction of the real-axis
#: integral); `cumulants_of` halves it until the estimates settle.
CUMULANT_STEP = 1e-4


class DuschinskyNotOrthogonal(RuntimeWarning):
    """The overlap of two normal-mode sets is not orthogonal to
    DUSCHINSKY_ORTHOGONALITY_TOL: it was replaced by the nearest orthogonal
    matrix, and the deviation is in `DuschinskyModel.orthogonality`."""


# --------------------------------------------------------------- the Hessian
@dataclass(frozen=True)
class CartesianHessian:
    """A nuclear Hessian and the geometry it belongs to.

    mol: the molecule at the geometry the Hessian was taken at (a minimum of
        the state it describes).
    hessian: (3N, 3N) in Ha/Bohr^2, or pyscf's (N, N, 3, 3), in `mol`'s frame.
    source: what produced it ('pyscf analytic DF-RHF', 'ORCA TD-DFT', ...),
        carried into the model's record.
    """

    mol: object
    hessian: np.ndarray
    source: str = ''


def as_cartesian_hessian(obj):
    """`obj` as a `CartesianHessian`: anything with `.mol` and `.hessian`."""
    if isinstance(obj, CartesianHessian):
        return obj
    if not (hasattr(obj, 'mol') and hasattr(obj, 'hessian')):
        raise TypeError(f'{type(obj).__name__} is not a Hessian: it needs '
                        f'.mol (the geometry) and .hessian (Ha/Bohr^2)')
    return CartesianHessian(obj.mol, np.asarray(obj.hessian, float),
                            str(getattr(obj, 'source', '')))


def _orca_blocks(lines, start):
    """The value lines of one ORCA $-block after its header at `start`."""
    out = []
    for line in lines[start + 1:]:
        if line.strip().startswith('$'):
            break
        if line.strip():
            out.append(line)
    return out


def read_orca_hessian(path, template, source='ORCA'):
    """An ORCA .hess file as a `CartesianHessian` on `template`'s atoms.

    Reads `$hessian` (n x n in column blocks, Ha/Bohr^2) and the geometry of
    `$atoms` (Bohr). `template` is the caller's molecule at any geometry: the
    elements must match it atom by atom, and the returned `mol` is it moved to
    the file's geometry. The Hessian is read as data; which method made it is
    the caller's `source`.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    heads = {line.strip().split()[0]: i for i, line in enumerate(lines)
             if line.strip().startswith('$')}
    for need in ('$hessian', '$atoms'):
        if need not in heads:
            raise ValueError(f'{path}: no {need} block')
    body = _orca_blocks(lines, heads['$hessian'])
    n = int(body[0].split()[0])
    h = np.full((n, n), np.nan)
    row = 1
    while row < len(body):
        cols = [int(c) for c in body[row].split()]
        for r in range(n):
            parts = body[row + 1 + r].split()
            if int(parts[0]) != r:
                raise ValueError(f'{path}: $hessian row {r} reads {parts[0]}')
            h[r, cols] = [float(v) for v in parts[1:1 + len(cols)]]
        row += 1 + n
    if np.isnan(h).any():
        raise ValueError(f'{path}: $hessian is incomplete')
    atoms = _orca_blocks(lines, heads['$atoms'])
    natm = int(atoms[0].split()[0])
    symbols = [a.split()[0] for a in atoms[1:1 + natm]]
    coords = np.array([[float(v) for v in a.split()[2:5]]
                       for a in atoms[1:1 + natm]])
    if n != 3 * natm:
        raise ValueError(f'{path}: a {n} x {n} Hessian for {natm} atoms')
    mine = [re.sub(r'[^A-Za-z]', '', template.atom_symbol(i)).capitalize()
            for i in range(template.natm)]
    theirs = [re.sub(r'[^A-Za-z]', '', s).capitalize() for s in symbols]
    if mine != theirs:
        raise ValueError(f'{path}: atoms {theirs} are not the template\'s '
                         f'{mine}')
    return CartesianHessian(molecule_at(template, coords), h, source)


# ------------------------------------------------------------ the two states
@dataclass(frozen=True)
class HarmonicState:
    """One electronic state's harmonic expansion about its own minimum.

    omega: (N,) frequencies, Hartree; modes: (3 natm, N) mass-weighted,
    orthonormal; masses: (natm,) electron masses; mol: the minimum.
    """

    mol: object
    omega: np.ndarray
    modes: np.ndarray
    masses: np.ndarray
    source: str = ''


def harmonic_state(hessian, isotope_avg=True):
    """Normal modes of a `CartesianHessian` (rigid-body directions projected
    out); an imaginary or rigid mode is refused: the expansion point is not a
    minimum."""
    hess = as_cartesian_hessian(hessian)
    omega, modes, masses, _ = normal_modes(None, mol=hess.mol,
                                           hess=hess.hessian,
                                           isotope_avg=isotope_avg)
    require_vibrational_modes(omega, f'harmonic_state ({hess.source})')
    return HarmonicState(hess.mol, omega, modes, masses, hess.source)


def rotated_hessian(h, rot):
    """A Cartesian Hessian in the frame `coords @ rot` (every 3x3 block
    turned with the coordinates)."""
    n = h.shape[0] // 3
    big = np.kron(np.eye(n), rot)
    return big.T @ h @ big


@dataclass(frozen=True)
class DuschinskyModel:
    """q_e = J q_g + K between two harmonic states' mass-weighted normal
    coordinates, each state with its own frequencies.

    omega_g, omega_e: (N,) Hartree. J: (N, N). K: (N,), the ground minimum
    in the excited state's coordinates, Bohr sqrt(m_e). coordinates:
    'rectilinear', J orthogonal; or 'curvilinear', J the map through the
    valence internals, its singular values within
    [DUSCHINSKY_CURVILINEAR_SV_MIN, 1 / DUSCHINSKY_CURVILINEAR_SV_MIN].
    `orthogonality`: max |s - 1| over J's singular values (before it was
    made orthogonal, for a rectilinear model; 0 for one built from given J).
    """

    omega_g: np.ndarray
    omega_e: np.ndarray
    J: np.ndarray
    K: np.ndarray
    orthogonality: float = 0.0
    source: str = ''
    coordinates: str = 'rectilinear'

    def __post_init__(self):
        n = len(self.omega_g)
        if (np.shape(self.omega_e) != (n,) or np.shape(self.J) != (n, n)
                or np.shape(self.K) != (n,)):
            raise ValueError(
                f'a Duschinsky model of {n} ground modes needs {n} excited '
                f'frequencies, an {n} x {n} J and {n} displacements; got '
                f'{np.shape(self.omega_e)}, {np.shape(self.J)}, '
                f'{np.shape(self.K)}')
        require_vibrational_modes(self.omega_g, 'DuschinskyModel (ground)')
        require_vibrational_modes(self.omega_e, 'DuschinskyModel (excited)')
        if self.coordinates not in COORDINATES:
            raise ValueError(f'coordinates are {COORDINATES}, got '
                             f'{self.coordinates!r}')
        if self.coordinates == 'curvilinear':
            s = np.linalg.svd(self.J, compute_uv=False)
            bar = DUSCHINSKY_CURVILINEAR_SV_MIN
            if s.min() < bar or s.max() > 1.0 / bar:
                raise ValueError(
                    f'the curvilinear J has singular values {s.min():.3g}..'
                    f'{s.max():.3g}, outside [{bar:g}, {1.0 / bar:g}]: the '
                    f'two minima are too far apart for one linear map')
            return
        dev = np.abs(self.J.T @ self.J - np.eye(n)).max()
        if dev > DUSCHINSKY_ORTHOGONALITY_TOL:
            raise ValueError(
                f'J is not orthogonal (|J^T J - 1| = {dev:.3g}); build the '
                f'model with `duschinsky_model`, which orthogonalizes it')

    @classmethod
    def from_displacement(cls, omega, dq):
        """The displaced oscillator as a Duschinsky model: J = 1, one set of
        frequencies, and the SIGNED displacement dq_k of the excited minimum
        on the ground modes (`vibronic.huang_rhys_of`'s second return),
        S_k = omega_k dq_k^2 / 2."""
        omega = np.asarray(omega, float)
        return cls(omega, omega.copy(), np.eye(len(omega)),
                   -np.asarray(dq, float), source='displaced oscillator')

    def oriented(self, direction):
        """(omega_i, omega_f, J, K) with q_f = J q_i + K for the transition
        out of the initial state: 'absorption' starts on the ground state,
        'emission' on the excited one."""
        if direction == 'absorption':
            return self.omega_g, self.omega_e, self.J, self.K
        if direction == 'emission':
            if self.coordinates == 'curvilinear':
                inv = np.linalg.inv(self.J)
                return self.omega_e, self.omega_g, inv, -inv @ self.K
            return self.omega_e, self.omega_g, self.J.T, -self.J.T @ self.K
        raise ValueError(f"direction is 'absorption' or 'emission', got "
                         f'{direction!r}')

    def correlation(self, temperature, direction, dipole=None):
        """The `_Correlation` of one direction: ln G(t) on a time grid."""
        wi, wf, j, k = self.oriented(direction)
        if self.coordinates == 'rectilinear':
            return _Correlation(wi, wf, j, k, temperature, dipole=dipole)
        return _Correlation(wi, wf, j, k, temperature, dipole=dipole,
                            left=np.linalg.inv(j).T)

    def huang_rhys(self):
        """S_k = omega_e,k K_k^2 / 2 on the excited modes: the displacement
        part alone, for comparison with the displaced-oscillator routes."""
        return self.omega_e * self.K ** 2 / 2.0

    def zero_point_shift(self):
        """ZPE_e - ZPE_g: where the 0-0 line sits above the adiabatic gap."""
        return 0.5 * float(self.omega_e.sum() - self.omega_g.sum())


def duschinsky_model(ground, excited, coordinates='rectilinear'):
    """The Duschinsky model between two Hessians at their own minima.

    ground, excited: `CartesianHessian`s (or anything with `.mol` and
        `.hessian`). The excited minimum and its Hessian are turned onto the
        ground minimum's frame first (mass-weighted Kabsch, `align_to`), so a
        Hessian computed in another orientation -- another program's standard
        frame -- gives the same model.

    J = L_e^T L_g and K = L_e^T M^{1/2} (x_g - x_e) on the mass-weighted modes,
    J replaced by its nearest orthogonal matrix (polar factor; a deviation
    above DUSCHINSKY_ORTHOGONALITY_TOL warns); with coordinates
    'curvilinear', J and K through the valence internals (`mode_sets`,
    `displacement_from_minima`).
    """
    sg, se, j, dev = mode_sets(ground, excited, coordinates)
    if coordinates == 'curvilinear':
        k = displacement_from_minima(sg, se, sg.mol, se.mol, 'curvilinear')
    else:
        k = _displacement(se, sg.masses, sg.mol, se.mol)
    return DuschinskyModel(sg.omega, se.omega, j, k, orthogonality=dev,
                           source=f'{sg.source} / {se.source}',
                           coordinates=coordinates)


def mode_sets(ground, excited, coordinates='rectilinear'):
    """(ground `HarmonicState`, excited `HarmonicState` turned onto the
    ground minimum's frame, J, deviation): the two mode sets of
    `duschinsky_model` and J. 'rectilinear': their overlap J = L_e^T L_g,
    replaced by its nearest orthogonal matrix (polar factor); `deviation` is
    max |s - 1| over J's singular values, and above
    DUSCHINSKY_ORTHOGONALITY_TOL it warns (`DuschinskyNotOrthogonal`).
    'curvilinear': J through the valence internals at each minimum
    (`curvilinear.curvilinear_duschinsky`), kept as it is; `deviation` the
    same measure, recorded."""
    if coordinates not in COORDINATES:
        raise ValueError(f'coordinates are {COORDINATES}, got '
                         f'{coordinates!r}')
    g, e = as_cartesian_hessian(ground), as_cartesian_hessian(excited)
    if [g.mol.atom_symbol(i) for i in range(g.mol.natm)] != \
            [e.mol.atom_symbol(i) for i in range(e.mol.natm)]:
        raise ValueError('the two Hessians are not of one molecule, atom by '
                         'atom')
    mol_e, rot = align_to(g.mol, e.mol, return_rotation=True)
    h_e = rotated_hessian(cartesian_hessian(e.hessian, e.mol.natm), rot)
    sg = harmonic_state(g)
    se = harmonic_state(CartesianHessian(mol_e, h_e, e.source))
    if len(sg.omega) != len(se.omega):
        raise ValueError(f'{len(sg.omega)} ground modes against '
                         f'{len(se.omega)} excited ones: one state is linear')
    if coordinates == 'curvilinear':
        j = curvilinear_duschinsky(sg.mol, sg.modes, se.mol, se.modes,
                                   sg.masses)
        s = np.linalg.svd(j, compute_uv=False)
        return sg, se, j, float(np.abs(s - 1.0).max())
    j = se.modes.T @ sg.modes
    u, s, vt = np.linalg.svd(j)
    dev = float(np.abs(s - 1.0).max())
    if dev > DUSCHINSKY_ORTHOGONALITY_TOL:
        warnings.warn(f'the Duschinsky matrix is orthogonal only to {dev:.3g} '
                      f'(singular values {s.min():.6f}..{s.max():.6f}); its '
                      f'nearest orthogonal matrix is used',
                      DuschinskyNotOrthogonal)
    return sg, se, u @ vt, dev


def _displacement(se, masses, mol_g, mol_e):
    """K = L_e^T M^{1/2} (x_g - x_e) of two geometries in one frame."""
    dx = (np.asarray(mol_g.atom_coords()) - np.asarray(mol_e.atom_coords()))
    return se.modes.T @ (dx.ravel() * np.repeat(masses, 3) ** 0.5)


def minima_in_frame(sg, mol_g, mol_e):
    """(ground minimum, its rotation, excited minimum, its rotation): two
    minima of any method, in any frame, turned into the frame of the ground
    mode set `sg` (`mode_sets`): the ground minimum superposed on `sg.mol`,
    the excited one on that (mass-weighted Kabsch, `align_to`). A rotation
    turns a per-atom vector of the minimum's own frame into this one
    (v @ rotation)."""
    g_in, rot_g = align_to(sg.mol, mol_g, return_rotation=True)
    e_in, rot_e = align_to(g_in, mol_e, return_rotation=True)
    return g_in, rot_g, e_in, rot_e


def displacement_from_minima(sg, se, mol_g, mol_e, coordinates):
    """K of `DuschinskyModel` from the two minima of any method (the
    displacement route): the minima are turned into the mode sets' frame
    (`minima_in_frame`) and their mass-weighted difference is projected on
    the excited modes. The mode sets may come from another method's
    Hessians at that method's own minima; only the displacement is the
    minima's.

    coordinates: 'cartesian', the straight line between the minima; or
    'curvilinear', the difference of the valence internals mapped linearly
    to Cartesians at the excited mode set's geometry
    (`curvilinear.curvilinear_displacement`), which puts a rotated group on
    its torsion rather than on the stretches.
    """
    g_in, _, e_in, _ = minima_in_frame(sg, mol_g, mol_e)
    if coordinates == 'cartesian':
        return _displacement(se, sg.masses, g_in, e_in)
    if coordinates == 'curvilinear':
        dq, _ = curvilinear_displacement(se.mol, e_in, g_in, sg.masses)
        return se.modes.T @ dq
    raise ValueError(f"coordinates are 'cartesian' or 'curvilinear', got "
                     f'{coordinates!r}')


def displacement_from_gradient(sg, se, j, mol_g, mol_e, gradient, state):
    """K of `DuschinskyModel` from one state's gradient at the other's
    minimum (the gradient route, linear in the displacement).

    state 'excited': `gradient` is the excited state's at the ground
        minimum `mol_g`, K = Omega_e^-2 L_e^T M^{-1/2} g.
    state 'ground': `gradient` is the ground state's at the excited minimum
        `mol_e`, q_g = Omega_g^-2 L_g^T M^{-1/2} g, K = -J q_g.
    gradient: (natm, 3) Ha/Bohr in its own geometry's frame; both minima
    are turned into the mode sets' frame first (`minima_in_frame`).
    """
    _, rot_g, _, rot_e = minima_in_frame(sg, mol_g, mol_e)
    w = np.repeat(sg.masses, 3) ** -0.5
    if state == 'excited':
        g = np.asarray(gradient, float) @ rot_g
        return (se.modes.T @ (g.ravel() * w)) / se.omega ** 2
    if state == 'ground':
        g = np.asarray(gradient, float) @ rot_e
        return -j @ ((sg.modes.T @ (g.ravel() * w)) / sg.omega ** 2)
    raise ValueError(f"state is 'excited' or 'ground', got {state!r}")


# ------------------------------------------------ the correlation function
def _beta_factors(omega_i, temperature):
    kt = thermal_energy(1.0, temperature)       # the temperature guard only
    with np.errstate(over='ignore', under='ignore'):
        boltz = np.exp(-omega_i / kt)           # e^{-beta omega}, 0 when cold
    return kt, boltz


class _Correlation:
    """ln G(t) of one oriented model at one temperature, branch-continued
    along an ascending time grid from t = 0; with a dipole (mu0, m), the
    Herzberg-Teller factor <mu(q) ... mu(q)> on top (`herzberg_teller`)."""

    def __init__(self, omega_i, omega_f, J, K, temperature, dipole=None,
                 left=None):
        self.wi, self.wf = np.asarray(omega_i, float), np.asarray(omega_f, float)
        self.J, self.K = np.asarray(J, float), np.asarray(K, float)
        # J^-T of a non-orthogonal J where it multiplies Omega_i; J itself
        # where J is orthogonal
        self.left = self.J if left is None else np.asarray(left, float)
        self.kt, self.boltz = _beta_factors(self.wi, temperature)
        self.p0 = self.left * self.wi[None, :] + self.wf[:, None] * self.J
        self.p1 = self.left * self.wi[None, :] - self.wf[:, None] * self.J
        self.log_norm = float(np.sum(np.log(2.0 * np.sqrt(self.wi * self.wf)))
                              + np.sum(np.log1p(-self.boltz)))
        self.dipole = dipole

    def point(self, t):
        """(ln det N1 N2 on some branch, d/dt of it, the single-valued part
        of ln G) at one time t."""
        wi, wf, J, K = self.wi, self.wf, self.J, self.K
        e2 = np.exp(-1j * wf * t)[:, None]
        with np.errstate(under='ignore'):
            w = (np.exp(1j * wi * t) * self.boltz)[None, :]
        de2, dw = -1j * wf[:, None] * e2, 1j * wi[None, :] * w
        p0, p1 = self.p0, self.p1
        n1 = p0 + e2 * p1 - p1 * w - e2 * p0 * w
        n2 = p0 - e2 * p1 + p1 * w - e2 * p0 * w
        d1 = de2 * p1 - p1 * dw - de2 * p0 * w - e2 * p0 * dw
        d2 = -de2 * p1 + p1 * dw - de2 * p0 * w - e2 * p0 * dw
        log_det, rate = 0.0j, 0.0j
        for n, d in ((n1, d1), (n2, d2)):
            lu, piv = lu_factor(n, check_finite=False)
            swaps = int(np.count_nonzero(piv != np.arange(len(piv))))
            log_det += np.log(np.diag(lu).astype(complex)).sum() + 1j * np.pi * swaps
            rate += np.trace(lu_solve((lu, piv), d, check_finite=False))
        wv = w.ravel()
        ti = -1j * (1.0 - wv) / (1.0 + wv)
        r = (self.left * (wi * ti)[None, :]) @ self.left.T
        c, s = np.cos(0.5 * wf * t), np.sin(0.5 * wf * t)
        f = r * c[None, :] + np.diag(wf * s)
        frk = np.linalg.solve(f, r @ K)
        rest = (0.5j * float(np.sum(wi - wf)) * t
                - 1j * complex(K @ (wf * s * frk)))
        if self.dipole is not None:
            rest += self.dipole_log(f, frk, c, s, ti)
        return complex(log_det), complex(rate), rest

    def dipole_log(self, f, frk, c, s, ti):
        """ln <mu(x) mu(y)> over the Gaussian of the two kernels, for
        mu(q) = mu0 + m^T q in the initial coordinates:

            sum_a (mu0_a + m_a.x*)^2
            + (i/2) sum_a m_a^T [(P - Q)^-1 - (P + Q)^-1] m_a,

        x* = -J^T (K - C F^-1 R K) the common centre of x and y,
        (P - Q)^-1 = -J^T C F^-1 J and (P + Q)^-1 = J^T S H^-1 J with
        H = J Omega_i T_i^-1 J^T S + Omega_f C: both finite at every t.

        At a non-orthogonal J the two Gaussians of the sum and difference
        coordinates are (a + J^T b J)^-1 and (c + J^T d J)^-1, which are
        J^-1 (J^-T a J^-1 + b)^-1 J^-T and the like: the same expressions
        with J^-T (`left`) where J stands and J^-1 where J^T stands."""
        mu0, m = self.dipole
        left, wi, wf = self.left, self.wi, self.wf
        back = left.T                       # J^T, or J^-1 at a non-orthogonal J
        x_star = -back @ (self.K - c * frk)
        jm = left @ m                                # (N, 3) in final modes
        pq_minus = -back @ (c[:, None] * np.linalg.solve(f, jm))
        h = ((left * (wi / ti)[None, :]) @ left.T) * s[None, :] \
            + np.diag(wf * c)
        pq_plus = back @ (s[:, None] * np.linalg.solve(h, jm))
        mean = mu0 + m.T @ x_star
        value = (complex(np.sum(mean ** 2))
                 + 0.5j * complex(np.sum(m * (pq_minus - pq_plus))))
        with np.errstate(divide='ignore'):
            return np.log(value)

    def __call__(self, t):
        t = np.asarray(t, float)
        if t.ndim != 1 or t[0] < 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError('the time grid must ascend from t >= 0')
        out = np.empty(len(t), complex)
        here = self.point(0.0)
        # G(0) = 1: det N1 det N2 is real and positive there
        log_det = complex(here[0].real)
        ta, k0 = 0.0, 0
        if t[0] == 0.0:
            out[0] = self.log_norm - 0.5 * log_det + here[2]
            k0 = 1
        for k in range(k0, len(t)):
            log_det, here = self._advance(log_det, ta, here, t[k], 0)
            ta = t[k]
            out[k] = self.log_norm - 0.5 * log_det + here[2]
        return out

    def _advance(self, log_det, ta, pa, tb, depth):
        """ln det N1 N2 continued from ta to tb: of the increments 2 pi k
        apart, the one the trapezoid of d/dt ln det = tr(N^-1 dN/dt) points
        to; the step is halved while the rate turns, or the two disagree, by
        more than BRANCH_MAX_TURN."""
        pb = self.point(tb)
        h = tb - ta
        estimate = 0.5 * h * (pa[1] + pb[1])
        raw = pb[0] - pa[0]
        k = np.round((estimate - raw).imag / (2.0 * np.pi))
        inc = raw + 2j * np.pi * k
        if (abs((inc - estimate).imag) <= BRANCH_MAX_TURN
                and abs(0.5 * h * (pb[1] - pa[1]).imag) <= BRANCH_MAX_TURN):
            return log_det + inc, pb
        if depth >= BRANCH_MAX_HALVINGS:
            raise RuntimeError(f'the branch of sqrt(det) could not be '
                               f'continued from t = {ta:.6g} to {tb:.6g} a.u.')
        mid = 0.5 * (ta + tb)
        log_det, pm = self._advance(log_det, ta, pa, mid, depth + 1)
        return self._advance(log_det, mid, pm, tb, depth + 1)


def log_correlation(model, t, temperature, direction, dipole=None):
    """ln G(t) of `model` for absorption or emission at `temperature`, on an
    ascending grid t >= 0 (a.u.); the branch is continued from G(0) = 1.
    dipole: (mu0, m) in the initial state's coordinates
    (`herzberg_teller.TransitionDipole.in_initial`), G then the dipole
    correlation function, G(0) = <|mu|^2>."""
    return model.correlation(temperature, direction, dipole=dipole)(t)


def _richardson(log_g, h):
    """(kappa_1, kappa_2) from ln G at h and 2h, the t^3 terms cancelled."""
    lg = log_g(np.array([0.0, h, 2.0 * h]))
    lg = lg - lg[0]
    k1 = -(4.0 * lg[1].imag - lg[2].imag / 2.0) / (3.0 * h)
    k2 = -(16.0 * lg[1].real - lg[2].real) * 2.0 / (12.0 * h ** 2)
    return float(k1), float(k2)


def cumulants_of(log_g, omega_min):
    """(kappa_1, kappa_2), mean and variance of the energy x left in the
    nuclei, read off ln G at two short times (Richardson); ln G(0) must be
    0. The step starts at CUMULANT_STEP of the slowest period and is halved
    until the estimates at h and h/2 agree to DUSCHINSKY_CUMULANT_TOL (kappa_1
    in units of sqrt(kappa_2)); the estimate at the accepted h is returned,
    so a band that agrees at the first step keeps its bits. A wide band (a
    soft mode turned into a stiff one) needs steps far below the slowest
    period."""
    h = CUMULANT_STEP * 2.0 * np.pi / omega_min
    k1, k2 = _richardson(log_g, h)
    for _ in range(DUSCHINSKY_CUMULANT_MAX_HALVINGS):
        h1, h2 = _richardson(log_g, 0.5 * h)
        scale = np.sqrt(abs(h2))
        if (abs(h1 - k1) <= DUSCHINSKY_CUMULANT_TOL * scale
                and abs(h2 - k2) <= DUSCHINSKY_CUMULANT_TOL * abs(h2)):
            return k1, k2
        h, k1, k2 = 0.5 * h, h1, h2
    raise RuntimeError(f'the band cumulants did not settle in '
                       f'{DUSCHINSKY_CUMULANT_MAX_HALVINGS} halvings of the '
                       f'step (h = {h:.3g} a.u., kappa_1 {k1:.6g}, kappa_2 '
                       f'{k2:.6g})')


def duschinsky_density(model, delta_e, temperature, direction,
                       broadening=0.0, lorentzian=0.0, dipole=None,
                       dt_per_period=24, t_decay=9.0, max_points=1 << 20):
    """rho(dE) = (1/2 pi) int dt e^{-i dE t} G(t) e^{-sig^2 t^2/2 - gamma|t|}.

    The band-shape density of `fc_weighted_dos` (dE = -x, x the vibrational
    energy the transition leaves behind) for the Duschinsky model, on the
    real time axis. A band needs a stated width: broadening (Gaussian
    standard deviation) or lorentzian (half width) must be positive. With a
    dipole (mu0, m) (`herzberg_teller`), the density is the dipole-weighted
    one and its area is <|mu|^2> over the initial state's thermal ensemble.
    """
    if broadening < 0.0 or lorentzian < 0.0:
        raise ValueError('a broadening cannot be negative')
    if broadening == 0.0 and lorentzian == 0.0:
        raise ValueError('the Duschinsky band needs a stated broadening (a '
                         'Gaussian or a Lorentzian width): without one it is '
                         'a set of delta lines')
    wi, wf, _, _ = model.oriented(direction)
    corr = model.correlation(temperature, direction, dipole=dipole)
    gauss = 0.5 * float(broadening) ** 2
    area = 1.0
    if dipole is not None:
        area = float(np.exp(corr(np.array([0.0]))[0]).real)
        if not area > 0.0:
            raise ValueError('the transition dipole vanishes at every '
                             'geometry: no intensity, no band')

    def log_g(t):
        return corr(t) - np.log(area) - gauss * t ** 2

    k1, k2 = cumulants_of(log_g, float(min(wi.min(), wf.min())))
    de = np.atleast_1d(np.asarray(delta_e, float))
    rho = real_axis_density(de, log_g, k2, k1, float(lorentzian),
                             float(max(wi.max(), wf.max())), dt_per_period,
                             t_decay, max_points,
                             envelope_time(gauss, float(lorentzian)))
    rho = area * np.maximum(rho, 0.0)
    return rho if np.ndim(delta_e) else float(rho[0])


class DuschinskyOscillators:
    """`band_shape.VibronicModel` of a `DuschinskyModel`: Condon, or with
    `dipole` (a `herzberg_teller.TransitionDipole`) Herzberg-Teller."""

    def __init__(self, model, dipole=None):
        self.model = model
        self.dipole = dipole

    def initial_dipole(self, direction):
        """(mu0, m) in the initial state's coordinates, None for Condon."""
        return None if self.dipole is None else \
            self.dipole.in_initial(self.model, direction)

    def density(self, delta_e, temperature, direction, broadening=0.0,
                lorentzian=0.0):
        return duschinsky_density(self.model, delta_e, temperature, direction,
                                  broadening=broadening, lorentzian=lorentzian,
                                  dipole=self.initial_dipole(direction))

    def cumulants(self, temperature, direction):
        corr = self.model.correlation(temperature, direction,
                                      dipole=self.initial_dipole(direction))
        wmin = float(min(self.model.omega_g.min(), self.model.omega_e.min()))
        return cumulants_of(corr, wmin)


def mean_energy_transfer(model, temperature, direction):
    """<H_f - H_i> over the initial state's thermal ensemble, in closed form:
    the first moment of the band, an independent check of G.

    With p_f = J^-T p_i (J p_i at an orthogonal J) and <q_i q_i^T> =
    diag(coth(beta w/2) / 2w), <p_i p_i^T> = diag(w coth(beta w/2) / 2):
    Tr(J^-1 J^-T P_i)/2 + Tr(J^T W_f^2 J C_i)/2 + K^T W_f^2 K/2
    - sum w_i coth/2, the first term sum w_i coth/4 at an orthogonal J.
    """
    wi, wf, J, K = model.oriented(direction)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    coth = 1.0 / np.tanh(wi / (2.0 * kt))
    ci = coth / (2.0 * wi)
    jw = (J ** 2 * (wf ** 2)[:, None]).sum(0)        # diag of J^T W_f^2 J
    kinetic = 0.25 * (wi * coth).sum()
    if model.coordinates == 'curvilinear':
        inv = np.linalg.inv(J)
        kinetic = 0.25 * float(((inv ** 2).sum(1) * wi * coth).sum())
    return float(kinetic + 0.5 * (jw * ci).sum()
                 + 0.5 * (K ** 2 * wf ** 2).sum() - 0.5 * (wi * coth).sum())


# ------------------------------------------------------------------ the guard
def soft_stiff_weight(model, direction):
    """The largest weight a soft mode (below DUSCHINSKY_SOFT_MODE_CM) of the
    initial state puts on the final state's stiff modes (above
    DUSCHINSKY_STIFF_MODE_CM): over the stiff rows of its column of the
    oriented J, sum J_lk^2; 0 where either class is empty. A thermally
    excited soft mode carries that weight of its large amplitude into the
    stretches."""
    wi, wf, j, _ = model.oriented(direction)
    soft, stiff = (DUSCHINSKY_SOFT_MODE_CM / HARTREE_TO_CM,
                   DUSCHINSKY_STIFF_MODE_CM / HARTREE_TO_CM)
    j2 = np.asarray(j, float) ** 2                 # rows final, columns initial
    return float(j2[wf > stiff][:, wi < soft].sum(0).max(initial=0.0))


def sum_rule_lambda(model, direction):
    """K^T W_f^2 K / 2: the final state's harmonic energy at the initial
    minimum above its own, the reorganization the model carries."""
    _, wf, _, k = model.oriented(direction)
    return float(0.5 * np.sum((wf * k) ** 2))


def _ratio(value, reference):
    """value / reference, 1 where both vanish, inf against a reference at or
    below zero; None without a reference."""
    if reference is None:
        return None
    if reference > 0.0:
        return value / reference
    return 1.0 if value == 0.0 and reference == 0.0 else float('inf')


def duschinsky_guard(model, temperature, direction, lambda_energy,
                     classical=0.0):
    """Whether the harmonic Duschinsky model describes one crossing direction.

    classical: the reorganization of a classical block the model leaves out
    (`split_soft_modes`), Hartree, added to its sum-rule lambda and to its
    band's first moment.

    lambda_energy: the accepting state's energy at the initial minimum above
    its own minimum, Hartree (None: no energy to check against). Refused
    ('status' 'refused', the failed tests in 'refused_by') when
      - mixing: a soft mode of the initial state puts more than
        DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX of its J^2 weight on the final
        state's stiff modes (`soft_stiff_weight`);
      - sum_rule: the model's lambda (`sum_rule_lambda`) and lambda_energy
        differ by more than RATE_FC_LAMBDA_RATIO_MAX either way;
      - first_moment: the band's mean vibrational energy less the zero-point
        difference (`mean_energy_transfer`) and lambda_energy differ by more
        than DUSCHINSKY_FIRST_MOMENT_RATIO_MAX either way;
    every test is refused without lambda_energy but mixing. The record
    carries every number, eV.
    """
    weight = soft_stiff_weight(model, direction)
    lam = sum_rule_lambda(model, direction) + classical
    zpe = model.zero_point_shift() * (1.0 if direction == 'absorption'
                                      else -1.0)
    moment = mean_energy_transfer(model, temperature, direction) + classical
    r_lam = _ratio(lam, lambda_energy)
    r_mom = _ratio(moment - zpe, lambda_energy)

    def inside(r, bar):
        return r is not None and 1.0 / bar <= r <= bar

    refused = [name for name, ok in (
        ('mixing', weight <= DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX),
        ('sum_rule', inside(r_lam, RATE_FC_LAMBDA_RATIO_MAX)),
        ('first_moment', inside(r_mom, DUSCHINSKY_FIRST_MOMENT_RATIO_MAX)))
        if not ok]
    return {'status': 'refused' if refused else 'computed',
            'refused_by': refused,
            'soft_stiff_weight': weight,
            'soft_stiff_weight_max': DUSCHINSKY_SOFT_STIFF_WEIGHT_MAX,
            'lambda_classical_block_eV': classical * HARTREE_TO_EV,
            'lambda_sum_rule_eV': lam * HARTREE_TO_EV,
            'lambda_energy_eV': (None if lambda_energy is None
                                 else lambda_energy * HARTREE_TO_EV),
            'lambda_ratio': r_lam, 'lambda_ratio_max': RATE_FC_LAMBDA_RATIO_MAX,
            'first_moment_eV': moment * HARTREE_TO_EV,
            'zero_point_eV': zpe * HARTREE_TO_EV,
            'first_moment_ratio': r_mom,
            'first_moment_ratio_max': DUSCHINSKY_FIRST_MOMENT_RATIO_MAX}


def guard_message(record):
    """One line saying why `duschinsky_guard` refused, with its numbers."""
    def num(x):
        return 'none' if x is None else f'{x:.4g}'
    centre = record['first_moment_eV'] - record['zero_point_eV']
    lam_e = num(record['lambda_energy_eV'])
    parts = {'mixing': f'a soft mode puts {record["soft_stiff_weight"]:.3g} '
                       f'of its J^2 weight on stiff modes (bar '
                       f'{record["soft_stiff_weight_max"]:g})',
             'sum_rule': f'sum-rule lambda {record["lambda_sum_rule_eV"]:.4g} '
                         f'eV against {lam_e} eV from the energies',
             'first_moment': f'band first moment less the zero-point shift '
                             f'{centre:.4g} eV against lambda {lam_e} eV'}
    return '; '.join(parts[name] for name in record['refused_by'])


# -------------------------------------------------------------- soft modes
@dataclass(frozen=True)
class ClassicalSoftBlock:
    """The soft modes `split_soft_modes` takes out of a Duschinsky model.

    cutoff: Hartree. omega_ground: the 'ground' modes below it; omega_excited:
    the frequencies of the 'excited' Hessian on their image. lambda_ground,
    lambda_excited: each state's own reorganization along the block (the
    'excited' minimum on the ground soft modes on the ground surface, and the
    reverse), Hartree. coupling: |Q_s^T W_e^2 Q_c| / |Q_c^T W_e^2 Q_c|, the
    excited Hessian's coupling between the block and the rest, which the
    split drops. zero_point: ZPE_excited - ZPE_ground the split takes out of
    the correlation function's phase, the full model's less the stiff one's,
    so the two together keep the zero-point difference of the two Hessians.
    """

    cutoff: float
    omega_ground: np.ndarray
    omega_excited: np.ndarray
    lambda_ground: float
    lambda_excited: float
    coupling: float
    zero_point: float

    def lam(self, direction):
        """The block's classical reorganization out of the initial state of
        `direction`: the ground's own for absorption, the excited's for
        emission (the Marcus and Marcus-Levich-Jortner convention)."""
        if direction == 'absorption':
            return self.lambda_ground
        if direction == 'emission':
            return self.lambda_excited
        raise ValueError(f"direction is 'absorption' or 'emission', got "
                         f'{direction!r}')

    def zero_point_shift(self, direction):
        """ZPE_final - ZPE_initial the block carries, the sign of
        `direction` (`zero_point`)."""
        return self.zero_point if direction == 'absorption' \
            else -self.zero_point

    def record(self):
        return {'cutoff_cm': self.cutoff * HARTREE_TO_CM,
                'omega_ground_cm': (self.omega_ground * HARTREE_TO_CM).tolist(),
                'omega_excited_cm': (self.omega_excited
                                     * HARTREE_TO_CM).tolist(),
                'lambda_ground_eV': self.lambda_ground * HARTREE_TO_EV,
                'lambda_excited_eV': self.lambda_excited * HARTREE_TO_EV,
                'zero_point_shift_eV': (self.zero_point_shift('absorption')
                                        * HARTREE_TO_EV),
                'coupling': self.coupling}


def split_soft_modes(model, cutoff):
    """(stiff `DuschinskyModel`, `ClassicalSoftBlock`): the 'ground' modes
    below `cutoff` (Hartree) and their image in the 'excited' coordinates
    taken out as a classical block; (model, None), the model itself, where
    no mode is below the cutoff.

    The image is the span Q_s of J's soft columns; the excited Hessian on the
    rest Q_c is diagonalized again for the stiff excited modes E_c, and the
    stiff model is J_c = E_c^T J (stiff columns), K_c = E_c^T K, ground
    frequencies the stiff ones. At an orthogonal J, J_c is orthogonal.
    """
    soft = np.asarray(model.omega_g) < cutoff
    n_soft = int(soft.sum())
    if n_soft == 0:
        return model, None
    if n_soft == len(soft):
        raise ValueError(f'every mode is below {cutoff * HARTREE_TO_CM:g} '
                         f'cm^-1: no quantum modes are left')
    q, _ = np.linalg.qr(model.J[:, soft], mode='complete')
    q_s, q_c = q[:, :n_soft], q[:, n_soft:]
    w2 = model.omega_e ** 2
    h_c = (q_c * w2[:, None]).T @ q_c
    h_s = (q_s * w2[:, None]).T @ q_s
    w2_c, u = np.linalg.eigh(h_c)
    e_c = q_c @ u
    stiff = DuschinskyModel(
        model.omega_g[~soft], np.sqrt(w2_c), e_c.T @ model.J[:, ~soft],
        e_c.T @ model.K, orthogonality=model.orthogonality,
        source=model.source, coordinates=model.coordinates)
    k_g = model.oriented('emission')[3]       # the excited minimum, ground
    k_s = q_s.T @ model.K                     # the ground minimum, excited
    block = ClassicalSoftBlock(
        cutoff=float(cutoff), omega_ground=np.asarray(model.omega_g)[soft],
        omega_excited=np.sqrt(np.linalg.eigvalsh(h_s)),
        lambda_ground=float(0.5 * np.sum((model.omega_g[soft]
                                          * k_g[soft]) ** 2)),
        lambda_excited=float(0.5 * k_s @ h_s @ k_s),
        coupling=float(np.linalg.norm((q_s * w2[:, None]).T @ q_c)
                       / np.linalg.norm(h_c)),
        zero_point=model.zero_point_shift() - stiff.zero_point_shift())
    return stiff, block
