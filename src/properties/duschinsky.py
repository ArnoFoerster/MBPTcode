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

Energies in Hartree, time in atomic units, temperature in Kelvin.
"""
import re
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import BOLTZMANN_HARTREE_PER_KELVIN
from src.properties.franck_condon import (envelope_time, real_axis_density,
                                          require_vibrational_modes,
                                          thermal_energy)
from src.properties.curvilinear import curvilinear_displacement
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
#: Step, in units of the slowest period, at which the first two cumulants of a
#: band are read off ln G (they set the grid of `vibronic_band_grid` and the
#: Lorentzian kink correction of the real-axis integral).
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

    omega_g, omega_e: (N,) Hartree. J: (N, N), orthogonal. K: (N,), the
    ground minimum in the excited state's coordinates, Bohr sqrt(m_e).
    `orthogonality`: max |s - 1| over J's singular values before it was made
    orthogonal (0 for a model built from given J).
    """

    omega_g: np.ndarray
    omega_e: np.ndarray
    J: np.ndarray
    K: np.ndarray
    orthogonality: float = 0.0
    source: str = ''

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
            return self.omega_e, self.omega_g, self.J.T, -self.J.T @ self.K
        raise ValueError(f"direction is 'absorption' or 'emission', got "
                         f'{direction!r}')

    def huang_rhys(self):
        """S_k = omega_e,k K_k^2 / 2 on the excited modes: the displacement
        part alone, for comparison with the displaced-oscillator routes."""
        return self.omega_e * self.K ** 2 / 2.0

    def zero_point_shift(self):
        """ZPE_e - ZPE_g: where the 0-0 line sits above the adiabatic gap."""
        return 0.5 * float(self.omega_e.sum() - self.omega_g.sum())


def duschinsky_model(ground, excited):
    """The Duschinsky model between two Hessians at their own minima.

    ground, excited: `CartesianHessian`s (or anything with `.mol` and
        `.hessian`). The excited minimum and its Hessian are turned onto the
        ground minimum's frame first (mass-weighted Kabsch, `align_to`), so a
        Hessian computed in another orientation -- another program's standard
        frame -- gives the same model.

    J = L_e^T L_g and K = L_e^T M^{1/2} (x_g - x_e) on the mass-weighted modes.
    J is replaced by its nearest orthogonal matrix (polar factor); a
    deviation above DUSCHINSKY_ORTHOGONALITY_TOL warns.
    """
    sg, se, j, dev = mode_sets(ground, excited)
    k = _displacement(se, sg.masses, sg.mol, se.mol)
    return DuschinskyModel(sg.omega, se.omega, j, k, orthogonality=dev,
                           source=f'{sg.source} / {se.source}')


def mode_sets(ground, excited):
    """(ground `HarmonicState`, excited `HarmonicState` turned onto the
    ground minimum's frame, J, deviation): the two mode sets of
    `duschinsky_model` and their overlap J = L_e^T L_g, replaced by its
    nearest orthogonal matrix (polar factor); `deviation` is max |s - 1|
    over J's singular values, and above DUSCHINSKY_ORTHOGONALITY_TOL it
    warns (`DuschinskyNotOrthogonal`)."""
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

    def __init__(self, omega_i, omega_f, J, K, temperature, dipole=None):
        self.wi, self.wf = np.asarray(omega_i, float), np.asarray(omega_f, float)
        self.J, self.K = np.asarray(J, float), np.asarray(K, float)
        self.kt, self.boltz = _beta_factors(self.wi, temperature)
        self.p0 = self.J * self.wi[None, :] + self.wf[:, None] * self.J
        self.p1 = self.J * self.wi[None, :] - self.wf[:, None] * self.J
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
        r = (J * (wi * ti)[None, :]) @ J.T
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
        H = J Omega_i T_i^-1 J^T S + Omega_f C: both finite at every t."""
        mu0, m = self.dipole
        J, wi, wf = self.J, self.wi, self.wf
        x_star = -J.T @ (self.K - c * frk)
        jm = J @ m                                   # (N, 3) in final modes
        pq_minus = -J.T @ (c[:, None] * np.linalg.solve(f, jm))
        h = ((J * (wi / ti)[None, :]) @ J.T) * s[None, :] + np.diag(wf * c)
        pq_plus = J.T @ (s[:, None] * np.linalg.solve(h, jm))
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
    return _Correlation(*model.oriented(direction), temperature,
                        dipole=dipole)(t)


def cumulants_of(log_g, omega_min):
    """(kappa_1, kappa_2), mean and variance of the energy x left in the
    nuclei, read off ln G at two short times (Richardson); ln G(0) must be
    0."""
    h = CUMULANT_STEP * 2.0 * np.pi / omega_min
    lg = log_g(np.array([0.0, h, 2.0 * h]))
    lg = lg - lg[0]
    k1 = -(4.0 * lg[1].imag - lg[2].imag / 2.0) / (3.0 * h)
    k2 = -(16.0 * lg[1].real - lg[2].real) * 2.0 / (12.0 * h ** 2)
    return float(k1), float(k2)


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
    corr = _Correlation(*model.oriented(direction), temperature,
                        dipole=dipole)
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
        corr = _Correlation(*self.model.oriented(direction), temperature,
                            dipole=self.initial_dipole(direction))
        wmin = float(min(self.model.omega_g.min(), self.model.omega_e.min()))
        return cumulants_of(corr, wmin)


def mean_energy_transfer(model, temperature, direction):
    """<H_f - H_i> over the initial state's thermal ensemble, in closed form:
    the first moment of the band, an independent check of G.

    With p_f = J p_i and <q_i q_i^T> = diag(coth(beta w/2) / 2w):
    sum w_i coth/4 + Tr(J^T W_f^2 J C_i)/2 + K^T W_f^2 K/2 - sum w_i coth/2.
    """
    wi, wf, J, K = model.oriented(direction)
    kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
    coth = 1.0 / np.tanh(wi / (2.0 * kt))
    ci = coth / (2.0 * wi)
    jw = (J ** 2 * (wf ** 2)[:, None]).sum(0)        # diag of J^T W_f^2 J
    return float(0.25 * (wi * coth).sum() + 0.5 * (jw * ci).sum()
                 + 0.5 * (K ** 2 * wf ** 2).sum() - 0.5 * (wi * coth).sum())
