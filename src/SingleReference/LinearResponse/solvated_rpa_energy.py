"""The solute's share of the coupled solute-solvent dRPA correlation energy.

Fold the ACFDT log-determinant exactly over a solvent whose orbitals do not
overlap the solute's, so that chi0 = diag(P_1, P_2) is block diagonal while v
is not (Duchemin, Jacquemin and Blase, J. Chem. Phys. 144, 164106 (2016),
Eqs. 11-13). The solute's share is

    E_c^solute = (1/2pi) int dw [ Tr ln(1 - vtilde(iw) P_1(iw)) + Tr(v P_1(iw)) ]

-- the DRESSED interaction in the logarithm and the BARE one in the linear
counter-term, because a block-diagonal chi0 never lets v_12 into a trace.
Dressing the counter-term as well cancels the leading solute-solvent dispersion
term, which is the correlation-energy analogue of what the bare-exchange
reference does to the reaction field in a self-energy; dressing neither is the
PTE-style answer every correlated-PCM method in production computes.

The solvent's response has its own frequency dependence, and a single-pole
model makes the reaction field at finite frequency the static one times a
scalar (Duchemin, Amblard and Blase, J. Chem. Theory Comput. 20, 9072 (2024)):

    vtilde(iw) = g(iw) vtilde ,   g(iw) = Omega_p^2 / (w^2 + Omega_p^2) .

So three members of ONE functional, evaluated on ONE set of factors and ONE
frequency grid:

    A   g = 0, bare counter-term        the bare interaction in both places
    B   g = 1, dressed counter-term     the default of the dRPA surfaces
    F   g(iw), bare counter-term        the fold with the solvent's own dynamics

and the split of F - A into its first order in vtilde,

    E_disp = -(1/2pi) int dw g(w) Tr[vtilde P_1(iw)] ,

the Casimir-Polder solute-solvent dispersion at the RPA level -- negative,
because both vtilde and P_1 are negative semi-definite -- and the rest, which
is the screening reduction of the solute's own correlation. At constant g the
dispersion integral closes: int_0^inf dw Delta/(Delta^2 + w^2) = pi/2 turns it
into sum_ia <ia|vtilde|ia> over spatial occupied-virtual pairs, equivalently
half the sum over spin-orbital pairs -- the Born-like reaction field of every
transition density, which a constant dielectric grants to arbitrarily fast
fluctuations and which is why a frequency-independent vtilde cannot carry this
term.

WHAT MAKES IT WRONG IF MISUSED. An absolute total energy in a continuum under
B is missing the dispersion term -- about 29.6 kcal/mol for water in water,
and enough to leave water DESTABILIZED by toluene -- while excitation
energies, quasiparticle levels and any difference taken at a FIXED geometry
are free of it, as is every gas-phase total energy. The fold restores it on
the cavity the caller gives, and the term belongs on the solvent-accessible
surface: on the electrostatic cavity it is several times too large. The
rescaled logarithm and the counter-term are ONE choice, which is why
`fold_terms` returns them together; picking up one half leaves a first-order
term of order Tr(vtilde P_1) uncancelled, tens of kcal/mol.
"""
from dataclasses import dataclass

import numpy as np

from src.Base.environment import dresses_interaction
from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import dynamic_screening_factor


@dataclass(frozen=True)
class SolvatedCorrelation:
    """The three correlation energies (Hartree) and the split of F - A.

    bare, dressed, fold: A, B and F(Omega_p) above. dispersion: the first order
    in vtilde of F, negative. beyond_first_order: F - bare - dispersion, the
    screening reduction of the solute's own correlation. omega_p in Hartree.
    traces: Tr[vtilde P_1(iw)] per frequency, independent of Omega_p, so
    `solute_solvent_dispersion_energy` re-evaluates the dispersion term at any
    pole energy without touching the polarizability again.
    """

    bare: float
    dressed: float
    fold: float
    dispersion: float
    beyond_first_order: float
    omega_p: float
    traces: np.ndarray
    omega: np.ndarray
    weights: np.ndarray


def screening_matrix(auxmol, environment, V=None):
    """N = V_d^(-1/2) vtilde V_d^(-1/2), the reaction field in the dressed
    auxiliary gauge, or None when nothing screens.

    One matrix carries both halves of the functional: I - (1 - g) N rescales
    the interaction inside the logarithm to v + g vtilde, and I - N is the BARE
    interaction in the same gauge, V_d^(-1/2) v V_d^(-1/2), hence the
    counter-term matrix. N is negative semi-definite because a reaction field
    screens, and its eigenvalues are bounded below by -1 for the same reason
    V_d = V + vtilde stays positive.
    """
    kernel = environment.aux_kernel(auxmol)
    if kernel is None:
        return None
    V = auxmol.intor('int2c2e', aosym='s1') if V is None else V
    root = aux_metric_sqrt(auxmol, environment, V=V)
    n_mat = np.linalg.solve(root, np.linalg.solve(root, kernel).T).T
    return 0.5 * (n_mat + n_mat.T)


def fold_terms(environment, auxmol, omega_points):
    """(screening, counter_term) of the fold at these frequencies, or (None, None).

    ONE function, because the two arguments are ONE choice: the linear term of
    log det(I - S c) is -Tr(S c), so a counter-term that does not match the
    logarithm's own interaction leaves a first-order term of order
    Tr(vtilde P_1) uncancelled -- tens of kcal/mol. Every chain that folds asks
    here rather than assembling the pair itself, so no route can pick up one
    half of it.

    (None, None) whenever nothing is dressed, which is the gas phase and the
    fixed charges: there is no fold to take and the caller's own default
    interaction is already the answer.
    """
    if not dresses_interaction(environment, auxmol):
        return None, None
    n_mat = screening_matrix(auxmol, environment)
    if n_mat is None:
        return None, None
    g = np.asarray(environment.dynamic_factor(np.asarray(omega_points, float)),
                   float)
    return (n_mat, g), np.eye(n_mat.shape[0]) - n_mat


def solute_solvent_dispersion_energy(traces, grid, omega_p):
    """-(1/2pi) sum_w W_w g(w; Omega_p) Tr[vtilde P_1(iw)] in Hartree.

    The first-order-in-vtilde term of the fold: the solute-solvent dispersion
    at the RPA level, with the solvent's spectrum collapsed to one pole at
    Omega_p (Hartree). Negative, and vanishing as Omega_p -> 0 because a
    solvent with no oscillator strength cannot disperse. It is a property of
    the coupled response and not of the solute's own density, so it is a
    different quantity from the empirical D3/D4 `Base.dispersion` correction,
    which no continuum enters.
    """
    g = dynamic_screening_factor(grid.omega_points, omega_p)
    return -float(grid.omega_weights @ (g * np.asarray(traces, float))) / (
        2.0 * np.pi)
