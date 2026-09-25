"""What a potential-energy surface computes, declared apart from how it is realized.

A surface is a functional E_0(R) plus, optionally, a state that sits on it: a
neutral excitation (BSE/TDA on top of GW or a mean field) or a charged one
(remove or add an electron at a quasiparticle energy). Two surfaces can only
be differenced -- a total energy subtracted, a gradient compared -- if they
share the same ground-state functional and the same environment; a differing
excitation is fine, because that difference is what a gap or an IP/EA *is*.
This module is pure vocabulary: dataclasses and their validation, no numerics,
no pyscf, no `src` imports, so that both the energy and the gradient code can
depend on it without depending on each other.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

_GROUND_STATE_KINDS = ('rpa', 'dft')
_SPINS = ('singlet', 'triplet')
_KERNELS = ('bse', 'bse-tda')
_QP_METHODS = ('g0w0', 'evgw')
_SCREENINGS = ('rpa',)
_CHARGE_CHANGES = (-1, 1)
_QP_STATES_KINDS = ('admitted', 'frontier', 'valence', 'all')
_QP_STATES_THRESHOLDS = ('gap', 'omega1')


@dataclass(frozen=True)
class GroundState:
    """Which functional E_0(R) is.

    kind='rpa'   E_0 = E_ref + (E_x^HF[rho_ref] - E_xc[rho_ref]) + E_c^dRPA.
                 The middle term is the exact-exchange double counting that
                 turns E_KS into E_HF at the same density (Toelle, Kitsaras,
                 Loos 2025 Eq. 15 wants E_HF). It is identically zero on a
                 Hartree-Fock reference and is ALWAYS carried and ALWAYS
                 printed; there is no keyword to drop it, because a surface
                 without it is a different functional and belongs to kind='dft'.
    kind='dft'   E_0 = E_KS[xc], the mean field's own energy; xc='hf' is HF.
    """
    kind: str
    xc: str

    def __post_init__(self) -> None:
        if self.kind not in _GROUND_STATE_KINDS:
            raise ValueError(
                f"GroundState.kind={self.kind!r} not in {_GROUND_STATE_KINDS}")
        object.__setattr__(self, 'xc', self.xc.lower())

    def terms(self) -> Tuple[str, ...]:
        """Which additive pieces this functional's energy is reported as."""
        if self.kind == 'dft':
            return ('E_ref',)
        return ('E_ref', 'E_x^HF - E_xc', 'E_c^dRPA')

    def label(self) -> str:
        """The printable name of E_0, e.g. 'E_PBE0 + (E_x^HF - E_xc)[rho] + E_c^dRPA'."""
        functional = f'E_{self.xc.upper()}'
        if self.kind == 'dft':
            return functional
        return f'{functional} + (E_x^HF - E_xc)[rho] + E_c^dRPA'


@dataclass(frozen=True)
class Excitation:
    """Which neutral state sits on E_0."""
    spin: str
    root: int = 1
    irrep: Optional[str] = None
    kernel: str = 'bse'
    qp: str = 'g0w0'
    screening: str = 'rpa'

    def __post_init__(self) -> None:
        if self.spin not in _SPINS:
            raise ValueError(f"Excitation.spin={self.spin!r} not in {_SPINS}")
        if self.root < 1:
            raise ValueError(f"Excitation.root={self.root!r} must be >= 1")
        if self.kernel not in _KERNELS:
            raise ValueError(f"Excitation.kernel={self.kernel!r} not in {_KERNELS}")
        if self.qp not in _QP_METHODS:
            raise ValueError(f"Excitation.qp={self.qp!r} not in {_QP_METHODS}")
        if self.screening not in _SCREENINGS:
            raise ValueError(
                f"Excitation.screening={self.screening!r} not in {_SCREENINGS}")


@dataclass(frozen=True)
class ChargedExcitation:
    """E^{N-1} or E^{N+1}: E_0 -/+ eps^QP_p."""
    orbital: int
    charge_change: int

    def __post_init__(self) -> None:
        if self.orbital < 0:
            raise ValueError(f"ChargedExcitation.orbital={self.orbital!r} must be >= 0")
        if self.charge_change not in _CHARGE_CHANGES:
            raise ValueError(
                f"ChargedExcitation.charge_change={self.charge_change!r} "
                f"not in {_CHARGE_CHANGES}")


@dataclass(frozen=True)
class QPStates:
    """Which orbitals carry an explicitly solved quasiparticle energy on the
    BSE diagonal; every other orbital carries the frozen scissor shift.

    kind='admitted'  every orbital whose quasiparticle root satisfies the pole
                     condition |omega_p - eps_q| < Omega_1 for every pole q the
                     contour sweeps, Omega_1 the lowest neutral dRPA excitation;
                     threshold='gap' tests against the particle-hole gap E_g,
                     a lower bound on Omega_1 that needs no solve;
                     threshold='omega1' tests against the lowest dRPA root.
    kind='frontier'  [nocc - half_width, nocc + half_width), widened over
                     degenerate blocks.
    kind='valence'   the valence occupied orbitals plus extra_virtuals
                     virtuals, optionally filtered on pole strength Z > 0.5.
    kind='all'       every orbital.
    """
    kind: str = 'admitted'
    threshold: str = 'gap'
    half_width: int = 2
    extra_virtuals: int = 10
    filter_z: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _QP_STATES_KINDS:
            raise ValueError(f"QPStates.kind={self.kind!r} not in {_QP_STATES_KINDS}")
        if self.threshold not in _QP_STATES_THRESHOLDS:
            raise ValueError(
                f"QPStates.threshold={self.threshold!r} not in {_QP_STATES_THRESHOLDS}")
        if self.half_width < 1:
            raise ValueError(f"QPStates.half_width={self.half_width!r} must be >= 1")
        if self.extra_virtuals < 0:
            raise ValueError(
                f"QPStates.extra_virtuals={self.extra_virtuals!r} must be >= 0")


@dataclass(frozen=True)
class SurfacePhysics:
    """What a surface computes: the functional, the state on it, the environment."""
    ground_state: GroundState
    excitation: Optional[Union[Excitation, ChargedExcitation]]
    environment: str = 'gas'

    def label(self) -> str:
        """The printable name of the total energy this surface reports."""
        base = self.ground_state.label()
        if self.excitation is None:
            label = base
        elif isinstance(self.excitation, ChargedExcitation):
            sign = '-' if self.excitation.charge_change == -1 else '+'
            label = f'{base} {sign} eps^QP_p'
        else:
            label = f'{base} + Omega'
            if self.ground_state.kind == 'dft':
                label += ' (mean-field ground state)'
        if self.environment != 'gas':
            label += f' in {self.environment}'
        return label

    def comparable_with(self, other: 'SurfacePhysics') -> bool:
        """True iff two surfaces share a ground state and environment, so their energies may be differenced."""
        return (self.ground_state == other.ground_state
                and self.environment == other.environment)


class PhysicsMismatch(ValueError):
    """Two surfaces whose declared physics cannot be differenced."""

    def __init__(self, a: SurfacePhysics, b: SurfacePhysics) -> None:
        self.a = a
        self.b = b
        super().__init__(
            f"cannot difference {a.label()!r} against {b.label()!r}: "
            "ground state or environment disagree")
