"""What the rate layer reads of a geometry, whatever method computed it.

`StateMinimum` is one geometry's record: a relaxed excited minimum, or the
vertical point R0. GW-BSE (`from_gw_bse_record`, the stages' own records) and
ADC(2) (`from_adc2`, `from_adc2_spectrum`) fill the same fields, so `rate_models` takes either and
a table says which method each geometry came from. A field a method does not
provide is None, and the rate model refuses the channel that needs it by name
rather than reading a zero.

`GroundModes` is the S0 Hessian's normal modes at R0, the bath every
Huang-Rhys factor and coupling is expanded in.
"""
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from src.properties.transition_moments import spectrum_properties

#: The method label of a GW-BSE minimum (a record without one is GW-BSE).
GW_BSE = 'gw-bse'
#: The method labels of an ADC(2) and an SCS-ADC(2) minimum.
ADC2_METHODS = ('adc2', 'scs-adc2')
#: Why an SCS-ADC(2) minimum carries no spin-orbit coupling.
SCS_SOC_REFUSAL = ('SCS is not well-defined for spin-orbit coupling: spin-component scaling breaks '
                   'spin-rotation symmetry and splits the sublevels of a triplet (Ms = +-1 against Ms = 0, '
                   'SingleReference.ADC.eeADC.ee_ri_flip), so SCS-ADC(2) gives energies and '
                   'geometries only; take the spin-orbit couplings from ADC(2)')


@dataclass(frozen=True)
class GroundModes:
    """The S0 normal modes at R0: frequencies (Hartree; imaginary ones
    negative), mass-weighted modes, masses, and the Hessian's route."""
    omega: np.ndarray
    modes: np.ndarray
    masses: np.ndarray
    route: str

    @classmethod
    def from_record(cls, record):
        """The modes of a hess stage record."""
        return cls(np.asarray(record['omega'], float),
                   np.asarray(record['modes'], float),
                   np.asarray(record['masses'], float), str(record['route']))


@dataclass(frozen=True)
class StateMinimum:
    """One geometry's ingredients of the rate table, method-neutral.

    method: who computed it ('gw-bse', 'adc2', ...); where: 'R0', 'S1' or
        'T1' (the vertical point, or the minimum of that state).
    coords_bohr: (natm, 3) the geometry.
    e0: E_0 at this geometry, Hartree (the R0 one is the table's zero).
    energy: label -> total energy E_0 + Omega in Hartree here, at least the
        emitter and the reservoir, and T(n+1) above the reservoir for the T2
        channel. omega: label -> Omega (Hartree), the same labels.
    emitter, reservoir: the singlet and triplet labels this geometry's own
        evaluation reported ('S1', 'T2').
    socme: (n_s + 1, n_t) |<S_I|H_SO|T_J>| here, rows S0, S1, ..., or None.
    soc_dv_cart: (natm, 3) the Cartesian derivative of the crossing's
        element <emitter|H_SO|reservoir> here, or None (Condon only);
        soc_dv_paths: name -> such a derivative per contribution.
    dipole: (n, 3) the singlets' transition dipoles here, f_osc their
        oscillator strengths (row 0 = S1), or None.
    d_ground: (natm, 3) the derivative coupling <S0|d/dR|emitter> here, or
        None (no internal conversion).
    solvation: 'gas', 'equilibrium' or 'nonequilibrium'.
    provenance: anything the method wants recorded with it (its route,
        its SOC approximation).
    """
    method: str
    where: str
    coords_bohr: np.ndarray
    e0: float
    energy: dict
    omega: dict
    emitter: str
    reservoir: str
    socme: Optional[np.ndarray]
    soc_dv_cart: Optional[np.ndarray]
    soc_dv_paths: Optional[dict]
    dipole: Optional[np.ndarray]
    f_osc: Optional[np.ndarray]
    d_ground: Optional[np.ndarray]
    solvation: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('method', 'where', 'emitter', 'reservoir', 'solvation'):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f'StateMinimum.{name} is a label')
        if self.where not in ('R0', 'S1', 'T1'):
            raise ValueError(f'where={self.where!r}: R0, S1 or T1')
        for name in (self.emitter, self.reservoir):
            if name not in self.energy:
                raise ValueError(f'{self.method} at {self.where}: no energy '
                                 f'of {name}, which it names')

    def to_record(self):
        """The minimum as a record: {field: value}."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def from_gw_bse_record(record, where):
    """The `StateMinimum` of a stage record (vert, s1 or t1) of
    `stages`."""
    props = record['props']
    return StateMinimum(
        method=record.get('method', GW_BSE), where=where,
        coords_bohr=np.asarray(props['coords_bohr'], float),
        e0=float(props['e0']), energy=dict(props['energy']),
        omega=dict(props['omega']), emitter=props['emitter'],
        reservoir=props['reservoir'], socme=np.asarray(props['socme']),
        soc_dv_cart=np.asarray(props['soc_dv_cart']),
        soc_dv_paths=dict(props['soc_dv_paths']),
        dipole=np.asarray(props['dipole']),
        f_osc=np.asarray(props['f_osc']),
        d_ground=np.asarray(props['d_ground']),
        solvation=record.get('solvation') or 'gas',
        provenance={'kernel': record.get('kernel')})


def from_adc2(where, coords_bohr, e0, omega_singlet, omega_triplet,
              socme=None, dipole=None, f_osc=None, d_ground=None,
              soc_dv_cart=None, solvation='gas', emitter='S1',
              reservoir='T1', provenance=None, method='adc2'):
    """The `StateMinimum` of an ADC(2) evaluation at one geometry: E_0 (the
    MP2 ground state there, or its solvated G), and the excitation energies
    of the singlets and triplets (Hartree, lowest first), which make every
    total energy E_0 + Omega. The spin-orbit table, the transition dipoles,
    the oscillator strengths and the derivative couplings are what the
    method's own evaluation provides; each left None removes the channels
    that need it (`rate_models.refused_channels`). `provenance` names the
    route of each (an SOC table of the ADC(2) singles block, a dipole of the
    unrelaxed transition density); `method` is 'adc2' or 'scs-adc2', and an
    SCS-ADC(2) minimum takes no spin-orbit table or derivative
    (`SCS_SOC_REFUSAL`)."""
    if method not in ADC2_METHODS:
        raise ValueError(f'method={method!r}: one of {ADC2_METHODS}')
    if method == 'scs-adc2' and (socme is not None or soc_dv_cart is not None):
        raise ValueError(f'an scs-adc2 minimum with spin-orbit couplings: {SCS_SOC_REFUSAL}')
    energy, omega = {}, {}
    for spin, values in (('S', omega_singlet), ('T', omega_triplet)):
        for i, w in enumerate(np.asarray(values, float)):
            omega[f'{spin}{i + 1}'] = float(w)
            energy[f'{spin}{i + 1}'] = float(e0) + float(w)
    return StateMinimum(
        method=method, where=where,
        coords_bohr=np.asarray(coords_bohr, float), e0=float(e0),
        energy=energy, omega=omega, emitter=emitter, reservoir=reservoir,
        socme=None if socme is None else np.asarray(socme, float),
        soc_dv_cart=None if soc_dv_cart is None else np.asarray(soc_dv_cart),
        soc_dv_paths=None, dipole=None if dipole is None else np.asarray(
            dipole, float),
        f_osc=None if f_osc is None else np.asarray(f_osc, float),
        d_ground=None if d_ground is None else np.asarray(d_ground, float),
        solvation=solvation, provenance=dict(provenance or {}))


def from_adc2_spectrum(spectrum, where, n_states, solvation='gas', emitter='S1',
                       reservoir='T1', *, numerics):
    """The `StateMinimum` of an ADC(2) spectrum at one geometry
    (`gradients.adc.ee_spectrum.ADC2Spectrum`): E_0, the n_states lowest
    roots of each spin, and the method's own spin-orbit table, transition
    dipoles and oscillator strengths from its ISR transition densities
    (`properties.transition_moments.spectrum_properties`, the spin-orbit
    operator `numerics` declares, named in the provenance's soc_term). The triplet sublevels follow from the Ms = 0
    element by the Wigner-Eckart theorem, exact for ADC(2). An SCS-ADC(2)
    spectrum is refused (`SCS_SOC_REFUSAL`); its minimum comes from
    `from_adc2` without a table."""
    if spectrum.method != 'adc2':
        raise ValueError(f'spectrum method {spectrum.method!r}: {SCS_SOC_REFUSAL}')
    props = spectrum_properties(spectrum, n_states, numerics=numerics)
    return from_adc2(where, spectrum.mol.atom_coords(), spectrum.e0,
                     spectrum.omega['singlet'][:n_states],
                     spectrum.omega['triplet'][:n_states], socme=props['socme'],
                     dipole=props['dipole'], f_osc=props['f_osc'],
                     solvation=solvation, emitter=emitter, reservoir=reservoir,
                     provenance={'socme': 'ISR transition densities',
                                 'soc_term': props['soc_term'],
                                 'soc_sublevels': 'Ms = 0 element, Wigner-Eckart (exact)',
                                 'dipole': 'ISR ground-state transition density',
                                 'flags': props['flags']})
