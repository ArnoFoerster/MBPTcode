"""Rate models: a rate table from the S0 modes and three `StateMinimum`
records (R0, the S1 minimum, the T1 minimum), on `properties.rates`.

`tadf_rate_table` is the TADF table of the staged run: every Huang-Rhys
factor (measured in curvilinear coordinates, `properties.curvilinear`),
derivative coupling and spin-orbit derivative expanded in the S0
modes, each minimum superposed on R0 first; the rules of the table (which
minimum each crossing reads, which lambda, the triplet sublevel average) are
`rates.rate_table`'s. The three geometries must come from ONE method: the
table differences their energies.

A method that leaves an ingredient out leaves its channels out, by name
(`refused_channels`): no spin-orbit table, no crossing at all; no
spin-orbit derivative, Condon only; no transition dipole, no k_r and no
kinetics; no derivative coupling, no internal conversion.

The Samanta rows of the UISC protocol (`uisc`) are a second rate model on
the same records.
"""
from dataclasses import replace

import numpy as np

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.geometry.geometry import molecule_at
from src.photophysics.records import key_of, log
from src.photophysics import rates
from src.properties import franck_condon
from src.properties.curvilinear import huang_rhys_curvilinear
from src.properties.spin_vibronic import mode_derivatives
from src.properties.vibronic import align_to, project_coupling


class RateRefused(ValueError):
    """Records the rate layer cannot read together."""


def refused_channels(minimum, supplied=None):
    """{ingredient: what it removes} for the fields `minimum` leaves None.
    `supplied`: where another model supplies the crossing's spin-orbit
    derivative, its `rates.Minimum` fields, or why it has none (a string)."""
    out = {}
    if isinstance(supplied, str):
        out['soc_dv_cart'] = supplied
    elif supplied is None and minimum.soc_dv_cart is None:
        out['soc_dv_cart'] = 'Condon only (no Herzberg-Teller term)'
    if minimum.where == 'S1':
        if minimum.dipole is None or minimum.f_osc is None:
            out['dipole'] = 'no k_r and no photoluminescence kinetics'
        if minimum.d_ground is None:
            out['d_ground'] = 'no internal conversion S1 -> S0'
    return out


def rate_minimum(minimum, modes, rotation, emits=False):
    """The `rates.Minimum` of one minimum: its labels, energies and
    spin-orbit table, the derivative of its crossing's element on the S0
    modes (`mode_derivatives`, rotated into R0's frame first) where it has
    one; with `emits` the emitter's energy, dipole and oscillator strength
    there where it has them. Its ground-state energy is the minimum's E_0,
    which guards the decay to S0 from there (`rates.fc_lambda_guard`)."""
    if minimum.socme is None:
        raise RateRefused(f'{minimum.method} at {minimum.where} carries no '
                          f'spin-orbit table: no crossing is defined')
    franck_condon.require_vibrational_modes(modes.omega, 'rate_minimum')

    def per_mode(dv):
        return mode_derivatives(np.asarray(dv), modes.modes, modes.masses,
                                modes.omega, rotation=rotation)[1]

    derivative = {}
    if minimum.soc_dv_cart is not None:
        derivative = dict(
            dv_dq=per_mode(minimum.soc_dv_cart),
            omega_promoting=np.asarray(modes.omega),
            dv_dq_paths={path: per_mode(dv) for path, dv in
                         (minimum.soc_dv_paths or {}).items()})
    emission = {}
    if emits and minimum.dipole is not None and minimum.f_osc is not None:
        root = key_of(minimum.emitter)[1]
        emission = dict(emission=minimum.omega[minimum.emitter],
                        dipole=np.asarray(minimum.dipole)[root],
                        f_osc=float(np.asarray(minimum.f_osc)[root]))
    return rates.Minimum(emitter=minimum.emitter, reservoir=minimum.reservoir,
                         energy=minimum.energy,
                         socme=np.asarray(minimum.socme), **derivative,
                         **emission, e_ground=minimum.e0)


def aligned_minima(template, coords0, coords_s1, coords_t1):
    """(R0, (S1 minimum, rotation), (T1 minimum, rotation)): molecules of
    `template` at the three geometries (Bohr), each minimum rigidly
    superposed on R0 (`align_to`), where the S0 modes are expressed; the
    rotation turns a vector of the minimum's own frame into R0's."""
    mol0 = molecule_at(template, coords0)
    return (mol0,
            align_to(mol0, molecule_at(template, coords_s1),
                     return_rotation=True),
            align_to(mol0, molecule_at(template, coords_t1),
                     return_rotation=True))


def manifold_gaps(rule):
    """The `rates.ManifoldGaps` of the declared `rule`: the run relaxes S1
    and T1 alone, so under 'adiabatic' no other state has a minimum."""
    return rates.ManifoldGaps(rule, None if rule == 'stacked' else {})


def tadf_rate_table(modes, vert, s1, t1, template, temperature,
                    lambda_env_ev, gap_rule, derivatives=None):
    """`rates.rate_table` of the S0 `modes` and the `StateMinimum` records
    `vert` (R0), `s1` and `t1`, with what it was built on. `template` is the
    run's molecule (atoms and basis); temperature K, lambda_env_ev eV;
    gap_rule the declared gaps of the state manifold (`manifold_gaps`).

    derivatives: {'S1': fields, 'T1': fields} of `rates.Minimum` (dv_dq,
    omega_promoting, dv_dq_paths) that replace each minimum's own spin-orbit
    derivative, where another model supplies it (`uisc`), or a string saying
    why that crossing has none (Condon only); None reads the minima's
    own."""
    methods = {m.where: m.method for m in (vert, s1, t1)}
    if len(set(methods.values())) != 1:
        raise RateRefused(f'one table differences the energies of R0, S1 and '
                          f'T1: they come from one method, not {methods}')
    if (vert.where, s1.where, t1.where) != ('R0', 'S1', 'T1'):
        raise RateRefused('the records are R0, the S1 minimum and the T1 '
                          'minimum, in that order')
    omega = modes.omega
    mol0, (s1_geom, rot_s1), (t1_geom, rot_t1) = aligned_minima(
        template, vert.coords_bohr, s1.coords_bohr, t1.coords_bohr)
    # displacements in curvilinear coordinates, linearized at R0 where the
    # modes were taken: a rotated group puts nothing on the stretches
    s_k, _ = huang_rhys_curvilinear(mol0, s1_geom, t1_geom, omega,
                                    modes.modes, modes.masses)
    s_ic, _ = huang_rhys_curvilinear(mol0, mol0, s1_geom, omega, modes.modes,
                                     modes.masses)
    s_t, _ = huang_rhys_curvilinear(mol0, mol0, t1_geom, omega, modes.modes,
                                    modes.masses)
    d_k = None
    if s1.d_ground is not None:
        d_k = project_coupling(np.asarray(s1.d_ground) @ rot_s1, modes.modes,
                               modes.masses)
    minima = [rate_minimum(s1, modes, rot_s1, emits=True),
              rate_minimum(t1, modes, rot_t1)]
    supplied = derivatives or {}
    minima = [replace(m, **supplied[where])
              if isinstance(supplied.get(where), dict) else m
              for m, where in zip(minima, ('S1', 'T1'))]
    out = rates.rate_table(minima[0], minima[1],
                           vert.e0, omega, s_k, temperature,
                           lambda_env_ev / HARTREE_TO_EV,
                           s_ic=None if d_k is None else s_ic, d_ic=d_k,
                           s_t=s_t, manifold_gaps=manifold_gaps(gap_rule))
    out['labels']['R0'] = [vert.emitter, vert.reservoir]
    out.update(
        dE_ST_vertical_eV=(vert.omega[vert.emitter]
                           - vert.omega[vert.reservoir]) * HARTREE_TO_EV,
        hessian_route=modes.route, relaxed_solvation=s1.solvation,
        method=methods['S1'],
        channels_refused={m.where: refused_channels(
            m, None if derivatives is None else derivatives[m.where])
            for m in (s1, t1)})
    return out


def print_rate_table(where, table):
    """The rate block a log reader wants."""
    log(f'=============== {where}, {table["temperature_K"]:.0f} K, '
        f'{table.get("method")} ===============')
    for key in ('dE_ST_adiabatic_eV', 'dE_ST_vertical_eV', 'S1_adiabatic_eV',
                'T1_adiabatic_eV', 'lambda_T_eV', 'lambda_S_eV',
                'huang_rhys_S1T1_sum', 'lambda_sum_rule_eV', 'emission_eV',
                'f_emission', 'v_s0t1_cm', 'dE_T2T1_vertical_eV', 'v_s1t2_cm'):
        value = table.get(key)
        log(f'  {key:24s} '
            + ('None' if value is None else f'{value:12.6g}'))
    for name, ch in table['channels'].items():
        rms = ch['dv_dq_rms_cm']
        log(f'  {name}: |V| {ch["v_condon_cm"]:.4f} cm^-1, rms |dV/dq| '
            + ('Condon only' if rms is None else
               f'{rms:.4f} cm^-1 (' + ', '.join(
                   f'{p} {x:.4f}' for p, x in
                   ch['dv_dq_rms_by_path_cm'].items()) + ')'))
        lams = ch['marcus_mlj']
        if lams['status'] == 'refused':
            log(f'      A1_marcus, A2_mlj REFUSED: sum-rule lambda '
                f'{lams["lambda_sum_rule_eV"]:.4f} eV against '
                f'{lams["lambda_energy_eV"]:.4f} eV from the energies')
        for k in ('A1_marcus', 'A2_mlj', 'A3_fc_condon', 'A4_fc_condon_ht',
                  'A4_ht_part'):
            if ch[k] is not None:
                log(f'      {k:18s} {ch[k]:12.4e} s^-1')
    for key in ('k_r', 'k_ic', 'k_nr_t1', 'k_risc_t2', 'k_isc_t2'):
        value = table.get(key)
        log(f'  {key:24s} ' + ('None' if value is None
                               else f'{value:12.4e} s^-1'))
    if table.get('t2_status') == 'refused':
        log(f'  T2 channels REFUSED with {", ".join(table["t2_refused_by"])}')
    for name in ('pl', 'pl_with_t2', 'pl_manifold'):
        pl = table.get(name)
        if pl is None:
            continue
        mark = ('' if name != 'pl_manifold'
                or table.get('pl_manifold_complete')
                else ' [INCOMPLETE manifold: a lower bound on its crossings]')
        log(f'  {name}: prompt {pl["k_prompt"]:.4e} s^-1 (phi '
            f'{pl["phi_prompt"]:.4f}), delayed {pl["k_delayed"]:.4e} s^-1 '
            f'(phi {pl["phi_delayed"]:.4f}), PLQY {pl["phi_total"]:.4f}'
            + mark)
    flags = table.get('numerical_flags')
    if flags:
        log(f'  FLAGGED: every number above stands on {flags}')


__all__ = ['RateRefused', 'refused_channels', 'rate_minimum',
           'aligned_minima', 'manifold_gaps', 'tadf_rate_table',
           'print_rate_table']
