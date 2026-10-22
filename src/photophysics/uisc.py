"""The UISC protocol of Samanta et al. as a second rate model on the shared
stages.

Samanta, Kim, Coropceanu and Bredas, J. Am. Chem. Soc. 139, 4042 (2017),
compute intersystem crossing between a singlet and a triplet manifold with
Fermi's golden rule. Their spin-orbit element is evaluated at one geometry
(the Condon term), their reorganization energy is scanned rather than
computed, and the Huang-Rhys factor of the crossing is set to zero (classical
Marcus rate). This module keeps that protocol as one rate model beside the
TADF one and lets each approximation be switched on in turn:

    A0  Samanta: Marcus, S = 0, lambda scanned, |V| at one geometry
    A1  lambda computed from the two minima
    A2  the Huang-Rhys factor computed (Marcus-Levich-Jortner)
    A3  every mode explicit (Franck-Condon, Condon)
    A4  A3 plus the spin-vibronic term dV/dq (Herzberg-Teller)

A1 to A4 and the decay channels are `rates.rate_table`, the table the TADF
model reads too; only two things are added here.

THE VIB STAGE (`vib`). An El-Sayed-forbidden pair has a Condon element near
zero, and the derivative of the spin-orbit element along a mode that lends
the pair the symmetry it lacks carries the process. The derivative is taken
by finite differences at each crossing's initial minimum (intersystem
crossing at the S1 minimum, reverse crossing at the T1 minimum), along the
ground-state modes of largest Huang-Rhys weight for the S1 <-> T1
displacement. The promoting modes are unknown until they are computed and a
full scan costs two displaced GW-BSE solves per mode and minimum, so the modes
are picked by Huang-Rhys weight, which is not the same as picking them by
|dV/dq|.

|V| is a norm and passes through zero for a forbidden pair, so the slope is
differenced through V^2 (`spin_orbit.slope_from_squares`, exact for V linear
in q, Condon term included). Here V^2 is the Frobenius
sum |<S_i|H_SO|T_j>|^2 over the lowest `s_block` singlets and `t_block`
triplets: the root that carries the coupling can change its index under a
displacement, so the pair is not assumed. Each displaced root is matched to
the centre's by the overlap of its Casida vector (`follow`), a sign being
arbitrary and an index meaningless across geometries.

The displaced manifolds are solved on the chain of the run (the declared
mean field, the quasiparticle set, the continuum and the frozen conventions),
rebuilt once at each crossing's centre so that every displaced point shares
the centre's conventions.

THE RATE ROWS (`uisc_rate_table`). The same table as the TADF model, with each
crossing's spin-orbit derivative taken from the vib record in place of the
analytic one, and per channel the A0 rows: Marcus rates with S = 0 at each
reorganization energy Samanta et al. scan.
"""
import time
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from src.Base.constants import HARTREE_TO_CM, HARTREE_TO_EV
from src.Base.pyscf_interface import occupied_count
from src.photophysics import rate_models
from src.photophysics.declaration import UiscModel
from src.photophysics.minimum import GroundModes
from src.photophysics.rate_models import RateRefused, aligned_minima
from src.photophysics.records import StageRefused, log
from src.properties import rates
from src.properties.nonadiabatic import align_roots
from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR,
                                       interstate_vectors, slope_from_squares,
                                       soc_operator_mo)
from src.properties.surface import lockstep_geometry
from src.properties.curvilinear import huang_rhys_curvilinear
from src.properties.vibronic import displace_along_mode



@dataclass(frozen=True)
class SamantaRow:
    """One row of Samanta et al., Table 1: the calculated and measured
    singlet-triplet gap (eV), and the upper bounds they report for the
    spin-orbit element (cm^-1) and the rate (s^-1)."""
    row: str
    de_calc_ev: float
    de_expt_ev: float
    socme_below_cm: float
    k_below: float


class Crossing(NamedTuple):
    """A crossing: its record key, the minimum it starts from, that
    minimum's stage, and its channel of `rates.rate_table`."""
    name: str
    state: str
    stage: str
    channel: str


#: The rows of Samanta et al., Table 1 checked against the paper, by system
#: name; a system absent here gets no comparison line.
SAMANTA_REFERENCE = {
    'cbp': SamantaRow(row='CBP', de_calc_ev=0.88, de_expt_ev=0.71,
                      socme_below_cm=0.01, k_below=0.01),
}

#: The crossings: intersystem crossing from the S1 minimum, reverse crossing
#: from the T1 minimum.
CROSSINGS = (Crossing('isc', 'S1', 's1', 'ISC S1->T1'),
             Crossing('risc', 'T1', 't1', 'RISC T1->S1'))

#: The derivative keys of the vib record (it also holds `huang_rhys` and
#: `seconds`).
VIB_KEYS = ('modes', 'omega', 'dv_dq_isc', 'dv_dq_risc', 'v0_isc', 'v0_risc',
            'overlap_isc', 'overlap_risc')


# ---------------------------------------------------------- the electronics
def t_vectors(h_mo, nocc, singlet, triplet):
    """(n_s, n_t, 3): T_eta of every singlet-triplet root pair, the vector
    whose norm is the spin-orbit coupling. `singlet` and `triplet` are the
    solver's (omega, X, Y) triples; Y may be None (Tamm-Dancoff)."""
    _, x_s, y_s = singlet
    _, x_t, y_t = triplet
    return interstate_vectors(h_mo, nocc, x_s, y_s, x_t, y_t) \
        * EXCITED_SPIN_FACTOR


def follow(reference, displaced):
    """(order, phase, cosine) matching the roots of `displaced` to those of
    `reference`, both (omega, X, Y) triples.

    A Casida eigenvector's sign is arbitrary and its index means nothing
    across geometries, so the pair being differenced is identified by the
    overlap of the full (X, Y) vectors rather than assumed: root i of the
    reference is root order[i] of the displaced set, up to the sign
    phase[i], with cosine[i] the size of the overlap (1 is the same root).
    Each displaced root is assigned once (`nonadiabatic.align_roots`, the
    largest overlap first), so two reference roots never take the same one.
    """
    ref_x, ref_y = reference[1], reference[2]
    x, y = displaced[1], displaced[2]
    ref_y = np.zeros_like(ref_x) if ref_y is None else ref_y
    y = np.zeros_like(x) if y is None else y
    ov = ref_x.T @ x + ref_y.T @ y
    nr = np.sqrt((ref_x ** 2 + ref_y ** 2).sum(axis=0))
    nd = np.sqrt((x ** 2 + y ** 2).sum(axis=0))
    # align_roots reads the excited block of an overlap with the ground
    # state in row and column 0
    s = np.zeros((ov.shape[0] + 1, ov.shape[1] + 1))
    s[1:, 1:] = ov / np.outer(nr, nd)
    order, phase, _ = align_roots(s, ov.shape[0])
    return order, phase, np.abs(s[1:, 1:])[np.arange(len(order)), order]


def chain_at(run, centre, record):
    """The run's relaxed chain with its conventions rebuilt at `centre` (the
    minimum of the stage `record`), grow-only from that minimum's explicit
    quasiparticle set: the chain every displaced evaluation about `centre`
    shares."""
    floor = (record.get('qp_partition') or {}).get('explicit')
    return run.rebuilt_for_t1(centre, floor)


def solve_at(run, chain_, mol):
    """(singlet, triplet, T) at the geometry `mol` on `chain_`: both
    manifolds as (omega, X, Y) triples from ONE quasiparticle set and one W,
    and `t_vectors` of them. The spin-orbit operator is the mean-field one of
    the spin-orbit stage ('somf')."""
    manifold = run.manifold(run.s_key, run.t_key, chain=chain_)
    ev = manifold.evaluate(mol)
    singlet, triplet = ev.spectrum['singlet'], ev.spectrum['triplet']
    h_mo = soc_operator_mo(ev.mf, ev.mol, two_electron='somf')
    return singlet, triplet, t_vectors(h_mo, occupied_count(ev.mf), singlet,
                                       triplet)


# ------------------------------------------------------------ the derivative
def promoting_modes(s_k, omega, model):
    """The indices of the `model.n_modes` modes of largest Huang-Rhys
    weight `s_k`, largest first. An imaginary mode is not a displacement
    direction at a minimum and is refused
    (`rates.require_vibrational_modes`)."""
    rates.require_vibrational_modes(omega, 'promoting_modes')
    order = [int(k) for k in np.argsort(-np.asarray(s_k), kind='stable')]
    return order[:model.n_modes]


def block_norm(t_vec, model):
    """sqrt(sum |T|^2) over the lowest `s_block` singlets and `t_block`
    triplets of the (n_s, n_t, 3) table `t_vec`, and the block's shape."""
    ns = min(model.s_block, t_vec.shape[0])
    nt = min(model.t_block, t_vec.shape[1])
    return float(np.sqrt((t_vec[:ns, :nt] ** 2).sum())), ns, nt


def crossing_derivative(run, chain_, centre, modes, picked, model, label):
    """(v0, dv_dq, overlap) of one crossing at its minimum `centre`: the
    Condon |V| of the block, |dV/dq_k| for each picked mode k by the
    difference of V^2 (module docstring), and the smallest overlap of any
    followed root with its centre root, per mode."""
    singlet0, triplet0, t0 = solve_at(run, chain_, centre)
    v0, ns, nt = block_norm(t0, model)
    dv, overlap = [], []
    for k in picked:
        w_k = float(modes.omega[k])
        v2, worst = {}, 1.0
        for sign in (+1, -1):
            t_start = time.time()
            m = lockstep_geometry(displace_along_mode(
                centre, modes.modes, modes.masses, modes.omega, k,
                sign * model.dq))
            singlet, triplet, t_vec = solve_at(run, chain_, m)
            o_s, p_s, c_s = follow(singlet0, singlet)
            o_t, p_t, c_t = follow(triplet0, triplet)
            worst = min(worst, float(c_s[:ns].min()), float(c_t[:nt].min()))
            t_vec = (t_vec[np.ix_(o_s, o_t)] * p_s[:, None, None]
                     * p_t[None, :, None])
            v2[sign] = float((t_vec[:ns, :nt] ** 2).sum())
            log(f'vib {label} minimum, mode {k} '
                f'({w_k * HARTREE_TO_CM:7.1f} cm^-1) '
                f'{"+" if sign > 0 else "-"}dq  overlap {worst:.3f}', t_start)
        dv.append(slope_from_squares(v2[+1], v2[-1], v0 ** 2, model.dq))
        overlap.append(worst)
    return v0, np.array(dv), np.array(overlap)


def vib(run):
    """vib: dV/dq along the promoting modes at the S1 and T1 minima.

    Returns the record (`VIB_KEYS`, `huang_rhys`, `seconds`): the picked
    modes' indices and frequencies (Hartree), |dV/dq| of each in Hartree per
    dimensionless coordinate at the S1 minimum (`dv_dq_isc`) and at the T1
    minimum (`dv_dq_risc`), each minimum's Condon |V| of the block and the
    smallest root-following overlap met per mode. Taken from the store where it is
    there under this declaration's stamp.
    """
    d = run.declaration
    model = d.rate_model
    if not isinstance(model, UiscModel):
        raise StageRefused('vib is a stage of the UISC rate model; this run '
                           'declares another')
    if run.reusable('vib'):
        log('vib stored')
        return run.read('vib')
    missing = [n for n in ('hess', 'vert', 's1', 't1') if not run.reusable(n)]
    if missing:
        raise StageRefused(f'vib: needs {", ".join(missing)}')
    t_all = time.time()
    modes = GroundModes.from_record(run.read('hess'))
    records = {n: run.read(n) for n in ('vert', 's1', 't1')}
    if run.mol0 is None:
        run.mol0 = run.molecule_at(records['vert']['props']['coords_bohr'])
    # each minimum in R0's frame, where the S0 modes are expressed
    mol_r0, (s1_geom, _), (t1_geom, _) = aligned_minima(
        run.mol_in, *(records[n]['props']['coords_bohr']
                      for n in ('vert', 's1', 't1')))
    centres = {'S1': run.molecule_at(s1_geom.atom_coords()),
               'T1': run.molecule_at(t1_geom.atom_coords())}
    s_k, _ = huang_rhys_curvilinear(
        mol_r0, centres['S1'], centres['T1'], modes.omega, modes.modes,
        modes.masses)
    picked = promoting_modes(s_k, modes.omega, model)
    log(f'vib: {len(picked)} promoting modes, chosen by Huang-Rhys weight; '
        f'{4 * len(picked)} displaced GW-BSE + SOC solves follow, at the S1 '
        f'and the T1 minimum')
    for k in picked:
        log(f'    mode {k:4d}  {modes.omega[k] * HARTREE_TO_CM:8.1f} cm^-1  '
            f'S_k {s_k[k]:8.4f}')
    out = {'modes': np.array(picked, int), 'omega': modes.omega[picked],
           'huang_rhys': s_k[picked]}
    for crossing in CROSSINGS:
        where = crossing.state
        centre = centres[where]
        chain_ = chain_at(run, centre, records[crossing.stage])
        v0, dv, overlap = crossing_derivative(run, chain_, centre, modes,
                                              picked, model, where)
        out.update({f'dv_dq_{crossing.name}': dv,
                    f'v0_{crossing.name}': v0,
                    f'overlap_{crossing.name}': overlap})
        log(f'vib {where} minimum: Condon |V| {v0 * HARTREE_TO_CM:.5f} '
            f'cm^-1, rms |dV/dq| '
            f'{np.sqrt((dv ** 2).mean()) * HARTREE_TO_CM:.5f}')
    out['seconds'] = time.time() - t_all
    run.write('vib', out)
    return run.read('vib')


# ------------------------------------------------------------------- rates
def derivative_of(vib_record, crossing, modes):
    """({'dv_dq', 'omega_promoting'}) of `crossing` from the vib record, or
    why a crossing without one is Condon only. Refused where the record's
    modes are not the ground-state modes it is read against."""
    if vib_record is None:
        return 'Condon only (no vib record)'
    picked = np.asarray(vib_record['modes'], int)
    omega = np.asarray(vib_record['omega'], float)
    if picked.size and (picked.max() >= len(modes.omega)
                        or not np.array_equal(modes.omega[picked], omega)):
        raise RateRefused('the vib record was taken on other ground-state '
                          'modes than these: its frequencies are not theirs')
    return {'dv_dq': np.asarray(vib_record[f'dv_dq_{crossing}'], float),
            'omega_promoting': omega}


def samanta_rows(table, model, temperature):
    """The A0 rows of `table`, per channel: the Marcus rate with S = 0 at
    each of `model.lambda_samanta` (eV), the spin-orbit element at the
    crossing's own minimum and the gap's sign of the table (energy of the
    final state less the initial one)."""
    gap = table['dE_ST_adiabatic_eV'] / HARTREE_TO_EV
    out = {}
    for crossing in CROSSINGS:
        ch = table['channels'][crossing.channel]
        # final less initial: down from S1, up from T1
        de = -gap if crossing.state == 'S1' else gap
        v = ch['v_condon_cm'] / HARTREE_TO_CM
        out[crossing.channel] = {
            'v_condon_cm': ch['v_condon_cm'],
            'delta_E_eV': de * HARTREE_TO_EV,
            'lambda_eV': [float(x) for x in model.lambda_samanta],
            'k': [float(rates.marcus_rate(v, de, lam / HARTREE_TO_EV,
                                          temperature))
                  for lam in model.lambda_samanta]}
    return out


def uisc_rate_table(modes, vert, s1, t1, vib_record, template, temperature,
                    lambda_env_ev, model, gap_rule):
    """The UISC rate table: `rates.rate_table` of the S0 `modes` and the
    `StateMinimum` records `vert` (R0), `s1` and `t1`, the spin-orbit
    derivative of each crossing from the vib record (`dv_dq_isc` at the S1
    minimum, `dv_dq_risc` at the T1 minimum) in place of the analytic one,
    and the A0 rows of Samanta et al. under 'samanta'.

    `vib_record` None leaves both crossings Condon only. `template` is the
    run's molecule (atoms and basis); temperature K, lambda_env_ev eV, `model`
    the `UiscModel`, `gap_rule` the declared gaps of the state manifold. Every geometry is superposed on R0 first, as in the TADF
    table (`rate_models.tadf_rate_table`), and the three records must come
    from one method.
    """
    if not isinstance(model, UiscModel):
        raise TypeError('model is the UiscModel of the declaration')
    # the analytic derivative is not this model's: the vib record's is
    own = [replace(m, soc_dv_cart=None, soc_dv_paths=None) for m in (s1, t1)]
    out = rate_models.tadf_rate_table(
        modes, vert, own[0], own[1], template, temperature, lambda_env_ev,
        gap_rule, derivatives={c.state: derivative_of(vib_record, c.name, modes)
                     for c in CROSSINGS})
    out.update(rate_model='uisc',
               samanta=samanta_rows(out, model, temperature))
    return out


def print_uisc_table(system, environment, table):
    """The block of the UISC table a log reader wants: what is Samanta's and
    what is computed here, the A0 rows against the reference values they
    report where known (the system's name selects the row), then the table
    itself (`rate_models.print_rate_table`)."""
    where = f'{system}, {environment}'
    ref = SAMANTA_REFERENCE.get(system.lower())
    log(f'=============== {where}: Samanta rows (A0), '
        f'{table["temperature_K"]:.0f} K ===============')
    lams = next(iter(table['samanta'].values()))['lambda_eV']
    log(f'  Delta_E_ST adiabatic {table["dE_ST_adiabatic_eV"]:8.4f} eV'
        + (f'   [Samanta {ref.de_calc_ev} calc / {ref.de_expt_ev} exptl]'
           if ref else '   [no Samanta reference for this system]'))
    log(f'  lambda_T (ISC)       {table["lambda_T_eV"]:8.4f} eV   '
        f'[Samanta ASSUMED {" / ".join(f"{x:.2f}" for x in lams)}]')
    log(f'  lambda_S (RISC)      {table["lambda_S_eV"]:8.4f} eV')
    log(f'  sum S_k              {table["huang_rhys_S1T1_sum"]:8.4f}'
        f'          [Samanta set S = 0]')
    for name, row in table['samanta'].items():
        log(f'  {name}, |V| Condon {row["v_condon_cm"]:8.4f} cm^-1'
            + (f'   [Samanta <{ref.socme_below_cm}]' if ref else ''))
        for lam, k in zip(row['lambda_eV'], row['k']):
            log(f'    A0  Samanta (lam={lam:.2f})        {k:12.4e} s^-1')
    rate_models.print_rate_table(where, table)


__all__ = ['SamantaRow', 'Crossing', 'SAMANTA_REFERENCE', 'CROSSINGS',
           'VIB_KEYS', 't_vectors', 'follow', 'vib', 'uisc_rate_table',
           'print_uisc_table']
