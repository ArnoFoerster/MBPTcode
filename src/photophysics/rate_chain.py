"""Intersystem-crossing and reverse intersystem-crossing rates of one method
from its own minima, on the best affordable model, and every ablation of it.

ONE PROTOCOL FOR EVERY METHOD. `RateInputs` holds what a method's run hands
the rates: its S1 and T1 minima (`minimum.StateMinimum`: energies of every
computed state there, the spin-orbit table, the Cartesian derivative of the
crossing's element), the gradient of each state at the other's minimum, the
solvent's equilibrium and non-equilibrium energies, and two Hessians at
their own minima (S1 and T1; in practice TD-DFT at the TD-DFT minima, read
with `duschinsky.read_orca_hessian`). GW-BSE, full BSE, ADC(2) and SCS-ADC(2)
fill the same fields; nothing below knows which method it reads.

THE FULL MODEL (`RateModel.full`). For S1 -> T1 (ISC) and T1 -> S1 (RISC):

- the bath is a Duschinsky model of the two states (`duschinsky.mode_sets`
  of the two Hessians), its rotation J measured through the valence
  internals ('curvilinear', so a large torsion between S1 and T1 stays a
  torsion; 'rectilinear', the overlap of the Cartesian mode sets, is a rung
  of the ladder), its T1 modes below DUSCHINSKY_CLASSICAL_BELOW_CM taken out
  as one classical, frequency-unchanged block whose reorganization (the
  initial state's own) joins the classical broadening and whose zero-point
  difference shifts the gap (`duschinsky.split_soft_modes`; quantum soft
  modes are a rung of the ladder), its displacement K taken from the METHOD's own
  minima in valence internal coordinates (`displacement_from_minima`) or
  from its gradient at the other state's minimum
  (`displacement_from_gradient`), as declared; a model that does not
  describe a direction (`duschinsky.duschinsky_guard`: soft modes mixed into
  stiff ones, a sum-rule lambda or a band first moment far from the
  accepting state's energy difference between the minima) is refused, and
  the ladder goes on to the displaced oscillators;
- the zero-point energy difference of the two mode sets enters through the
  Duschinsky correlation function's phase;
- the classical broadening is the solvent reorganization, the accepting
  state's non-equilibrium minus equilibrium energy at the initial minimum
  (`solvent_reorganization`), zero in the gas phase;
- the coupling is the Condon element plus its nuclear derivative (direct
  Herzberg-Teller and second-order spin-vibronic paths through the higher
  states, as the method's `soc_dv_cart` carries them; a producer with
  spin-orbit vectors and derivative couplings builds it with
  `spin_vibronic.second_order_derivative`), on the initial state's modes,
  added incoherently (`rates.spin_vibronic_rate`);
- the triplets T1, T2, T3 at the T1 minimum and the singlets at the S1
  minimum are Boltzmann populated and every thermally reachable pair is a
  channel (`rates.manifold_crossings`, stacked gaps), which carries the
  T2 -> S1 route of reverse intersystem crossing.

ABLATIONS. Every ingredient is a field of `RateModel`, recorded with the
rates, and `ablation_ladder` removes them one at a time, in the order
`ABLATION_LADDER`, down to the effective-mode protocol of Samanta, Kim,
Coropceanu and Bredas, J. Am. Chem. Soc. 139, 4042 (2017): the method's own
spin-orbit coupling -> a shared reference coupling; spin-vibronic -> Condon;
the triplet manifold -> T1 alone; (beside the ladder: every soft mode
quantum, and the Cartesian Duschinsky matrix in place of the curvilinear
one); Duschinsky off
(displaced oscillators on the T1 modes, the zero-point difference kept as a
shift); zero-point difference off; every mode -> one Marcus-Levich-Jortner effective mode; the
computed classical reorganization -> an assumed one.

Energies in Hartree (eV where a name says so), temperature in Kelvin, rates
in s^-1.
"""
import re
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from src.Base.geometry.geometry import molecule_at
from src.Base.numerics.constants import (BOLTZMANN_HARTREE_PER_KELVIN,
                                         DUSCHINSKY_CLASSICAL_BELOW_CM,
                                         HARTREE_TO_CM, HARTREE_TO_EV,
                                         RATE_ASSUMED_LAMBDAS_EV,
                                         RATE_FC_DENSITY_FLOOR_REL,
                                         RATE_MIXING_RATIO_MAX,
                                         RATE_MLJ_N_MAX, TRIPLET_SUBLEVELS)
from src.photophysics import rates
from src.photophysics.rate_models import RateRefused
from src.photophysics.records import key_of
from src.properties.duschinsky import (DuschinskyModel, DuschinskyOscillators,
                                       displacement_from_gradient,
                                       displacement_from_minima,
                                       duschinsky_density, duschinsky_guard,
                                       guard_message, minima_in_frame,
                                       mode_sets, split_soft_modes)
from src.properties.franck_condon import (FCIntegrandNotDecayed,
                                          fc_weighted_dos)
from src.properties.spin_vibronic import (mixed_weight, mode_derivatives,
                                          path_ratio)

#: Values of each `RateModel` switch.
SOC_SOURCES = ('own', 'shared')
COUPLINGS = ('spin_vibronic', 'condon')
TRIPLET_MODELS = ('manifold', 'T1')
DENSITIES = ('multimode', 'mlj')
MIXINGS = ('auto', 'diabatic', 'perturbative')
HUANG_RHYS_ROUTES = ('curvilinear', 'cartesian', 'gradient')
DUSCHINSKY_COORDINATES = ('curvilinear', 'rectilinear')

#: The gradient keys of `RateInputs`: 'S1@T1' is S1's gradient at the T1
#: minimum, 'T1@S1' T1's at the S1 minimum.
GRADIENT_KEYS = ('S1@T1', 'T1@S1')
#: The solvent keys of `RateInputs`: the accepting state at the initial
#: minimum, of ISC ('T1@S1') and of RISC ('S1@T1').
SOLVENT_KEYS = ('T1@S1', 'S1@T1')


class FCDensityAtRoundoff(RuntimeWarning):
    """A Duschinsky Franck-Condon density below RATE_FC_DENSITY_FLOOR_REL of
    its band's peak: the real-time integral does not resolve it, and the
    rate on it is the roundoff floor's ('fc_density_at_roundoff' in the
    rate chain's numerical flags)."""


@dataclass(frozen=True)
class RateModel:
    """The declared rate model: one value per ingredient, no defaults.

    soc: 'own', the method's spin-orbit table and derivative; or 'shared',
        those of the reference passed to `rate_chain` (GW-BSE's), read at
        the reference's own minima.
    coupling: 'spin_vibronic', Condon plus the element's derivative; or
        'condon'.
    triplets: 'manifold', every thermally reachable singlet and triplet; or
        'T1', the S1 <-> T1 pair alone.
    duschinsky: the two mode sets with their rotation and frequencies; off,
        displaced oscillators on the T1 modes.
    duschinsky_coordinates: J through the valence internals
        ('curvilinear', `curvilinear.curvilinear_duschinsky`) or the overlap
        of the Cartesian mode sets ('rectilinear'). It also carries the
        displacement onto the T1 modes of the displaced oscillators.
    classical_below_cm: the Duschinsky bath's T1 modes below this (cm^-1)
        and their image in S1 form one classical, frequency-unchanged block
        (`duschinsky.split_soft_modes`) whose reorganization joins the
        classical broadening; 0 keeps every mode quantum. Read only with
        duschinsky on.
    zero_point: the zero-point difference of the two mode sets in the gap.
    density: 'multimode', every mode at its own frequency; or 'mlj', one
        Marcus-Levich-Jortner effective mode (needs duschinsky off and no
        inhomogeneous width).
    classical_lambda: 'computed', the solvent reorganization of the run; or
        an assumed value in eV, for both directions.
    huang_rhys_route: the displacement K from the method's two minima,
        measured in valence internal coordinates ('curvilinear') or on the
        straight Cartesian line ('cartesian', which puts a rotated methyl or
        amino group on the C-H and N-H stretches), or from its gradient at
        the other minimum ('gradient').
    gaussian_sigma_ev: an inhomogeneous Gaussian standard deviation on top,
        eV; zero for none (a gas-phase Duschinsky density needs one).
    mixing: how a second-order path enters the coupling: 'perturbative',
        its derivative as given (first-order mixing, which diverges as the
        path state's gap closes); 'diabatic', the two-state linear vibronic
        pair mixed exactly over the thermal ensemble (`_mixing`); 'auto',
        diabatic where a path's coupling-to-gap ratio exceeds
        RATE_MIXING_RATIO_MAX, perturbative otherwise. Each direction
        records which it used.
    """
    soc: str
    coupling: str
    triplets: str
    duschinsky: bool
    duschinsky_coordinates: str
    classical_below_cm: float
    zero_point: bool
    density: str
    classical_lambda: object
    huang_rhys_route: str
    gaussian_sigma_ev: float
    mixing: str

    def __post_init__(self):
        for name, allowed in (('soc', SOC_SOURCES), ('coupling', COUPLINGS),
                              ('triplets', TRIPLET_MODELS),
                              ('density', DENSITIES),
                              ('mixing', MIXINGS),
                              ('duschinsky_coordinates',
                               DUSCHINSKY_COORDINATES),
                              ('huang_rhys_route', HUANG_RHYS_ROUTES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f'RateModel.{name} = {getattr(self, name)!r} '
                                 f'not in {allowed}')
        for name in ('duschinsky', 'zero_point'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'RateModel.{name} is True or False')
        lam = self.classical_lambda
        if lam != 'computed' and not (isinstance(lam, (int, float))
                                      and not isinstance(lam, bool)
                                      and lam > 0):
            raise ValueError(f"classical_lambda is 'computed' or a positive "
                             f'energy in eV, got {lam!r}')
        cut = self.classical_below_cm
        if isinstance(cut, bool) or not isinstance(cut, (int, float)) \
                or not cut >= 0.0:
            raise ValueError(f'classical_below_cm is a cutoff >= 0 in '
                             f'cm^-1, got {cut!r}')
        if not self.gaussian_sigma_ev >= 0.0:
            raise ValueError(f'gaussian_sigma_ev = {self.gaussian_sigma_ev} '
                             f'is negative')
        if self.density == 'mlj' and self.duschinsky:
            raise ValueError('one effective mode has no Duschinsky rotation: '
                             'the MLJ density needs duschinsky=False')
        if self.density == 'mlj' and self.gaussian_sigma_ev > 0.0:
            raise ValueError('the MLJ density has no inhomogeneous width: '
                             'gaussian_sigma_ev must be 0')

    @classmethod
    def full(cls, huang_rhys_route, gaussian_sigma_ev):
        """The best affordable model: every ingredient on."""
        return cls(soc='own', coupling='spin_vibronic', triplets='manifold',
                   duschinsky=True, duschinsky_coordinates='curvilinear',
                   classical_below_cm=DUSCHINSKY_CLASSICAL_BELOW_CM,
                   zero_point=True, density='multimode',
                   classical_lambda='computed',
                   huang_rhys_route=huang_rhys_route,
                   gaussian_sigma_ev=gaussian_sigma_ev, mixing='auto')


#: The ablation ladder: (step, the switches it changes), cumulative, each
#: removing one ingredient from the step before, except the SIDE_STEPS, which
#: change the step before them and are not carried on; the last rung is taken
#: at every RATE_ASSUMED_LAMBDAS_EV.
ABLATION_LADDER = (
    ('full', {}),
    ('perturbative_mixing', {'mixing': 'perturbative'}),
    ('shared_soc', {'soc': 'shared'}),
    ('condon', {'coupling': 'condon'}),
    ('t1_only', {'triplets': 'T1'}),
    ('quantum_soft_modes', {'classical_below_cm': 0.0}),
    ('rectilinear_duschinsky', {'duschinsky_coordinates': 'rectilinear'}),
    ('no_duschinsky', {'duschinsky': False}),
    ('no_zero_point', {'zero_point': False}),
    ('mlj', {'density': 'mlj'}),
) + tuple((f'assumed_lambda_{lam:g}eV', {'classical_lambda': lam})
          for lam in RATE_ASSUMED_LAMBDAS_EV)
#: The rungs of the ladder that are not carried on to the next.
SIDE_STEPS = ('quantum_soft_modes', 'rectilinear_duschinsky') + tuple(
    f'assumed_lambda_{lam:g}eV' for lam in RATE_ASSUMED_LAMBDAS_EV)


@dataclass(frozen=True)
class RateInputs:
    """One method's ingredients of the rates, method-neutral.

    s1, t1: the `StateMinimum` of the S1 and of the T1 minimum, one method.
    hessians: {'S1': h, 'T1': h}, each a `duschinsky.CartesianHessian` (or
        anything with .mol and .hessian) at its own minimum, the source of
        the mode sets; the method's minima need not be those geometries.
    gradients: {'S1@T1': g, 'T1@S1': g}, (natm, 3) Ha/Bohr, the gradient of
        one state at the other's minimum in that minimum's frame; either may
        be absent, both may (then the gradient route is refused).
    solvent: {'T1@S1': {'equilibrium': E, 'nonequilibrium': E}, 'S1@T1':
        ...}: the accepting state's total energy at the initial minimum, in
        its own equilibrium reaction field and in the initial state's slow
        one, Hartree; empty in the gas phase, and where no scheme-B energies
        exist (a model with classical_lambda='computed' then refuses minima
        in a medium).
    """
    s1: object
    t1: object
    hessians: dict
    gradients: dict
    solvent: dict

    def __post_init__(self):
        if (self.s1.where, self.t1.where) != ('S1', 'T1'):
            raise RateRefused('RateInputs takes the S1 minimum and the T1 '
                              f'minimum, got {self.s1.where}, {self.t1.where}')
        if self.s1.method != self.t1.method:
            raise RateRefused(f'one method differences the two minima, not '
                              f'{self.s1.method} and {self.t1.method}')
        if set(self.hessians) != {'S1', 'T1'}:
            raise RateRefused(f'hessians are S1 and T1, got '
                              f'{sorted(self.hessians)}')
        extra = set(self.gradients) - set(GRADIENT_KEYS)
        if extra:
            raise RateRefused(f'gradients {sorted(extra)} not in '
                              f'{GRADIENT_KEYS}')
        if self.gas and self.solvent:
            raise RateRefused('gas-phase minima carry no solvent energies')
        if self.solvent and set(self.solvent) != set(SOLVENT_KEYS):
            raise RateRefused(f"the accepting state's solvent energies are "
                              f'{SOLVENT_KEYS}, both, got '
                              f'{sorted(self.solvent)}')

    @property
    def gas(self):
        return {self.s1.solvation, self.t1.solvation} == {'gas'}

    @property
    def method(self):
        return self.s1.method


def solvent_reorganization(entry):
    """lambda_solv = E_noneq - E_eq of one `RateInputs.solvent` entry,
    Hartree; a negative one is refused (the equilibrium field is the
    accepting state's lowest)."""
    lam = float(entry['nonequilibrium']) - float(entry['equilibrium'])
    if lam < 0.0:
        raise RateRefused(f'non-equilibrium energy below the equilibrium one '
                          f'by {-lam * HARTREE_TO_EV * 1e3:.3f} meV: the '
                          f'solvent energies are not of one state')
    return lam


class _Bath:
    """rho_FC(dE) of one crossing direction on the declared model, memoized,
    with the numerical flags its evaluations raised."""

    def __init__(self, model, dmodel, s_g, omega_g, direction, lam, zpe,
                 temperature, reach, flags, zpe_block=0.0):
        self.model, self.dmodel, self.direction = model, dmodel, direction
        self.s_g, self.omega_g = s_g, omega_g
        self.lam, self.zpe, self.temperature = lam, zpe, temperature
        # the classical block's zero-point difference, outside G's phase
        self.zpe_block = zpe_block
        self.flags, self.values = flags, {}
        kt = BOLTZMANN_HARTREE_PER_KELVIN * float(temperature)
        self.sigma2 = (model.gaussian_sigma_ev / HARTREE_TO_EV) ** 2
        if model.density == 'mlj' and not lam > 0.0:
            raise RateRefused('the MLJ density needs a positive classical '
                              'reorganization: none was computed (gas phase); '
                              'assume one')
        if model.duschinsky:
            self.width2 = 2.0 * lam * kt + self.sigma2
            if not self.width2 > 0.0:
                raise RateRefused('a Duschinsky density needs a width: no '
                                  'solvent reorganization and no '
                                  'gaussian_sigma_ev')
            k2 = DuschinskyOscillators(dmodel).cumulants(temperature,
                                                         direction)[1]
            # beyond every gap the chain reads: the pair's and the
            # manifold's, each shifted by a mode and the classical bath
            self.anchor = (2.0 * float(max(dmodel.omega_g.max(),
                                           dmodel.omega_e.max()))
                           + lam + abs(zpe) + reach)
            self.floor = (RATE_FC_DENSITY_FLOOR_REL
                          / np.sqrt(2.0 * np.pi * (k2 + self.width2)))

    def prefetch(self, points):
        new = sorted({float(p) for p in points} - set(self.values))
        if new:
            self.values.update(zip(new, self._evaluate(np.array(new))))

    def __call__(self, de):
        self.prefetch([de])
        return self.values[float(de)]

    def _flag(self, name):
        if name not in self.flags:
            self.flags.append(name)

    def _evaluate(self, de):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', FCIntegrandNotDecayed)
            rho = self._density(de)
        for caveat in caught:
            if issubclass(caveat.category, FCIntegrandNotDecayed):
                self._flag('fc_integrand_not_decayed')
            warnings.warn_explicit(caveat.message, caveat.category,
                                   caveat.filename, caveat.lineno)
        return rho

    def _density(self, de):
        m = self.model
        if m.duschinsky:
            # the correlation function carries the zero-point phase; off,
            # the density is read where it would stand without it
            shift = self.lam + (self.zpe_block if m.zero_point
                                else -self.zpe)
            # one time grid for every point: the grid follows the largest
            # |dE| asked for, so a fixed anchor beyond them makes each value
            # independent of which others it was evaluated with
            x = de + shift
            anchor = max(self.anchor, float(np.abs(x).max()))
            rho = duschinsky_density(
                self.dmodel, np.append(x, anchor), self.temperature,
                self.direction, broadening=np.sqrt(self.width2))[:-1]
            low = rho < self.floor
            if low.any():
                warnings.warn(f'{int(low.sum())} Duschinsky densities below '
                              f'{self.floor:.3g} / Ha, the real-time '
                              f'integral\'s roundoff floor: those rates are '
                              f'not resolved', FCDensityAtRoundoff)
                self._flag('fc_density_at_roundoff')
            return rho
        shifted = de + (self.zpe if m.zero_point else 0.0)
        if m.density == 'mlj':
            s_sum = float(self.s_g.sum())
            om_eff = float((self.s_g * self.omega_g).sum() / max(s_sum, 1e-30))
            return np.array([rates.mlj_density(e, self.lam, s_sum, om_eff,
                                               self.temperature,
                                               RATE_MLJ_N_MAX)
                             for e in shifted])
        return np.atleast_1d(fc_weighted_dos(
            shifted, self.s_g, self.omega_g, self.temperature,
            lambda_classical=self.lam,
            broadening=np.sqrt(self.sigma2)))


def _element(minimum, singlet, triplet):
    """|<singlet|H_SO|triplet>| of a minimum's table, by label."""
    return abs(float(np.asarray(minimum.socme)[key_of(singlet)[1] + 1,
                                               key_of(triplet)[1]]))


def _coupling_minima(inputs, model, shared):
    """(S1 minimum, T1 minimum) whose spin-orbit table and derivative the
    model reads: the method's own, or the shared reference's."""
    if model.soc == 'own':
        return inputs.s1, inputs.t1
    if shared is None:
        raise RateRefused("soc='shared' needs the reference RateInputs")
    return shared.s1, shared.t1


def _pair_rate(at, de, average, model, omega_modes, modes, masses, rotation,
               bath):
    """(k, k_condon, k_ht, record, mixing) of the reference pair read at
    `at`; `mixing` is None or the path weights `_manifold_minimum` rotates
    the spin-orbit table by."""
    v = _element(at, at.emitter, at.reservoir)
    dv = np.zeros(0)
    w = np.zeros(0)
    rec, mixing = {}, None
    if model.coupling == 'spin_vibronic':
        if at.soc_dv_cart is None:
            raise RateRefused(f'{at.method} at {at.where} has no derivative '
                              f"of its element: coupling='spin_vibronic' "
                              f"needs one, 'condon' does not")

        def on_modes(x):
            return mode_derivatives(np.asarray(x), modes, masses,
                                    omega_modes, rotation=rotation)[1]

        dv_cart, condon, rec, mixing = _mixing(at, model, on_modes,
                                               omega_modes, bath.temperature)
        v = v * np.sqrt(condon)
        dv = on_modes(dv_cart)
        w = np.asarray(omega_modes, float)
    bath.prefetch(np.concatenate([[de], de + w, de - w]))
    k, k_c, k_h = rates.spin_vibronic_rate(
        de, v / np.sqrt(average), dv / np.sqrt(average), w, bath,
        bath.temperature)
    return k, k_c, k_h, {'v_condon_cm': v * HARTREE_TO_CM,
                         'dv_dq_rms_cm': _rms_cm(dv) if dv.size else None,
                         'path_gaps_meV': _path_gaps(at), **rec}, mixing


#: A second-order path's label in `StateMinimum.soc_dv_paths`: 'triplet2',
#: 'singlet2' (the GW-BSE records) or 'T2', 'S2'.
_PATH_LABEL = re.compile(r'^(singlet|triplet|S|T)(\d+)$')


def _path_state(label):
    """The state label ('T2') of a path label, None for another part."""
    m = _PATH_LABEL.match(label)
    if m is None:
        return None
    return f'{m.group(1)[0].upper()}{int(m.group(2))}'


def _mixing(at, model, on_modes, omega, temperature):
    """(Cartesian derivative, Condon factor, record, weights) of the
    crossing at `at` under the declared mixing.

    Each second-order path to a state n (`StateMinimum.soc_dv_paths`) has
    its coupling-to-gap ratio r (`spin_vibronic.path_ratio`), read with
    |V_path| between n and the crossing's other state. Diabatic, the path
    state's exact thermal weight s = `mixed_weight(r)` replaces the
    first-order one r^2: the path's derivative is scaled by sqrt(s / r^2),
    which saturates where the first-order one diverges, the Condon element
    by sqrt(prod (1 - s)), the weight the crossing's own state keeps, and
    the rest of the derivative (`soc_dv_cart` less its paths) stays. A weak
    coupling (s -> r^2) is the perturbative coupling to O(r^2). `weights`
    {path state: s} rotate the manifold's table (`_manifold_minimum`).
    """
    total = np.asarray(at.soc_dv_cart, float)
    paths = {}
    for label, part in (at.soc_dv_paths or {}).items():
        state = _path_state(label)
        if state is None:
            continue
        mate = at.reservoir if state[0] == 'T' else at.emitter
        other = at.emitter if state[0] == 'T' else at.reservoir
        pair = (other, state) if state[0] == 'T' else (state, other)
        try:
            v_path = _element(at, *pair)
        except IndexError:
            v_path = None
        if v_path is None or state not in at.energy:
            raise RateRefused(f'path {label!r} at {at.where}: {state} has no '
                              f'energy or spin-orbit element there')
        r = path_ratio(on_modes(part), omega, v_path, temperature)
        paths[label] = {'state': state, 'part': np.asarray(part, float),
                        'v_path_cm': v_path * HARTREE_TO_CM,
                        'gap_meV': (at.energy[state] - at.energy[mate])
                        * HARTREE_TO_EV * 1e3,
                        'ratio': r, 'sin2_second_order': r * r,
                        'sin2': mixed_weight(r),
                        'dv_dq_rms_cm': _rms_cm(on_modes(part))}
    r_max = max((p['ratio'] for p in paths.values()), default=0.0)
    diabatic = (model.mixing == 'diabatic'
                or (model.mixing == 'auto' and r_max > RATE_MIXING_RATIO_MAX))
    record = {'mixing': {
        'model': 'diabatic' if diabatic else 'perturbative',
        'declared': model.mixing, 'ratio_max': RATE_MIXING_RATIO_MAX,
        'paths': {lab: {k: x for k, x in p.items() if k != 'part'}
                  for lab, p in paths.items()}}}
    if not diabatic:
        record['mixing']['condon_factor'] = 1.0
        return total, 1.0, record, None
    dv = total - sum((p['part'] for p in paths.values()), np.zeros_like(total))
    condon = 1.0
    for p in paths.values():
        r, s = p['ratio'], p['sin2']
        if 0.0 < r < np.inf:
            dv = dv + np.sqrt(s / (r * r)) * p['part']
        elif r == 0.0:
            dv = dv + p['part']
        condon *= 1.0 - s
    record['mixing']['condon_factor'] = condon
    return dv, condon, record, {p['state']: p['sin2'] for p in paths.values()}


def _rms_cm(x):
    """Root mean square over the modes, cm^-1."""
    return float(np.sqrt((np.asarray(x) ** 2).mean()) * HARTREE_TO_CM)


def _path_gaps(at):
    """{'T2-T1': meV, 'S2-S1': meV} at the coupling minimum: the gaps the
    second-order paths divide by. A path whose gap is small against its
    vibronic coupling is outside the perturbative model (a near-degenerate
    pair needs a diabatic treatment); the record shows both sizes."""
    out = {}
    for low in (at.reservoir, at.emitter):
        root = key_of(low)[1]
        up = f'{low[0]}{root + 2}'
        if up in at.energy:
            out[f'{up}-{low}'] = (at.energy[up] - at.energy[low]) * \
                HARTREE_TO_EV * 1e3
    return out


def _displacements(inputs, sg, se, j):
    """{route: K} of the method's minima: both displacement routes always,
    the gradient route where a gradient is supplied (S1's at the T1
    minimum first), and which gradient that was."""
    tmpl = sg.mol
    mol_t = molecule_at(tmpl, inputs.t1.coords_bohr)
    mol_s = molecule_at(tmpl, inputs.s1.coords_bohr)
    out = {c: displacement_from_minima(sg, se, mol_t, mol_s, c)
           for c in ('curvilinear', 'cartesian')}
    for key, state in (('S1@T1', 'excited'), ('T1@S1', 'ground')):
        if key in inputs.gradients:
            out['gradient'] = displacement_from_gradient(
                sg, se, j, mol_t, mol_s, inputs.gradients[key], state)
            out['gradient_from'] = key
            break
    return out


def _rotations(sg, s1, t1):
    """{where: rotation} of two minima into the mode sets' frame."""
    _, rot_t, _, rot_s = minima_in_frame(
        sg, molecule_at(sg.mol, t1.coords_bohr),
        molecule_at(sg.mol, s1.coords_bohr))
    return {'S1': rot_s, 'T1': rot_t}


def rate_chain(inputs, model, temperature, shared=None):
    """ISC and RISC of one method on the declared `RateModel`, s^-1.

    inputs: the method's `RateInputs`. shared: the reference `RateInputs`
    (GW-BSE's) whose coupling soc='shared' reads; None otherwise.

    The S1 <-> T1 bath is one Duschinsky model, T1 the 'ground' state and
    S1 the 'excited' one: ISC is its emission direction, RISC its absorption.
    dE = E_final - E_initial between the method's minima. Each direction's
    classical broadening is the accepting state's solvent reorganization at
    the initial minimum (or the assumed lambda), and the Herzberg-Teller
    channels run on the initial state's modes. A crossing out of the triplet
    averages over its sublevels (|V|^2 / TRIPLET_SUBLEVELS). Every channel
    of the manifold reads the S1 <-> T1 bath of its direction.

    Returns the record: the model, the inputs' scalars, per direction the
    pair rate (Condon and Herzberg-Teller parts) and the manifold's
    effective rate with its populations and pairs, and 'numerical_flags'.
    A refused input raises `RateRefused`.
    """
    s_cm, t_cm = _coupling_minima(inputs, model, shared)
    sg, se, j, dev = mode_sets(inputs.hessians['T1'], inputs.hessians['S1'],
                               model.duschinsky_coordinates)
    ks = _displacements(inputs, sg, se, j)
    if model.huang_rhys_route not in ks:
        raise RateRefused('the gradient route needs a gradient of one state '
                          f'at the other minimum ({GRADIENT_KEYS})')
    k_vec = ks[model.huang_rhys_route]
    try:
        dmodel = DuschinskyModel(sg.omega, se.omega, j, k_vec,
                                 orthogonality=dev, source=f'{sg.source} / '
                                                           f'{se.source}',
                                 coordinates=model.duschinsky_coordinates)
    except ValueError as why:
        # a curvilinear J outside its singular-value bar: the inputs cannot
        # carry this model, every rung on it is refused, the ladder goes on
        raise RateRefused(f'no {model.duschinsky_coordinates} Duschinsky '
                          f'model of these Hessians: {why}') from why
    k_g = dmodel.oriented('emission')[3]  # the S1 minimum on the T1 modes
    s_g = sg.omega * k_g ** 2 / 2.0
    zpe = dmodel.zero_point_shift()       # ZPE_S1 - ZPE_T1
    e_s1 = inputs.s1.energy[inputs.s1.emitter]
    e_t1 = inputs.t1.energy[inputs.t1.reservoir]
    de_st = e_s1 - e_t1
    # the bath of the Duschinsky rungs: the soft block out, classical
    bath_model, soft = dmodel, None
    if model.duschinsky and model.classical_below_cm > 0.0:
        bath_model, soft = split_soft_modes(
            dmodel, model.classical_below_cm / HARTREE_TO_CM)
    guard = (_guard(inputs, bath_model, temperature, soft)
             if model.duschinsky else None)
    rot = _rotations(sg, s_cm, t_cm)
    # the widest gap any channel reads: every state of both minima
    levels = [e for m in (inputs.s1, inputs.t1) for e in m.energy.values()]
    reach = max(levels) - min(levels) + abs(de_st)
    flags = []
    out = {'method': inputs.method, 'model': asdict(model),
           'temperature_K': float(temperature),
           'coupling_from': s_cm.method,
           'dE_ST_adiabatic_eV': de_st * HARTREE_TO_EV,
           'zero_point_shift_eV': zpe * HARTREE_TO_EV,
           'duschinsky_orthogonality': dev,
           'duschinsky_guard': guard,
           'soft_modes': None if soft is None else soft.record(),
           'hessian_sources': [sg.source, se.source],
           'huang_rhys_sum': {
               route: float((se.omega * ks[route] ** 2 / 2.0).sum())
               for route in HUANG_RHYS_ROUTES if route in ks},
           'gradient_from': ks.get('gradient_from'),
           'huang_rhys_sum_t1_modes': float(s_g.sum())}
    for name, initial, de, direction, average, at, modes, lam_key in (
            ('ISC', inputs.s1, -de_st, 'emission', 1, s_cm, se, 'T1@S1'),
            ('RISC', inputs.t1, de_st, 'absorption', TRIPLET_SUBLEVELS,
             t_cm, sg, 'S1@T1')):
        lam_solv = (solvent_reorganization(inputs.solvent[lam_key])
                    if inputs.solvent else 0.0 if inputs.gas else None)
        if lam_solv is None and model.classical_lambda == 'computed':
            raise RateRefused(f'minima in a medium ({inputs.s1.solvation}) '
                              f'with no solvent energies: no computed '
                              f'reorganization; supply {SOLVENT_KEYS} or '
                              f'assume a lambda')
        lam = (lam_solv if model.classical_lambda == 'computed'
               else float(model.classical_lambda) / HARTREE_TO_EV)
        # ZPE_final - ZPE_initial: G's phase carries the quantum modes', the
        # classical block's is a shift of its own
        zpe_b = zpe if soft is None else bath_model.zero_point_shift()
        zpe_dir = -zpe_b if name == 'ISC' else zpe_b
        lam_soft, zpe_block = ((0.0, 0.0) if soft is None else
                               (soft.lam(direction),
                                soft.zero_point_shift(direction)))
        bath = _Bath(model, bath_model, s_g, sg.omega, direction,
                     lam + lam_soft, zpe_dir,
                     temperature, reach, flags, zpe_block=zpe_block)
        k, k_c, k_h, rec, mixing = _pair_rate(
            at, de, average, model, modes.omega, modes.modes, modes.masses,
            rot[at.where], bath)
        levels = _manifold_minimum(initial, at, model, mixing)
        first, second, e_first, e_second = (
            (initial.emitter, initial.reservoir, e_s1, e_t1) if name == 'ISC'
            else (initial.reservoir, initial.emitter, e_t1, e_s1))
        man = rates.manifold_crossings(
            levels, first, second, e_first, e_second, bath, temperature,
            average, k, rates.ManifoldGaps('stacked', None))
        out[name] = {'dE_eV': de * HARTREE_TO_EV,
                     'lambda_solvent_eV': (None if lam_solv is None
                                           else lam_solv * HARTREE_TO_EV),
                     'lambda_classical_eV': lam * HARTREE_TO_EV,
                     'lambda_soft_block_eV': lam_soft * HARTREE_TO_EV,
                     'zero_point_used_eV': ((zpe_dir + zpe_block)
                                            * HARTREE_TO_EV),
                     'k_pair': k, 'k_condon': k_c, 'k_ht': k_h, **rec,
                     'k_eff': man['k_eff'], 'manifold': man}
    out.update(_kinetics(inputs.s1, out['ISC']['k_eff'],
                         out['RISC']['k_eff']))
    out['numerical_flags'] = list(flags)
    return out


def _accepting_lambda(initial, final, label):
    """E_label at the initial minimum less at its own (the final one),
    Hartree; None where the initial minimum has no energy of that state."""
    if label not in initial.energy:
        return None
    return float(initial.energy[label]) - float(final.energy[label])


def _guard(inputs, dmodel, temperature, soft):
    """{'ISC': record, 'RISC': record} of `duschinsky_guard` on the S1 <-> T1
    model (with the classical block `soft`'s reorganization in each
    direction, or none), each against the accepting state's energy between
    the method's two minima; a refused direction raises `RateRefused`."""
    s1, t1 = inputs.s1, inputs.t1

    def block(direction):
        return 0.0 if soft is None else soft.lam(direction)
    out = {'ISC': duschinsky_guard(
               dmodel, temperature, 'emission',
               _accepting_lambda(s1, t1, t1.reservoir), block('emission')),
           'RISC': duschinsky_guard(
               dmodel, temperature, 'absorption',
               _accepting_lambda(t1, s1, s1.emitter), block('absorption'))}
    for name, rec in out.items():
        if rec['status'] == 'refused':
            raise RateRefused(f'the Duschinsky model does not describe '
                              f'{name}: {guard_message(rec)}')
    return out


def _manifold_minimum(initial, at, model, mixing):
    """The `rates.Minimum` the manifold reads: the method's own energies at
    the initial minimum (only the pair's two states for triplets='T1') and
    the coupling minimum's spin-orbit table, its squared elements mixed
    between each crossing state and its diabatic path state by the weights
    `mixing` {path state: s} (none: as given). The mixing moves coupling
    between the two, never creates it: each row and column sum of squares
    over the pair is kept."""
    energy = dict(initial.energy)
    if model.triplets == 'T1':
        energy = {lab: energy[lab] for lab in (initial.emitter,
                                               initial.reservoir)}
    socme = np.asarray(at.socme, float)
    if mixing:
        v2 = socme ** 2
        for state, s in mixing.items():
            mate = at.reservoir if state[0] == 'T' else at.emitter
            a, b = key_of(mate)[1], key_of(state)[1]
            # a triplet path mixes two columns, a singlet path two rows
            view = v2 if state[0] == 'T' else v2.T
            if state[0] == 'S':
                a, b = a + 1, b + 1
            x, y = view[:, a].copy(), view[:, b].copy()
            view[:, a] = (1.0 - s) * x + s * y
            view[:, b] = s * x + (1.0 - s) * y
        socme = np.sqrt(v2)
    return rates.Minimum(emitter=initial.emitter, reservoir=initial.reservoir,
                         energy=energy, socme=socme)


def _kinetics(s1, k_isc, k_risc):
    """k_r at the S1 minimum and the photoluminescence kinetics on the
    effective ISC and RISC, where the minimum carries a transition dipole;
    None otherwise."""
    if s1.dipole is None:
        return {'k_r': None, 'pl': None}
    root = key_of(s1.emitter)[1]
    k_r = rates.radiative_rate(s1.omega[s1.emitter],
                               np.asarray(s1.dipole)[root])
    return {'k_r': k_r, 'pl': rates.photoluminescence(k_r, k_isc, k_risc)}


def ablation_ladder(inputs, full, temperature, shared):
    """[{'step', 'model', 'table' or 'refused'}] down `ABLATION_LADDER`
    from the model `full`, each rung removing one more ingredient; the
    assumed-lambda rungs both stand on the 'mlj' one, and the quantum soft
    mode and rectilinear Duschinsky rungs on 't1_only' (the next rung keeps
    the classical block and the curvilinear J). `shared` is the
    reference `RateInputs` of the shared-coupling rung (the method's own
    for the reference method itself). A rung the inputs cannot carry, or
    whose switches do not make a `RateModel` (the 'mlj' rung of a model with
    an inhomogeneous width, and the rungs on it), is recorded as refused with
    the reason, and the ladder goes on."""
    out, base = [], asdict(full)
    for step, change in ABLATION_LADDER:
        fields = {**base, **change}
        if step not in SIDE_STEPS:
            base = fields
        try:
            model = RateModel(**fields)
        except ValueError as why:
            out.append({'step': step, 'model': fields,
                        'refused': f'not a rate model: {why}'})
            continue
        try:
            out.append({'step': step, 'model': fields,
                        'table': rate_chain(inputs, model, temperature,
                                            shared=shared)})
        except RateRefused as why:
            out.append({'step': step, 'model': fields, 'refused': str(why)})
    return out


__all__ = ['RateModel', 'RateInputs', 'ABLATION_LADDER', 'SIDE_STEPS',
           'FCDensityAtRoundoff', 'solvent_reorganization', 'rate_chain',
           'ablation_ladder']
