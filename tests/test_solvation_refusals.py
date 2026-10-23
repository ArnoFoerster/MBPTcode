"""Refusals and numerical flags of the state-specific solvation loop, its
state models, its medium and its driver (refuse inputs, warn numerics).
Water 6-31G
(bent, non-symmetric), toluene IEF-PCM, frozen 1, cc-pvdz-ri / cc-pvdz-jkfit.
Each refusal test raises where the code used to return a number;
each flag test shows the warning and the named flag where the code refused before.

  - a loop at max_cycle warns and returns its last cycle, flagged
      'solvation_loop_not_converged', its G the functional there; the driver
      keeps a walk that did not converge, warned about and flagged
      'walk_not_converged', and a rerun walks it again
  - an unconverged ADC(2) root is warned about and flagged
      'adc_roots_not_converged', in the call and in the force
  - the driver refuses a record made with other settings and builds
      ADC(2) on Hartree-Fock (and refuses --refreeze
      with ADC(2), which freezes nothing); a GW-BSE minimum whose refreeze
      drift stays above REFREEZE_TOL_MEV is kept, warned about and flagged
      (refreeze_not_converged), its G taken on the last rebuilt surface; the
      walk-versus-vertical bar of a refrozen walk is REFREEZE_TOL_MEV
  - overlap following flags a lost root and two mixed roots
      ('root_following_lost', 'root_following_mixed'), and its metric does not
      depend on the orbitals' phases
  - the loop's G is the functional E(q) - (1/2) q^T Q^-1 q at the charges
      it returns, also away from convergence
  - a Kohn-Sham reference gets its own gradient class (the loop's force is
      pyscf's PCM-RKS gradient); MP2 on it is refused
  - the PCM force refuses a non-symmetric density
  - the PCM force refuses a continuum around an enclosing shell
  - the loop refuses a state whose LR continuum is not its field's
  - a state model refuses an SCF whose orbital gradient is not converged
"""
import json
import os
import sys
import warnings
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft

from src.environments import solvation_loop as sl
from src.Base.numerics.numerics import Numerics
from src.environments.reaction_field import ContinuumField, NoMedium
from src.environments.solvent_screening import SolventScreening
from src.gradients.state_models import HFState, MP2State
from src.SingleReference.ADC.eeADC import ee_fold
from src.gradients.adc import ee_state_model
from src.SingleReference.ADC.eeADC.ee_follow import (FollowedRoot, FollowingWarning,
                                                     transition_amplitudes)
from src.gradients.adc.ee_state_model import ADC2State
from src.photophysics import solvated
from tests.adc_common import (ATOMS, AUX, NF, ROOT_CONV_TOL, SOLVATION_TIGHT,
                              SOLVATION_TIGHT_LOOP, df_rhf, make_scf_loop, molecule)


def toluene(mol):
    return ContinuumField.from_environment(SolventScreening(mol, solvent='toluene'))


def adc(mol, spin='singlet', nroots=3, root=0, follow='overlap', lr=None):
    mp2 = MP2State(HFState(mol, make_scf_loop), NF, AUX, force_route='dense')
    return ADC2State(mp2, spin, root, nroots, lr, conv_tol=ROOT_CONV_TOL, follow=follow)


def driver():
    """The solvated driver (`python -m src.photophysics solvated`)."""
    return solvated


# ------------------------------------------------------------------------------

def test_loop_at_max_cycle_warns_and_flags():
    mol = molecule(ATOMS)
    field, state = toluene(mol), HFState(mol, make_scf_loop)
    with pytest.warns(sl.SolvationNotConverged, match='did not converge'):
        r = sl.equilibrate(state, field, numerics=Numerics(
            solvent=dict(SOLVATION_TIGHT_LOOP, loop_max_cycle=2)))
    assert not r.converged and len(r.history) == 2
    assert [f['flag'] for f in r.flags] == ['solvation_loop_not_converged']
    # G is the functional at the charges the state was last solved in
    assert abs(r.energy - sl.free_energy(state, field, r.q)) < 1e-10
    with pytest.warns(sl.SolvationNotConverged, match='force of G_eq'):
        F = sl.force(state, field, r)
    assert F.shape == (mol.natm, 3) and np.isfinite(F).all()


def test_driver_keeps_an_unconverged_walk_flagged(tmp_path, monkeypatch):
    mod = driver()
    out = str(tmp_path / 'rec.json')
    walk = dict(converged=False, cycles=3, opt_grad_max=1e-2, geometry=[], G=-76.0,
                flags=['walk_not_converged'])
    monkeypatch.setattr(mod, '_minimum', lambda *a, **k: (None, dict(walk)))
    monkeypatch.setattr(mod, 'SolvatedSurface', lambda *a, **k: None)
    rec = {}
    R = rec.setdefault('gas', {})
    args = SimpleNamespace(out=out)
    # the walk itself warns (`_minimum`, excitations.relaxation_flags); the
    # store keeps the entry and its flag
    mod._ground_state(args, rec, R, 'gas', None, None, lambda kind: None, {})
    saved = json.load(open(out))['gas']['S0']
    assert saved['converged'] is False and saved['flags'] == ['walk_not_converged']
    # a rerun walks it again: the entry and everything built on it go
    R['S1'] = dict(absorption_eV=4.0)
    mod._drop_unconverged(R)
    assert R == {}


# ------------------------------------------------------------------------------

def _unconverged(fn):
    def wrapped(*a, **k):
        out = fn(*a, **k)
        res = out[-1] if isinstance(out, tuple) else out
        res.converged = np.zeros(np.size(res.converged), bool)
        return out
    return wrapped


def _names(flags):
    return [f['flag'] for f in flags]


def test_unconverged_root_flagged_in_the_call(monkeypatch):
    mol = molecule(ATOMS)
    state = adc(mol, follow='index')
    monkeypatch.setattr(ee_state_model, 'solve_ee_adc',
                        _unconverged(ee_state_model.solve_ee_adc))
    with pytest.warns(FollowingWarning, match='not converged'):
        r = state(np.zeros((mol.nao, mol.nao)), 0.0)
    assert 'adc_roots_not_converged' in _names(r.info['flags'])
    assert r.info['converged'] is False


def test_unconverged_root_flagged_in_the_force(monkeypatch):
    mol = molecule(ATOMS)
    state = adc(mol, follow='index')
    monkeypatch.setattr(ee_fold, 'solve_folded', _unconverged(ee_fold.solve_folded))
    with pytest.warns(FollowingWarning, match='not converged'):
        F, info = state.force(np.zeros((mol.nao, mol.nao)), 0.0)
    assert 'adc_roots_not_converged' in _names(info['flags'])
    assert np.isfinite(F).all()


# ------------------------------------------------------------------------------

def _args(tmp_path, **kw):
    xyz = tmp_path / 'w.xyz'
    xyz.write_text('3\n\nO 0 0 0\nH 0 0.76 0.59\nH 0 -0.76 0.59\n')
    a = dict(xyz=str(xyz), basis='6-31g', auxbasis='cc-pvdz-ri',
             jk_auxbasis='cc-pvdz-jkfit', solvent='toluene', frozen=1, nroots=3,
             follow='index', saddle_check=True, tol_q=1e-9, tol_energy=1e-11,
             method='adc2', xc=None, isdf_grid=None, bse_solver=None, bse_kernel=None,
             refreeze=None, maxiter=100, out=str(tmp_path / 'rec.json'))
    a.update(kw)
    return SimpleNamespace(**a)


def test_driver_refuses_a_record_of_other_settings(tmp_path):
    mod = driver()
    first = _args(tmp_path)
    mod._save(mod._start_record(first), first.out)
    assert mod._start_record(first)['basis'] == '6-31g'      # same settings resume
    for change in (dict(basis='cc-pvdz'), dict(follow='overlap'), dict(tol_q=1e-8),
                   dict(nroots=5), dict(method='gwbse', isdf_grid='G1'),
                   dict(method='gwbse', isdf_grid='G1', bse_kernel='full'),
                   dict(method='gwbse', isdf_grid='G1', refreeze=False)):
        with pytest.raises(SystemExit, match='other settings'):
            mod._start_record(_args(tmp_path, **change))


@pytest.mark.parametrize('change', [dict(isdf_grid='G1'), dict(xc='pbe0'),
                                    dict(bse_solver='dense'),
                                    dict(bse_kernel='full'),
                                    dict(refreeze=True), dict(refreeze=False),
                                    dict(method='gwbse', isdf_grid='G1', nroots=5),
                                    dict(method='gwbse', isdf_grid='G1', auxbasis='x'),
                                    dict(method='gwbse', isdf_grid='G1', frozen=1),
                                    dict(method='gwbse')])
def test_driver_refuses_the_other_methods_options(tmp_path, change):
    """An option the method does not read is refused before any work, and
    GW-BSE without an ISDF grid."""
    args = _args(tmp_path, **dict(dict(auxbasis=None, nroots=None, frozen=None), **change))
    with pytest.raises(SystemExit, match='takes no|needs --isdf-grid'):
        driver()._check_args(args, None)


#: The GW-BSE run of `_args` (the ADC(2) options unset).
GWBSE_ARGS = dict(method='gwbse', isdf_grid='G1', auxbasis=None, nroots=None, frozen=None)


class _Surface:
    """A stand-in surface: its label, a constant G and no flags logged."""

    def __init__(self, G):
        self.G = G
        self.flags = []

    def flags_at(self, mol):
        return []

    def label(self):
        return 'T1 stand-in'

    def total_energy(self, mol):
        return self.G


def test_driver_keeps_an_unconverged_refreeze_flagged(tmp_path, monkeypatch):
    """relax returning a refreeze that did not converge (properties.optimize
    warns, RefreezeNotConverged): the minimum is the last pass's, flagged
    refreeze_not_converged; its G is the last rebuilt surface's; the passes
    are recorded."""
    mod = driver()
    passes = [{'pass': k, 'drift_meV': 4.0, 'shift': 1e-3, 'converged': True}
              for k in (1, 2)]
    info = dict(converged=True, cycles=2, opt_grad_max=1e-5, refreeze='not converged',
                refreeze_drift_meV=4.0, refreeze_shift=1e-3, refreeze_converged=False,
                refreeze_denergy=-1e-4, refreeze_passes=passes,
                final_surface=_Surface(-2.0))

    def relax(surf, at, **kw):
        assert kw['refreeze'] == 1
        return 'R*', info
    monkeypatch.setattr(mod, 'relax', relax)
    monkeypatch.setattr(mod, '_geom', lambda m: m)
    args = _args(tmp_path, **GWBSE_ARGS, saddle_check=False)
    mod._check_args(args, None)
    m, out = mod._minimum(args, _Surface(-1.0), 'R0', 0.0, 'triplet')
    assert m == 'R*' and out['geometry'] == 'R*' and out['G'] == -2.0
    assert out['flags'] == ['refreeze_not_converged'] and out['refreeze'] == 'not converged'
    assert out['refreeze_drift_meV'] == 4.0 and len(out['refreeze_passes']) == 2
    assert out['refreeze_denergy_meV'] == -1e-4 * mod.HARTREE_TO_MEV


@pytest.mark.parametrize('method,kind,asked', [('gwbse', 'S0', 0), ('adc2', 'triplet', 0),
                                               ('gwbse', 'triplet', 1)])
def test_driver_refreezes_only_a_surface_that_freezes(tmp_path, monkeypatch, method,
                                                      kind, asked):
    """Only a GW-BSE excited state's walk runs the refreeze loop; ADC(2) and
    the GW-BSE reference record 'nothing frozen'."""
    mod = driver()
    seen = []

    def relax(surf, at, **kw):
        seen.append(kw['refreeze'])
        return at, dict(converged=True, cycles=1, opt_grad_max=0.0,
                        refreeze='measured' if kw['refreeze'] else 'not measured')
    monkeypatch.setattr(mod, 'relax', relax)
    monkeypatch.setattr(mod, '_geom', lambda m: m)
    args = _args(tmp_path, **(GWBSE_ARGS if method == 'gwbse' else {}),
                 saddle_check=False)
    mod._check_args(args, None)
    out = mod._minimum(args, _Surface(-1.0), 'R0', 0.0, kind)[1]
    assert seen == [asked]
    assert out['refreeze'] == ('measured' if asked else 'nothing frozen')


@pytest.mark.parametrize('refreeze,gap,flagged', [('measured', 1e-5, False),
                                                  ('measured', 2e-4, True),
                                                  ('nothing frozen', 1e-8, True)])
def test_walk_vs_vertical_bar(tmp_path, monkeypatch, refreeze, gap, flagged):
    """The walk-versus-vertical gap of a refrozen walk is flagged above
    REFREEZE_TOL_MEV (3 meV = 1.1e-4 Ha), of one that was not above ten loop
    tolerances."""
    mod = driver()
    monkeypatch.setattr(mod, '_vertical', lambda *a: dict(
        emission=0.1, G={'eq_state': -1.0}, flags={}))
    monkeypatch.setattr(mod, '_mol_of', lambda g, b: None)
    args = _args(tmp_path)
    R = {'S0': dict(G=-1.2, converged=True)}
    X = {'minimum': dict(G=-1.0 + gap, geometry=[], refreeze=refreeze, converged=True)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mod._emission(args, {}, R, X, 'triplet', None, None, {})
    assert X['walk_vs_vertical_flag'] is flagged
    assert X['walk_vs_vertical_G_Ha'] == pytest.approx(gap, abs=1e-15)
    assert any('differ by' in str(w.message) for w in caught) is flagged


def test_driver_builds_adc2_on_hartree_fock(tmp_path):
    """--method adc2 leaves --xc unset (None): its states stand on RHF, which
    MP2State requires (a DFRKS reference is refused)."""
    mod = driver()
    args = _args(tmp_path)
    mod._check_args(args, None)
    loop_kw = dict(numerics=Numerics(solvent=SOLVATION_TIGHT_LOOP))
    _, make_state = mod._factories(args, 'gas', loop_kw)
    mol = molecule(ATOMS)
    s0 = make_state('S0')(mol, NoMedium(mol))
    assert not isinstance(s0.hf.mf, dft.rks.KohnShamDFT)


@pytest.mark.parametrize('kind', ['hf', 'mp2', 'adc2'])
def test_a_surface_that_freezes_nothing_is_its_own_refreeze(kind):
    """HF, MP2 and ADC(2) (overlap following) in toluene carry no
    surface_key: SolvatedSurface.refreeze returns the surface itself, so
    their walks are what they were."""
    from src.gradients.solvated_surface import SolvatedSurface
    mol = molecule(ATOMS)

    def make_state(m, field):
        if kind == 'adc2':
            return adc(m)
        hf = HFState(m, make_scf_loop)
        return hf if kind == 'hf' else MP2State(hf, NF, AUX, force_route='dense')
    surf = SolvatedSurface(mol, toluene, make_state, label=kind,
                           numerics=Numerics(solvent=SOLVATION_TIGHT_LOOP))
    surf.total_energy(mol)
    assert surf.refreeze(molecule(ATOMS)) is surf


# ------------------------------------------------------------------------------

def test_a_lost_root_is_flagged():
    """The vector of root 5 (of 6) carried into a solve of 3 roots: it is not
    among them; the best overlap is taken, warned about and flagged."""
    mol = molecule(ATOMS)
    h = np.zeros((mol.nao, mol.nao))
    high = adc(mol, nroots=6, root=5)
    high(h, 0.0)
    low = adc(mol, nroots=3)
    low.y_follow = high.y_follow.copy()
    with pytest.warns(FollowingWarning, match='not be among'):
        r = low(h, 0.0)
    assert 'root_following_lost' in _names(r.info['flags'])


def test_two_mixed_roots_are_flagged():
    mol = molecule(ATOMS)
    h = np.zeros((mol.nao, mol.nao))
    r0, r1 = adc(mol, root=0), adc(mol, root=1)
    r0(h, 0.0)
    r1(h, 0.0)
    mixed = adc(mol)
    mixed.y_follow = FollowedRoot(mol, r0.y_follow.T + r1.y_follow.T)
    with pytest.warns(FollowingWarning, match='mixed'):
        r = mixed(h, 0.0)
    assert 'root_following_mixed' in _names(r.info['flags'])


def test_following_metric_is_gauge_free_and_follows_a_step():
    """Flipping the sign of every second orbital changes the MO singles vector
    of a root (its dot product with the old one drops below 1) and leaves its
    AO transition amplitude, hence every weight, unchanged; across a 0.02 Bohr
    step root 1 is found again with weight above 0.9."""
    mol = molecule(ATOMS)
    s = adc(mol, root=1)
    s(np.zeros((mol.nao, mol.nao)), 0.0)
    mf, ref = s.mp2.mf, s.y_follow
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    sign = np.where(np.arange(mf.mo_coeff.shape[1]) % 2, -1.0, 1.0)
    y = ref.in_orbitals(mf, NF)
    y_flipped = (sign[NF:nocc, None] * y.reshape(nocc - NF, -1) * sign[None, nocc:]).ravel()
    flipped = SimpleNamespace(mo_coeff=mf.mo_coeff * sign, mo_occ=mf.mo_occ)
    assert abs(y @ y_flipped) < 0.99 * (y @ y)
    T = transition_amplitudes(mf, NF, y[:, None])
    T_flipped = transition_amplitudes(flipped, NF, y_flipped[:, None])
    assert np.allclose(T, T_flipped, atol=1e-14, rtol=0)
    assert abs(ref.weights(mol, T_flipped)[0] - 1.0) < 1e-10
    step = molecule(ATOMS, 0.02 * np.ones((mol.natm, 3)) / np.sqrt(3 * mol.natm))
    moved = adc(step)
    moved.y_follow = ref.copy()
    assert moved(np.zeros((step.nao, step.nao)), 0.0).info['index'] == 1
    assert ref.weights(step, moved.y_follow.T[None])[0] > 0.9


# ------------------------------------------------------------------------------

def test_loop_reports_the_functional_away_from_convergence():
    """Stopped early (tol_q 1e-2), the reported G is G(q) = E(q) - (1/2)
    q^T Q^-1 q at the returned charges, to 1e-10 Ha; E(q) - (1/2) q . V(D(q))
    differs there at first order."""
    mol = molecule(ATOMS)
    field = toluene(mol)
    state = HFState(mol, make_scf_loop)
    r = sl.equilibrate(state, field, numerics=Numerics(
        solvent={'loop_tol_q': 1e-2, 'loop_tol_energy': 1e-2}))
    assert r.converged and len(r.history) <= 3
    G = sl.free_energy(state, field, r.q)
    assert abs(r.energy - G) < 1e-10, f'{r.energy - G:+.1e}'


# ------------------------------------------------------------------------------

def make_rks(mol):
    mf = dft.RKS(mol, xc='pbe0')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle, mf.verbose = 1e-12, 1e-10, 200, 0
    return mf


def test_kohn_sham_reference_force_in_a_continuum():
    """The equilibrium PBE0 loop is pyscf's PCM-RKS (energy to 1e-9 Ha) and its
    force pyscf's PCM-RKS gradient (to 1e-7 Ha/Bohr)."""
    mol = molecule(ATOMS)
    field = toluene(mol)
    state = HFState(mol, make_rks)
    r = sl.equilibrate(state, field, **SOLVATION_TIGHT)
    F = sl.force(state, field, r)
    env = SolventScreening(mol, solvent='toluene')
    mf = env.mean_field(mol, lambda m: make_rks(m).run())
    assert abs(r.energy - mf.e_tot) < 1e-9, f'{r.energy - mf.e_tot:+.1e}'
    g = mf.nuc_grad_method()
    g.verbose = 0
    d = float(np.max(np.abs(F - g.kernel())))
    assert d < 1e-7, f'max |dF| {d:.1e} Ha/Bohr'


def test_mp2_on_a_kohn_sham_reference_is_refused():
    mol = molecule(ATOMS)
    with pytest.raises(ValueError, match='Hartree-Fock reference'):
        MP2State(HFState(mol, make_rks), NF, AUX)


# ------------------------------------------------------------------------------

def test_pcm_force_refuses_a_non_symmetric_density():
    mol = molecule(ATOMS)
    field = toluene(mol)
    t = np.random.default_rng(7).standard_normal((mol.nao, mol.nao))
    with pytest.raises(ValueError, match='symmetric'):
        field.transition_force(t)
    with pytest.raises(ValueError, match='symmetric'):
        field.force(t)
    field.transition_force(0.5 * (t + t.T))      # the symmetrized one is taken


def test_pcm_force_refuses_an_enclosing_shell():
    """A continuum whose sites do not ride the atoms (from_environment of a
    SolventScreening with cavity_atoms passes its differentiable=False; built
    directly here, since pyscf 2.9+ cannot build the shell's PCM at all)."""
    mol = molecule(ATOMS)
    field = ContinuumField(mol, 2.38, 2.24, differentiable=False)
    dm = np.eye(mol.nao)
    with pytest.raises(ValueError, match='not differentiable'):
        field.force(dm)
    with pytest.raises(ValueError, match='not differentiable'):
        field.transition_force(dm)


# ------------------------------------------------------------------------------

class _WithLR(HFState):
    """Hartree-Fock declaring an LR continuum (the attribute the loop reads)."""

    def __init__(self, mol, make_scf, lr_environment):
        super().__init__(mol, make_scf)
        self.lr_environment = lr_environment


def test_loop_refuses_a_foreign_lr_continuum():
    mol = molecule(ATOMS)
    field = toluene(mol)
    water = SolventScreening(mol, solvent='water')
    with pytest.raises(ValueError, match='transition_environment'):
        sl.equilibrate(_WithLR(mol, make_scf_loop, water), field)
    with pytest.raises(ValueError, match='transition_environment'):
        sl.equilibrate(_WithLR(mol, make_scf_loop, field.transition_environment()),
                       NoMedium(mol))
    r = sl.equilibrate(_WithLR(mol, make_scf_loop, field.transition_environment()),
                       field)                      # the field's own is taken
    assert r.converged


# ------------------------------------------------------------------------------

def test_state_model_refuses_an_unconverged_orbital_gradient():
    mol = molecule(ATOMS)
    for grad in (None, 1e-6):
        with pytest.raises(ValueError, match='conv_tol_grad'):
            HFState(mol, lambda m: df_rhf(m, 1e-12, grad))


def test_a_refrozen_surface_shares_the_walks_flag_log():
    """SolvatedSurface.refreeze builds a new surface on a model that freezes
    conventions; its flags go into the same log, so a minimum's flags_at
    reads the walk's and the refrozen walk's evaluations alike."""
    from src.gradients.solvated_surface import SolvatedSurface
    mol = molecule(ATOMS)
    surf = SolvatedSurface(mol, toluene, lambda m, f: None, label='stand-in')
    state = SimpleNamespace(surface_key=object(), lr_environment=None,
                            make_scf=make_scf_loop)
    surf.last = {'state': state, 'mol': mol}
    surf.flags.append(dict(coords=mol.atom_coords().tolist(),
                           flags=['solvation_loop_not_converged'], messages=['planted']))
    again = surf.refreeze(mol)
    assert again is not surf and again.flags is surf.flags
    assert again.flags_at(mol) == ['solvation_loop_not_converged']


def test_driver_walk_warns_through_the_one_walk_policy(tmp_path, monkeypatch):
    """An unconverged walk is warned about by the policy every walk of the
    code follows (`excitations.relaxation_flags`, `WalkNotRelaxed`), and the
    saddle check is not taken at its last geometry."""
    from src.properties.excitations import WalkNotRelaxed
    mod = driver()

    def relax(surf, at, **kw):
        return at, dict(converged=False, cycles=1, opt_grad_max=1e-2,
                        status='trust radius collapsed')

    def no_check(*a, **k):
        raise AssertionError('a walk that did not relax is not checked')
    monkeypatch.setattr(mod, 'relax', relax)
    monkeypatch.setattr(mod, 'saddle_check', no_check)
    monkeypatch.setattr(mod, '_geom', lambda m: m)
    args = _args(tmp_path, **GWBSE_ARGS, saddle_check=True)
    mod._check_args(args, None)
    with pytest.warns(WalkNotRelaxed, match='did not relax'):
        out = mod._minimum(args, _Surface(-1.0), 'R0', 0.0, 'triplet')[1]
    assert out['flags'] == ['walk_not_converged'] and out['converged'] is False
