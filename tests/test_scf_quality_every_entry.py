"""Every force and coupling entry warns about, and flags, a loosely converged
mean field, and carries on.

The gradient Lagrangian assumes the occupied-virtual Fock block vanishes; what
an SCF leaves there is amplified in the force. The interstate numerator, the
quasiparticle force, the couplings read off a shared evaluation and
`StateManifold.evaluate` asked for couplings once took such a mean field
silently, then refused it; under the rule (refuse
inputs, warn numerics) each warns, flags it in the chain's `scf_flags`
(which the vertical record carries) or the coupling's diagnostics, and goes
on. Each is handed here a mean field converged to conv_tol_grad 1e-4, far
above `scf_force_tol`. The relaxed excitation density (the force's
Lagrangian, which a solvation loop reads each cycle), the equilibrium-shift
gradient and the dRPA correlation gradient do the same. A mean field
that was never run is an input, and is refused.
"""
import pytest
from pyscf import gto, scf

from src.Base.numerics.numerics import Numerics
from src.gradients.gwbse.derivative_coupling import (analytic_coupling,
                                                     analytic_ground_coupling)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from src.gradients.rpa.ground_state import RPAGroundStateChain
from src.gradients.reference_mean_field import check_scf_quality, scf_force_tol
from src.gradients.gwbse.state_manifold import StateEvaluation, StateManifold
from src.properties.spin_vibronic import evaluated_coupling
from tests.isdf_test_grid import g1_numerics

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

AUXBASIS = 'cc-pvdz-ri'
WATER = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: the loose SCF's orbital-gradient threshold, far above any force tolerance
LOOSE_GRAD = 1e-4
S0, S1 = ('singlet', 0), ('singlet', 1)


def water():
    return gto.M(atom=WATER, basis='cc-pvdz', verbose=0)


def fitted_rhf(mol, conv_tol_grad=1e-11):
    """Density-fitted Hartree-Fock converged to `conv_tol_grad`."""
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, conv_tol_grad, 200
    mf.kernel()
    return mf


@pytest.fixture(scope='module')
def case():
    """(chain, mol, loose mean field), the loose one's |F_ia| above tolerance."""
    mol = water()
    chain = ExcitedStateChain(mol, fitted_rhf, state=0,
                              numerics=g1_numerics(mol, Numerics(isdf={'auxbasis': AUXBASIS},
                                                                 bse={'nroots': 3})))
    loose = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    loose.conv_tol, loose.conv_tol_grad = 1e-3, LOOSE_GRAD
    loose.kernel()
    assert check_scf_quality(loose, 5) > scf_force_tol(loose)
    return chain, mol, loose


@pytest.fixture(scope='module')
def loose_evaluation(case):
    """A shared evaluation of the converged mean field, its mean field then
    swapped for the loose one: what a caller holding a stale evaluation has."""
    chain, mol, loose = case
    good = StateManifold(chain, states=(S0, S1)).evaluate(
        mol, couplings=((S0, S1),))
    ev = StateEvaluation(mol, loose)
    for name in ('energy', 'omega', 'root', 'target', 'interstate',
                 'spectrum'):
        setattr(ev, name, getattr(good, name))
    return ev


def flagged(chain, where):
    return [f for f in chain.scf_flags if f['where'] == where]


def past_the_check(*args, **kwargs):
    raise RuntimeError('past the SCF check')


def test_interstate_gradient_warns_and_flags(case, monkeypatch):
    chain, mol, loose = case
    monkeypatch.setattr(chain, 'forward', past_the_check)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        with pytest.raises(RuntimeError, match='past the SCF check'):
            chain.interstate_gradient(0, 1, mol=mol, mf=loose)
    assert flagged(chain, 'interstate_gradient')


def test_quasiparticle_gradient_warns_and_flags(case, monkeypatch):
    chain, mol, loose = case
    monkeypatch.setattr(chain, '_quasiparticle_solve', past_the_check)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        with pytest.raises(RuntimeError, match='past the SCF check'):
            chain.quasiparticle_gradient(0, mol=mol, mf=loose)
    assert flagged(chain, 'quasiparticle_gradient')


def test_manifold_couplings_warn_and_flag(case, monkeypatch):
    chain, mol, loose = case
    man = StateManifold(chain, states=(S0, S1))
    monkeypatch.setattr(man, 'forward', past_the_check)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        with pytest.raises(RuntimeError, match='past the SCF check'):
            man.evaluate(mol, loose, couplings=((S0, S1),))
    assert flagged(chain, 'StateManifold.evaluate')


def test_couplings_off_an_evaluation_warn_and_flag(loose_evaluation):
    ev = loose_evaluation
    with pytest.warns(RuntimeWarning, match='F_ia'):
        _, diags = analytic_coupling(None, S0, S1, evaluation=ev)
    flag, = diags['scf_flags']
    assert flag['residual'] > flag['tol']
    # flagged once on the evaluation, carried by every coupling read off it
    _, ground = analytic_ground_coupling(None, S0, evaluation=ev)
    _, again = evaluated_coupling(ev, S0, S1)
    assert ground['scf_flags'] == again['scf_flags'] == [flag]


def test_a_mean_field_never_run_is_refused(case):
    chain, mol, _ = case
    unrun = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    with pytest.raises(ValueError, match='never run'):
        chain.require_scf_quality(unrun, 'total_gradient')


def test_relaxed_density_warns_and_flags(case):
    chain, mol, loose = case
    with pytest.warns(RuntimeWarning, match='F_ia'):
        chain.excitation_density(mol=mol, mf=loose, relaxed=True)
    assert flagged(chain, 'excitation_density(relaxed=True)')


def test_equilibrium_shift_gradient_warns_and_flags(case, monkeypatch):
    chain, mol, loose = case

    def stop(*args):
        raise RuntimeError('past the SCF check')
    monkeypatch.setattr(chain, '_equilibrium_pieces', stop)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        with pytest.raises(RuntimeError, match='past the SCF check'):
            chain.equilibrium_shift_gradient(mol=mol, mf=loose)
    assert flagged(chain, 'equilibrium_shift_gradient')


def test_rpa_correlation_gradient_warns_and_flags(case, monkeypatch):
    _, mol, loose = case
    rpa = RPAGroundStateChain(mol, fitted_rhf, numerics=g1_numerics(
        mol, Numerics(isdf={'auxbasis': AUXBASIS})))

    def stop(*args, **kwargs):
        raise RuntimeError('past the SCF check')
    monkeypatch.setattr(rpa, '_forward', stop)
    with pytest.warns(RuntimeWarning, match='F_ia'):
        with pytest.raises(RuntimeError, match='past the SCF check'):
            rpa.correlation_gradient(mol=mol, mf=loose)
    assert flagged(rpa, 'correlation_gradient')
