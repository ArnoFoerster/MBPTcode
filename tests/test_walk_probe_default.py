"""The (A - B) probe after every matrix-free full-BSE solve of a chain is on
by default and off only by name.

`BSENumerics.probe` (True: the converged Lanczos from 1/d) reaches every
chain a `SurfaceSpec` builds, and so every walk of the photophysics stages
under either rate model; a non-positive min eig(A - B) refuses the step
(`UnstableReference`, a StepFailed). `probe=False` in a spec's numerics, and
--no-probe on the photophysics command line, switch it off; the default
leaves every record's stamp as it was (`chain_spec`, `excited_numerics`).
"""
import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.constants import (SCF_DIFFERENTIABLE_CONV_TOL,
                                SCF_DIFFERENTIABLE_GRAD_TOL)
from src.Base.numerics import Numerics
from src.Base.declaration import QPStates
from src.SingleReference.LinearResponse import davidson
from src.SingleReference.LinearResponse.davidson import UnstableReference
from src.gradients.excited_state import ExcitedStateChain
from src.properties.surfaces import EXCITED_NUMERICS, excited_numerics
from src.photophysics.chain import chain_spec
from src.photophysics.cli import parser
from tests.isdf_test_grid import g1_numerics
from tests.photophysics_cases import declaration
H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'


@pytest.fixture(scope='module')
def water():
    """Water/cc-pVDZ Hartree-Fock."""
    mol = gto.M(atom=H2O, basis='cc-pvdz', verbose=0)
    return mol, scf_factory(mol)


def scf_factory(mol):
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = SCF_DIFFERENTIABLE_CONV_TOL
    mf.conv_tol_grad = SCF_DIFFERENTIABLE_GRAD_TOL
    mf.kernel()
    assert mf.converged
    return mf


def test_the_probe_is_a_default_of_every_chain():
    """On by default for every chain (and so every SurfaceSpec), a numeric a
    spec may name, recorded only where it is not the default."""
    assert Numerics().bse.probe is True
    assert 'probe' in EXCITED_NUMERICS
    spec = chain_spec(declaration(qp_states=QPStates()), None)
    assert 'probe' not in dict(spec.numerics)
    off = chain_spec(declaration('--no-probe', qp_states=QPStates()), None)
    assert dict(off.numerics)['probe'] is False


def _negative_probe(*args, **kwargs):
    return np.array([-0.01])


def test_a_chain_step_is_refused_by_the_probe(water, monkeypatch):
    """The chain's matrix-free full-BSE solve runs the probe and refuses a
    non-positive min eig(A - B) (`UnstableReference`, a StepFailed);
    probe=False skips it. The probe here is stubbed to a negative value: the
    probe's own numbers are gated in test_probe_after_davidson."""
    mol, mf = water
    real = ExcitedStateChain(mol, scf_factory, mf=mf,
                             numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'nroots': 4})))
    omega = real.excitation()
    assert omega > 0
    assert 'probe' not in excited_numerics(real)
    monkeypatch.setattr(davidson, '_lowest_amb_from_action', _negative_probe)
    chain = ExcitedStateChain(mol, scf_factory, mf=mf,
                              numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'nroots': 4})))
    with pytest.raises(UnstableReference):
        chain.excitation()
    off = ExcitedStateChain(mol, scf_factory, mf=mf,
                            numerics=g1_numerics(mol, Numerics(bse={'solver': 'davidson', 'nroots': 4, 'probe': False})))
    assert off.excitation() == pytest.approx(omega, abs=1e-10)
    assert excited_numerics(off)['probe'] is False


@pytest.mark.parametrize('model', ['tadf', 'uisc'])
def test_the_command_line_switches_the_probe_off_only_by_name(model):
    """The photophysics command line carries --no-probe for either rate
    model, and leaves the probe on without it."""
    assert '--no-probe' in parser().format_help()
    on = declaration('--rate-model', model)
    assert on.numerics.bse.probe is True
    off = declaration('--rate-model', model, '--no-probe')
    assert off.numerics.bse.probe is False
