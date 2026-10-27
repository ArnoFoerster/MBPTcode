"""The contour continuations in a polarizable continuum screen bare.

Duchemin, Guido, Jacquemin and Blase, Chem. Sci. 9, 4430 (2018) build Sigma_c
from the gas-phase W and put the whole reaction field in the static shift of
their Eq. (18), Delta eps_p = -/+ (1/2) <pp|W[v + vtilde] - W[v]|pp>, added
to every G0W0 root whole, after the solve. The Pade
space-time route, the excited-state chain and `calc_qp_energy(mode=
'space-time', continuation='cd'|'laplace'|'sop')` all do: Sigma_c, its
residues and the pole fit are screened with W[v], never W[v + vtilde], and no
static COHSEX operator is added. Water/cc-pVDZ in PCM water (density-fitted
reference on
cc-pvdz-ri, ISDF factors at the shipped radii), eV:

    reference  route              HOMO       LUMO      gap
    HF         pade            -10.48875    3.22897   13.71771
    HF         chain sop       -10.48873    3.22899   13.71771
    HF         cd              -10.48873    3.22899   13.71771
    PBE0       pade             -9.86490    3.19380   13.05871
    PBE0       chain sop        -9.86457    3.19368   13.05825
    PBE0       cd               -9.86488    3.19368   13.05856

'laplace' is 'cd' to the printed digits on every row and 'sop' within 0.06
meV of it. The gas-phase rows are the same with and without the continuum
machinery.

The static term every imaginary-axis route shares, `static_exchange_matrix`,
REFUSES an attached environment it is handed no Eq. (18) array for, rather
than add the COHSEX operator in its place.
"""
import numpy as np
import pytest
from pyscf import dft, gto, scf

from src.Base.numerics.numerics import Numerics
from src.Base.numerics.constants import HARTREE_TO_EV
from src.environments.solvent_screening import (SolventScreening,
                                                attach_solvent_screening,
                                                detach_solvent_screening)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.qp_solve import (static_exchange_diagonal,
                                             static_exchange_matrix,
                                             static_exchange_mean_field_matrix)
from src.gradients.gwbse.excited_state import ExcitedStateChain
from tests.isdf_test_grid import g1, g1_numerics

BASIS = 'cc-pvdz'
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'
CONTOURS = ('cd', 'laplace', 'sop')

#: How far a contour continuation may sit from the Pade route and from the
#: chain's pole model, in eV. Measured: 0.32 meV at most between the routes
#: in the gas phase (PBE0 HOMO, sop against pade) and 0.30 meV in the
#: continuum; 0.07 meV against the chain. Screening with W[v + vtilde] puts a
#: route 79 meV off on the HOMO and 348 to 385 meV off on the LUMO.
ROUTE_BAR_EV = 1e-3

_CASES = {}


def scf_factory(functional):
    """A density-fitted Hartree-Fock or PBE0 reference converged for a chain."""
    def build(mol):
        if functional == 'hf':
            mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
        else:
            mf = dft.RKS(mol).density_fit(auxbasis=BASIS + '-ri')
            mf.xc = 'pbe0'
            mf.grids.level = 5
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-11, 200
        mf.kernel()
        assert mf.converged
        return mf
    return build


def case(functional):
    """(mol, mf, {(medium, route): (HOMO, LUMO) in eV}) for one reference."""
    if functional in _CASES:
        return _CASES[functional]
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    factory = scf_factory(functional)
    mf = factory(mol)
    nocc = mol.nelectron // 2
    energies = {}
    for medium in ('gas', 'water'):
        if medium == 'water':
            attach_solvent_screening(mf, solvent='water')
        try:
            for route in ('pade',) + CONTOURS:
                energies[medium, route] = np.asarray(calc_qp_energy(
                    mf, state=[nocc - 1, nocc], mode='space-time',
                    continuation=route, counts=g1(mf)), float)
        finally:
            detach_solvent_screening(mf)
        env = SolventScreening(mol, solvent='water') if medium == 'water' \
            else None
        chain = ExcitedStateChain(mol, factory, mf=mf, environment=env,
                                  numerics=g1_numerics(mol, Numerics(qp={'residue_route': 'sop'})))
        energies[medium, 'chain'] = np.array(
            [chain.quasiparticle(0, mol, mf),
             chain.quasiparticle(1, mol, mf)]) * HARTREE_TO_EV
    for (medium, route), (homo, lumo) in sorted(energies.items()):
        print(f'  {functional:5s} {medium:6s} {route:8s} HOMO {homo:10.5f} '
              f'LUMO {lumo:10.5f} gap {lumo - homo:9.5f} eV')
    _CASES[functional] = (mol, mf, energies)
    return _CASES[functional]


@pytest.mark.parametrize('functional', ('hf', 'pbe0'))
@pytest.mark.parametrize('route', CONTOURS)
def test_contour_meets_pade_in_a_continuum(functional, route):
    """HOMO, LUMO and the solvent shift of each against the Pade route."""
    _, _, e = case(functional)
    assert np.abs(e['water', route] - e['water', 'pade']).max() < ROUTE_BAR_EV
    shift = e['water', route] - e['gas', route]
    shift_pade = e['water', 'pade'] - e['gas', 'pade']
    assert np.abs(shift - shift_pade).max() < ROUTE_BAR_EV


@pytest.mark.parametrize('functional', ('hf', 'pbe0'))
@pytest.mark.parametrize('route', CONTOURS)
def test_contour_meets_the_chains_pole_model_in_a_continuum(functional, route):
    """HOMO and LUMO against `ExcitedStateChain` with residue_route='sop',
    which screens with its own bare factor and builds Eq. (18) from two
    factorizations instead of a congruence."""
    _, _, e = case(functional)
    assert np.abs(e['water', route] - e['water', 'chain']).max() < ROUTE_BAR_EV
    assert np.abs(e['gas', route] - e['gas', 'chain']).max() < ROUTE_BAR_EV


def test_the_static_term_refuses_a_continuum_without_eq18():
    """No Eq. (18) array, no static term: the COHSEX operator is not added."""
    mol, mf, _ = case('hf')
    nocc = mol.nelectron // 2
    states = [nocc - 1, nocc]
    gas = static_exchange_mean_field_matrix(mf, mol)
    attach_solvent_screening(mf, solvent='water')
    try:
        with pytest.raises(ValueError, match='reaction_field=None'):
            static_exchange_matrix(mf, mol)
        with pytest.raises(ValueError, match='reaction_field=None'):
            static_exchange_diagonal(mf, mol, states)
        with pytest.raises(ValueError, match='reaction_field=None'):
            static_exchange_diagonal(mf, mol, states, exchange='df-direct')
        # handed the shift, the term is the mean field's plus that shift
        shift = 0.01 * np.arange(mf.mo_coeff.shape[1])
        assert np.array_equal(
            static_exchange_matrix(mf, mol, reaction_field=shift,
                                   sigma_x_matrix=gas),
            gas + np.diag(shift))
    finally:
        detach_solvent_screening(mf)
    # and the gas phase needs none
    assert np.array_equal(static_exchange_matrix(mf, mol, sigma_x_matrix=gas),
                          gas)
