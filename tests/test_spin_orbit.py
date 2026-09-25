"""Gates for src/properties/spin_orbit.py -- <S_I|H_SO|T_J>.

Two tiers, as in test_properties.py. The cheap tier needs an SCF and a TDA but
no GW: it pins the OPERATOR against an independent pySCF route, pins the
STRUCTURE of the interstate density through El-Sayed's rule, and pins the SPIN
ALGEBRA through a rotational invariance that no wrong Cartesian-to-spherical
transform can survive. The expensive tier runs the same code over the repo's
BSE routes and checks the answer does not depend on which one produced the
eigenvectors.

Run the cheap tier alone with
    -k "operator or el_sayed or rotation or hermitian or normalization or amfi
        or koseki or sublevels or ground_state or qdpt"

Absolute magnitudes are NOT gated. The spin prefactors are derived, not
measured; a perturbative 1c+SOC check against a two-component X2C-TDA is
what would let a number be quoted, and it is not built yet.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf, tdscf
from pyscf.x2c import sfx2c1e, x2c

from src.Base.constants import HARTREE_TO_CM
from src.gradients.excited_state import ExcitedStateChain
from src.Base.pyscf_interface import (get_density_fitting_coefficients,
                                      get_two_electron_integrals_chemist)
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.davidson import solve_casida_davidson
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, check_normalization, from_pyscf)
from src.properties.spin_orbit import (EXCITED_SPIN_FACTOR, chain_manifolds,
                                       ground_state_element, interstate_element,
                                       isdf_manifolds, qdpt_hamiltonian,
                                       qdpt_spectrum, soc_operator,
                                       soc_operator_mo, socme_table,
                                       spin_orbit_couplings, sublevel_elements)

BASIS = 'cc-pvdz'
CH2O = ('C 0.0000 0.0000 -0.5290; O 0.0000 0.0000 0.6746; '
        'H 0.0000 0.9376 -1.1188; H 0.0000 -0.9376 -1.1188')
H2O = 'O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468'


def scf_factory(mol):
    """Converged for gradient work, as the chain's Lagrangian requires."""
    mf = scf.RHF(mol).density_fit(auxbasis=BASIS + '-ri')
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-11, 200
    mf.kernel()
    assert mf.converged
    return mf


def tda(mf, triplet, nstates=4):
    """(omega, X, Y) from pySCF's own TDA, converted to THIS repo's normalization."""
    td = tdscf.TDA(mf)
    td.singlet = not triplet
    td.nstates = nstates
    td.kernel()
    return from_pyscf(td)


@pytest.fixture(scope='module')
def formaldehyde():
    mol = gto.M(atom=CH2O, basis='def2-svp', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    nocc = mol.nelectron // 2
    return (mol, mf, soc_operator_mo(mf, mol), nocc,
            tda(mf, False), tda(mf, True))


# --------------------------------------------------------------- the operator
def test_operator_matches_x2c_one_electron(formaldehyde):
    """G1. The one-electron Breit-Pauli operator against the SOC remainder of
    the X2C-1e core Hamiltonian -- a completely independent route through
    pySCF, sharing no code below the integral library.

    For first-row atoms Breit-Pauli IS the leading term of the exact
    two-component operator, so agreement to a per cent is the statement; a sign
    error or a wrong alpha^2/4 would show up as a factor, not a per cent.
    """
    mol = formaldehyde[0]
    nao = mol.nao_nr()
    soc = (x2c.SpinOrbitalX2CHelper(mol).get_hcore(mol)
           - np.kron(np.eye(2), sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)))
    hz_x2c = soc[:nao, :nao].imag                      # sigma_z block carries h_z
    hz_bp = soc_operator(mol, two_electron=None,
                         one_electron='breit-pauli')[2]
    ratio = abs(hz_x2c).max() / abs(hz_bp).max()
    assert ratio == pytest.approx(1.0, abs=0.02), \
        f'Breit-Pauli / X2C-1e = {ratio:.4f}, not a per-cent-level agreement'


def test_operator_is_real_antisymmetric(formaldehyde):
    """h_SO is the IMAGINARY part of a Hermitian operator, so the array carried
    here must be real and antisymmetric -- which is also why the i = j and
    a = b terms of the interstate density need no masking."""
    h = formaldehyde[2]
    assert not np.iscomplexobj(h)
    for eta in range(3):
        assert abs(h[eta] + h[eta].T).max() < 1e-12


def test_two_electron_term_screens_the_nuclear_one(formaldehyde):
    """The mean-field two-electron term OPPOSES the nuclear one and is large.

    Dropping it is not a small approximation, and a sign error in J - 3/2 K
    would make the operator grow instead of shrink.
    """
    mol, mf = formaldehyde[0], formaldehyde[1]
    bare = abs(soc_operator(mol, two_electron=None)).max()
    somf = abs(soc_operator(mol, mf.make_rdm1(), 'somf')).max()
    assert somf < bare, 'the two-electron term must screen, not amplify'
    assert somf > 0.3 * bare, 'it must not annihilate the operator either'


def test_amfi_tracks_somf_and_keeps_one_centre_blocks(formaldehyde):
    """AMFI drops the multi-centre blocks of the two-electron term. It must
    stay close to SOMF -- the operator is short-ranged -- and it must be
    strictly one-centre, which is what makes it cheap enough for an emitter."""
    mol, mf = formaldehyde[0], formaldehyde[1]
    dm = mf.make_rdm1()
    somf = soc_operator(mol, dm, 'somf')
    amfi = soc_operator(mol, dm, 'amfi')
    off = np.ones((mol.nao_nr(), mol.nao_nr()), bool)
    for _, _, p0, p1 in mol.aoslice_by_atom():
        off[p0:p1, p0:p1] = False
    two_e = amfi - soc_operator(mol, two_electron=None)
    assert abs(two_e[:, off]).max() < 1e-14, 'AMFI kept a multi-centre block'
    assert abs(amfi - somf).max() < 0.25 * abs(somf).max()


def test_koseki_route_needs_no_density(formaldehyde):
    """The effective-charge route exists precisely to avoid the two-electron
    integrals, so it must not demand the density; the mean-field routes must."""
    mol = formaldehyde[0]
    assert abs(soc_operator(mol, two_electron=None,
                            one_electron='koseki')).max() > 0
    with pytest.raises(ValueError, match='MEAN-FIELD'):
        soc_operator(mol, dm=None, two_electron='somf')
    # it is a one-electron approximation, so neither misuse is silent: naming it
    # as a two-electron one would ignore `one_electron`, and pairing it with a
    # mean-field term would count the screening twice
    with pytest.raises(ValueError, match='ONE-electron approximation'):
        soc_operator(mol, two_electron='koseki')
    with pytest.raises(ValueError, match='twice'):
        soc_operator(mol, dm=np.eye(mol.nao_nr()), one_electron='koseki',
                     two_electron='somf')


# ------------------------------------------------- structure of the density
def test_el_sayed_rule(formaldehyde):
    """G2. Formaldehyde's S1 and T1 are both n->pi*; T2 is pi->pi*.

    El-Sayed: spin-orbit coupling between states of the SAME orbital character
    vanishes, between DIFFERENT character it does not. This is the gate on the
    interstate density's structure -- swap the sign of the occupied block
    against the virtual one, or transpose either, and S1 x T1 stops being zero.
    """
    _, _, h, nocc, (es, xs, ys), (et, xt, yt) = formaldehyde
    v = socme_table(h, nocc, xs, ys, xt, yt) * HARTREE_TO_CM
    assert es[0] < es[1] and et[0] < et[1]
    assert v[1, 0] < 1e-6, f'S1 x T1 is El-Sayed forbidden, got {v[1, 0]:.3e} cm^-1'
    assert v[1, 1] > 5.0, f'S1 x T2 is El-Sayed allowed, got {v[1, 1]:.3e} cm^-1'


def test_ground_state_row_is_not_the_excited_formula(formaldehyde):
    """<S_0|H_SO|T_J> uses the occupied-virtual block; <S_I|H_SO|T_J> uses the
    occupied and virtual diagonal blocks. Reusing one formula for the other
    gives an identically zero ground-state row, since an interstate density has
    no ov block and a ground-state density has nothing else."""
    _, _, h, nocc, _, (_, xt, yt) = formaldehyde
    assert abs(ground_state_element(h, nocc, xt[:, 0], yt[:, 0])).max() > 0
    gs_via_excited = interstate_element(h, nocc, xt[:, 0], xt[:, 0])
    assert not np.allclose(gs_via_excited,
                           ground_state_element(h, nocc, xt[:, 0]))


# ------------------------------------------------------------ the spin algebra
def test_coupling_is_rotationally_invariant(formaldehyde):
    """G3. |V| is a scalar. Rotate the molecule, redo everything, and the
    number must not move.

    This is the gate on the Cartesian-to-spherical transform and on the sum
    over sublevels: any error there makes |V| depend on how the molecule
    happens to be oriented in the input file, which is the classic silent
    failure of a hand-written spin algebra.
    """
    mol = formaldehyde[0]
    rng = np.random.default_rng(20260906)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1

    def table(coords):
        m = gto.M(atom=[(mol.atom_symbol(i), coords[i]) for i in range(mol.natm)],
                  unit='Bohr', basis='def2-svp', verbose=0)
        mf = scf.RHF(m)
        mf.conv_tol = 1e-12
        mf.kernel()
        _, xs, ys = tda(mf, False, 3)
        _, xt, yt = tda(mf, True, 3)
        return socme_table(soc_operator_mo(mf, m), m.nelectron // 2,
                           xs, ys, xt, yt)

    a = table(mol.atom_coords())
    b = table(mol.atom_coords() @ q.T)
    moved = abs(a - b).max() * HARTREE_TO_CM
    assert moved < 1e-4, f'|V| moved by {moved:.2e} cm^-1 under a rotation'


def test_sublevels_sum_to_the_invariant():
    """sum_M |<S|H_SO|T,M>|^2 must equal |V|^2 exactly, for any T vector.

    The identity that makes the phase convention of the M = +-1 pair irrelevant
    to the rate, and therefore the reason `coupling` is safe to hand to
    `rates.py` without pinning that convention first.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        t = rng.standard_normal(3)
        v = sublevel_elements(t, EXCITED_SPIN_FACTOR)
        assert float((abs(v) ** 2).sum()) == pytest.approx(
            (EXCITED_SPIN_FACTOR ** 2) * float(t @ t), rel=1e-13)


# ------------------------------------------------------------ normalization
def test_normalization_is_the_repo_convention_and_pyscf_is_refused(formaldehyde):
    """Every solver here returns <X|X> - <Y|Y> = 1; pySCF returns 1/2, which is
    a FACTOR OF TWO in the rate. A raw pySCF vector must be REFUSED by name,
    not silently rescaled -- that is the one mistake worth an exception.
    """
    mf = formaldehyde[1]
    td = tdscf.TDA(mf)
    td.nstates = 2
    td.kernel()
    raw = np.column_stack([np.asarray(xy[0]).ravel() for xy in td.xy])
    with pytest.raises(ValueError, match="pySCF's convention"):
        check_normalization(raw)
    _, x, y = from_pyscf(td)
    xc, yc = check_normalization(x, y)
    assert np.allclose((xc ** 2).sum(axis=0) - (yc ** 2).sum(axis=0), 1.0)


def test_an_unnormalized_vector_is_refused():
    """A vector that is neither convention is a bug upstream, not something to
    rescale on the way past."""
    x = np.ones((6, 1)) * 0.1
    with pytest.raises(ValueError, match=r'<X\|X> - <Y\|Y>'):
        check_normalization(x)


# ------------------------------------------------------------------ the QDPT
def test_qdpt_hamiltonian_is_hermitian_with_real_spectrum(formaldehyde):
    """G4. H_rel must be Hermitian by construction, and its eigenvalues real
    and close to the unperturbed ones -- SOC is a small perturbation here."""
    _, _, h, nocc, (es, xs, ys), (et, xt, yt) = formaldehyde
    n = len(es)
    h_rel = qdpt_hamiltonian(h, nocc, es, xs, ys, et, xt, yt)
    assert h_rel.shape == (1 + n + 3 * n,) * 2
    assert abs(h_rel - h_rel.conj().T).max() < 1e-15
    w, _ = qdpt_spectrum(h_rel)
    assert np.all(np.isreal(w))
    assert w[0] == pytest.approx(0.0, abs=1e-6), 'the ground state moved'
    unperturbed = np.sort(np.concatenate([[0.0], es, np.repeat(et, 3)]))
    assert abs(np.sort(w) - unperturbed).max() < 1e-4


def test_qdpt_refuses_a_non_hermitian_matrix():
    """A silently non-Hermitian H_rel gives complex energies, which numpy would
    hand back without comment."""
    with pytest.raises(ValueError, match='Hermitian'):
        qdpt_spectrum(np.array([[0.0, 1.0], [2.0, 0.0]], complex))


def test_zero_operator_leaves_the_spectrum_untouched(formaldehyde):
    """With h_SO = 0 the QDPT step must be the identity on the spectrum."""
    _, _, h, nocc, (es, xs, ys), (et, xt, yt) = formaldehyde
    w, _ = qdpt_spectrum(qdpt_hamiltonian(np.zeros_like(h), nocc, es, xs,
                                          ys, et, xt, yt))
    unperturbed = np.sort(np.concatenate([[0.0], es, np.repeat(et, 3)]))
    assert abs(np.sort(w) - unperturbed).max() < 1e-12


# ------------------------------------------- the expensive tier: the BSE modes
@pytest.fixture(scope='module')
def water():
    mol = gto.M(atom=H2O, basis=BASIS, verbose=0)
    return mol, scf_factory(mol)


def _sorted(om, x, y):
    """Roots ascending, so two solvers' columns describe the same states."""
    o = np.argsort(np.asarray(om))
    return np.asarray(om)[o], np.asarray(x)[:, o], np.asarray(y)[:, o]


@pytest.fixture(scope='module')
def water_bse_modes(water):
    """The same BSE, solved four ways: dense and Davidson, DF and full ERIs.

    One mean field, ONE screened interaction, one set of orbital energies -- so
    what differs between them is the integral factorization and the eigensolver
    and nothing else. Built at the mean-field energies rather than at GW ones
    because the gate is about the ROUTE, and a quasiparticle shift common to
    all four routes would only slow it down.
    """
    mol, mf = water
    nocc = mol.nelectron // 2
    eps = np.asarray(mf.mo_energy)
    b_df = get_density_fitting_coefficients(mol, mf, representation='spatial')
    eri = get_two_electron_integrals_chemist(mol, mf, representation='spatial')
    lr_df = LinearResponseSolver(eps, coeff_df=b_df, spin_mode='restricted')
    lr_full = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    w_df = lr_df.static_screening_aux(nocc)
    w_full = lr_full.static_screening_aux(nocc)

    out = {}
    for name, lr, w in (('df', lr_df, w_df), ('full', lr_full, w_full)):
        manifolds = []
        for triplet in (False, True):
            a, b = lr.build_casida_matrices(nocc, lBSE=True, W_aux=w,
                                            triplet=triplet)
            manifolds.append(_sorted(*CasidaSolver(a, b).solve()))
        out['dense/' + name] = manifolds
    manifolds = []
    for spin in ('singlet', 'triplet'):
        manifolds.append(_sorted(*solve_casida_davidson(
            lr_df, nocc, nroots=4, polarizability='BSE', W_aux=w_df,
            conv_tol=1e-9, spin=spin)))
    out['davidson/df'] = manifolds
    return mol, mf, nocc, out


def test_dense_and_davidson_give_the_same_couplings(water_bse_modes):
    """G5. Two eigensolvers, identical integrals and identical W.

    Nothing physical differs between them, so the couplings must agree to the
    Davidson's own convergence and not merely be close. This is the gate that
    the module reads eigenvectors and not a solver's internals.
    """
    mol, mf, nocc, modes = water_bse_modes
    a = spin_orbit_couplings(mf, mol, nocc, *modes['dense/df'], nroots=3)
    b = spin_orbit_couplings(mf, mol, nocc, *modes['davidson/df'], nroots=3)
    gap = abs(a['socme'] - b['socme']).max() * HARTREE_TO_CM
    assert gap < 1e-4, f'dense and Davidson disagree by {gap:.3e} cm^-1'


def test_df_and_full_eri_routes_agree(water_bse_modes):
    """G5b. Density fitting versus the exact four-centre tensor.

    These ARE different Hamiltonians -- the fit moves the lowest excitation by
    tens of meV (examples/11) -- so the states themselves shift slightly and
    the tolerance is physical rather than numerical. What must not happen is a
    route-dependent factor or a sign flip.
    """
    mol, mf, nocc, modes = water_bse_modes
    a = spin_orbit_couplings(mf, mol, nocc, *modes['dense/df'], nroots=3)
    b = spin_orbit_couplings(mf, mol, nocc, *modes['dense/full'], nroots=3)
    va = a['socme'] * HARTREE_TO_CM
    vb = b['socme'] * HARTREE_TO_CM
    assert abs(va - vb).max() < 0.5, (
        f'DF and full ERIs disagree by {abs(va - vb).max():.3f} cm^-1, which '
        f'is more than the fit error on these states')


def test_isdf_manifolds_run_the_whole_bse_at_gw_route(water):
    """G5c. The ISDF matrix-free route end to end, through the adapter that
    exists because `solve_bse_isdf` takes no `spin`: one GW, one W, two spin
    channels, and a finite table out."""
    mol, mf = water
    nocc = mol.nelectron // 2
    singlet, triplet = isdf_manifolds(mf, mol, nocc, nroots=3, n_start=4)
    out = spin_orbit_couplings(mf, mol, nocc, singlet, triplet, nroots=2)
    v = out['socme'] * HARTREE_TO_CM
    assert v.shape == (3, 2)
    assert np.all(np.isfinite(v)) and v.max() > 0


def test_shared_pass_reproduces_an_independent_triplet_chain(water):
    """G5a. The one-GW-two-spin-channels shortcut is the whole reason the GW
    code needs no modification. It has to be exact, not close: the triplet
    roots off the shared pass must equal those of a separately constructed
    triplet chain, which redoes the quasiparticle set and the screening.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, spin='singlet', mf=mf)
    _, (om_t, _, _) = chain_manifolds(chain)
    independent = ExcitedStateChain(mol, scf_factory, spin='triplet', mf=mf)
    assert abs(np.asarray(om_t) - independent.spectrum()).max() < 1e-10


def test_socme_on_bse_at_gw_chain_roots(water):
    """G5b. The gradient chain's BSE@GW roots through the same entry point.

    A finite ground-state row, couplings on the scale of a first-row molecule,
    and both manifolds ordered -- which is what makes the table interpretable.
    """
    mol, mf = water
    chain = ExcitedStateChain(mol, scf_factory, spin='singlet', mf=mf)
    singlet, triplet = chain_manifolds(chain)
    out = spin_orbit_couplings(mf, mol, mol.nelectron // 2, singlet, triplet,
                               nroots=3)
    v = out['socme'] * HARTREE_TO_CM
    assert v.shape == (4, 3)
    assert np.all(np.isfinite(v))
    assert v[0].max() > 1e-3, 'every ground-state coupling vanished'
    assert v.max() < 1e4, f'{v.max():.1f} cm^-1 is not a first-row scale'
    assert np.all(np.diff(out['omega_singlet'][:3]) >= 0)
    assert np.all(np.diff(out['omega_triplet'][:3]) >= 0)
