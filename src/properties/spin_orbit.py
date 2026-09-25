"""Spin-orbit coupling between excited states: <S_I|H_SO|T_J>.

The element the reverse-intersystem-crossing rate needs, and the last of the
five inputs `rates.py` was written around. It is computed from the Casida
eigenvectors and the mean field alone:

    <S_I|H_SO^eta|T_J> = Tr[ h_SO^eta . gamma^IJ ]

so NOTHING in the GW step enters. W, the quasiparticle energies and the
self-energy matter only in producing those eigenvectors, which has already
happened by the time this module is called. The self-energy never appears in
the coupling.

WHICH ROUTE PRODUCED THE EIGENVECTORS IS NOT THIS MODULE'S BUSINESS. Every BSE
mode in the repo -- Davidson or dense, ISDF or DF or the full four-centre
tensor, gas phase or solvated, `solve_bse_isdf` or `ExcitedStateChain` or
`CasidaSolver` on `build_casida_matrices`, singlet-triplet or the periodic
Casida -- hands back the same (omega, X, Y) with X, Y of shape
(n_ov, nroots), and that is the whole interface. The signature deliberately
mirrors `davidson.oscillator_strengths(mf, mol, nocc, omega, X, Y)`, the
codebase's other property over Casida vectors, for the same reason.

NORMALIZATION IS THE REPO'S, AND IT IS CHECKED. Every solver here returns
<X|X> - <Y|Y> = 1 per root: `bse_solve`, `CasidaSolver.solve`,
`solve_casida_davidson` (which says so in its own docstring), and the TDA
branches of all three. pySCF instead returns 1/2, which is a FACTOR OF TWO in
the rate. Vectors are validated against the repo convention on the way in and
refused otherwise -- silently rescaling would hide exactly the mistake that
matters. `linear_response.from_pyscf` converts, in one visible line.

THE OPERATOR IS TWO PIECES WITH DIFFERENT PROVENANCE. The one-electron part
is the spin-orbit remainder of pySCF's X2C core Hamiltonian, exact within that
decoupling and carrying its own normalization; `int1e_pnucxp` scaled by
alpha^2/4 is its leading Breit-Pauli term and remains available. The
two-electron part is the spin-orbit mean-field reduction of Neese
(J. Chem. Phys. 122, 034107 (2005), Eq. 19),

    (alpha^2/4) [ J^eta[P] - (3/2) K^eta[P] ] ,

over `int2e_p1vxp1`, which SCREENS the nuclear term and removes about 38% of
it. alpha^2/4 is the prefactor of the PAULI MATRIX, not of the spin operator
sigma/2, and the spin factors below carry the 2 that follows from that.

The array carried here is REAL AND ANTISYMMETRIC: it is the imaginary part of a
Hermitian operator, and the i lives in the spin algebra below.

THE DENSITY is the UNRELAXED interstate transition density: products of the
Casida amplitudes, with no orbital relaxation. That is the approximation every
published TD-DFT spin-orbit number carries, so it is also the consistent choice
for comparing against them.

WHAT COMES OUT is the rotational invariant

    |V|^2 = sum_M |<S_I|H_SO|T_J,M>|^2 = T_x^2 + T_y^2 + T_z^2 ,

which is what the golden rule wants (the sum over the three degenerate triplet
sublevels) and which, being a rotational invariant of a real vector, is free of
the phase convention the literature disagrees about for the M = +-1 pair. The
per-sublevel elements are available through `sublevel_elements`. Whether k_RISC
carries a further 1/3 for thermal averaging over the initial sublevels is the
CALLER's decision -- Bredas 2017 does not, parts of the Marian literature do --
and `rates.py` is handed |V|, not a rate.
"""
import copy
import warnings

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.scf import _vhf
from pyscf.x2c import sfx2c1e, x2c

from src.Base.constants import (HEAVY_ATOM_Z, QDPT_HERMITICITY_TOL,
                                SOC_MANIFOLD_CONV_TOL)
from src.Base.utils.linearAlgebra.diagonalization import diagonalize_matrix
from src.SingleReference.base import get_occ_virt_indices
from src.SingleReference.LinearResponse.davidson import (solve_bse_isdf,
                                                         solve_casida_davidson)
from src.SingleReference.LinearResponse.linear_response import (
    LinearResponseSolver, check_normalization)
from src.properties.vibronic import displace_along_mode

# alpha^2/4, the Breit-Pauli prefactor of the PAULI MATRIX: H_SO =
# (alpha^2/2) (r x p).s with s = sigma/2 is (alpha^2/4) (r x p).sigma. So the
# array this scales is the coefficient of sigma, not of s, and the spin factors
# below carry the 2 -- gated element-wise against pySCF's two-component
# operator, where reading it as the coefficient of s halves every element.
SOC_PREFACTOR = nist.ALPHA ** 2 / 4

# The spin-coupling factors: with |S_ia>, |T^M_ia> the orthonormal spin-adapted
# singles and h the coefficient of sigma, <S_I|H_SO,z|T_J,0> = VV - OO and
# <0|H_SO,z|T_J,0> = sqrt(2) sum h_ia X_ia. They play the role sqrt(2) plays in
# `oscillator_strengths`: the spin adaptation's own factor, in the same
# spatial-orbital normalization. Halving them is the sigma/s confusion, and it
# is invisible to every ratio, invariance and route-agreement test.
EXCITED_SPIN_FACTOR = 1.0
GROUND_SPIN_FACTOR = np.sqrt(2.0)

def koseki_charge(z):
    """Koseki effective nuclear charge for a one-electron-only SOC operator.

    Transcribed from `pyscf/properties` `prop/zfs/uhf.py`, which cites
    J. Phys. Chem. 96, 10768; 99, 12764; J. Phys. Chem. A 102, 10430. It is an
    alternative to computing the two-electron term at all, and is offered only
    because `int2e_p1vxp1` is nao^4 x 3 and eventually stops being affordable;
    `two_electron='amfi'` is the better answer to that and is the reason this
    stays a fallback rather than a default.
    """
    if z <= 2:
        return float(z)
    if z <= 10:
        return z * (.3 + z * .05)
    if z <= 18:
        return z * (1.05 - z * .0125)
    if z <= 30:
        return z * (0.385 + 0.025 * (z - 20))
    if z < 48:
        return z * (4.680 + 0.060 * (z - 38))
    return float(z)


def _soc_jk(mol, dm):
    """(vj, vk) of the two-electron SOC operator at density `dm`, (3, nao, nao).

    K here is the sum of BOTH exchange contractions, (pi|iq) + (iq|pi), which is
    what the SOMF factor of 3/2 multiplies.
    """
    vj, vk, vk1 = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                                    'a4ij', ('lk->s2ij', 'jk->s1il', 'li->s1kj'),
                                    dm, 3, mol._atm, mol._bas, mol._env)
    for i in range(3):
        lib.hermi_triu(vj[i], hermi=2, inplace=True)
    return vj, vk + vk1


def _soc_jk_amfi(mol, dm):
    """(vj, vk) in the atomic-mean-field approximation: one-centre blocks only.

    The two-electron SOC operator is short-ranged and dominated by the nucleus
    it is centred on, so the multi-centre blocks are discarded. Cost falls from
    nao^4 to sum_A nao_A^4, which is what makes a 60-90 atom emitter reachable.
    """
    nao = mol.nao_nr()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    atom = copy.copy(mol)
    for b0, b1, p0, p1 in mol.aoslice_by_atom():
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = _soc_jk(atom, np.asarray(dm)[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


def x2c_one_electron(mol):
    """(3, nao, nao) one-electron spin-orbit operator from the exact
    two-component Hamiltonian, Hartree.

    The SOC remainder of the X2C core Hamiltonian is i sum_eta h^eta sigma^eta
    in the spin-orbital basis; spin-major, its alpha-alpha block is i h_z and
    its alpha-beta block i h_x + h_y, which resolves the three real
    antisymmetric spatial components. `int1e_pnucxp` is only the leading term
    of this, so nothing here is scaled by SOC_PREFACTOR -- the X2C operator
    carries its own normalization, which is the reason to prefer it.

    PICTURE CHANGE: this operator belongs to the X2C-transformed picture, so it
    is consistent with a scalar-relativistic reference (`sfx2c1e`) and only
    approximately with a non-relativistic one. The mismatch is negligible for
    first-row atoms and is not for a 5d metal.
    """
    nao = mol.nao_nr()
    soc = (x2c.SpinOrbitalX2CHelper(mol).get_hcore(mol)
           - np.kron(np.eye(2), sfx2c1e.SpinFreeX2CHelper(mol).get_hcore(mol)))
    h = np.array([soc[:nao, nao:].imag, soc[:nao, nao:].real,
                  soc[:nao, :nao].imag])
    # antisymmetric by construction; the X2C decoupling leaves ~1e-11 of noise
    # on that, and the vanishing diagonal is what lets the interstate density
    # skip its i = j, a = b mask
    return 0.5 * (h - h.transpose(0, 2, 1))


def soc_operator(mol, dm=None, two_electron='somf', one_electron='x2c'):
    """(3, nao, nao) real antisymmetric spin-orbit operator, Hartree.

    dm: the ground-state AO density the mean field is built at. Required
        unless two_electron is None.
    one_electron: 'x2c' (the exact two-component operator, which carries its
        own normalization), 'breit-pauli' (`int1e_pnucxp` scaled by alpha^2/4,
        its leading term; the two differ by 0.17% on formaldehyde and by more
        as Z rises), or 'koseki' (effective nuclear charges, which absorb the
        screening INTO the one-electron term and therefore take
        two_electron=None and need no density).
    two_electron: 'somf' (Neese Eq. 19, J - 3/2 K, exact two-electron
        integrals), 'amfi' (the same with one-centre blocks only), or None (the
        one-electron operator alone, for diagnostics only -- it overestimates
        by about 38%, which is what the two-electron term screens away).
    """
    if two_electron == 'koseki':
        raise ValueError(
            "two_electron='koseki' names a ONE-electron approximation: it "
            "replaces the nuclear term with effective charges rather than "
            "adding a mean-field term to it. Pass one_electron='koseki', "
            "two_electron=None.")
    if one_electron == 'koseki':
        if two_electron is not None:
            raise ValueError(
                f"one_electron='koseki' already carries the two-electron "
                f"screening inside its effective charges, so "
                f"two_electron={two_electron!r} would count it twice. Pass "
                f"two_electron=None.")
        h1 = np.zeros((3, mol.nao_nr(), mol.nao_nr()))
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            h1 -= koseki_charge(mol.atom_charge(ia)) * \
                mol.intor_asymmetric('int1e_prinvxp', 3)
        return h1 * SOC_PREFACTOR

    if one_electron == 'x2c':
        h1 = x2c_one_electron(mol)
    elif one_electron == 'breit-pauli':
        h1 = mol.intor_asymmetric('int1e_pnucxp', 3) * SOC_PREFACTOR
    else:
        raise ValueError(f"one_electron={one_electron!r}: 'x2c', "
                         f"'breit-pauli' or 'koseki'")
    if two_electron is None:
        return h1
    if dm is None:
        raise ValueError(f"two_electron={two_electron!r} is a MEAN-FIELD "
                         f"reduction and needs the ground-state density; pass "
                         f"dm=mf.make_rdm1(). Only two_electron=None and "
                         f"one_electron='koseki' do not.")
    if two_electron == 'amfi':
        vj, vk = _soc_jk_amfi(mol, dm)
    elif two_electron == 'somf':
        vj, vk = _soc_jk(mol, dm)
    else:
        raise ValueError(f"two_electron={two_electron!r}: 'somf', 'amfi' "
                         f"or None")
    # the mean-field term stays Breit-Pauli whichever one-electron operator
    # is used, which is the standard x2c-plus-mean-field compromise
    return h1 + (vj - 1.5 * vk) * SOC_PREFACTOR


def soc_operator_mo(mf, mol=None, two_electron='somf', one_electron='x2c'):
    """(3, nmo, nmo) spin-orbit operator in the mean field's orbital basis.

    Built once per geometry; every root pair then costs two small contractions.
    """
    mol = mf.mol if mol is None else mol
    if (one_electron == 'x2c' and getattr(mf, 'with_x2c', None) is None
            and (mol.atom_charges() > HEAVY_ATOM_Z).any()):
        warnings.warn(
            'the X2C spin-orbit operator is being used with a mean field that '
            'is not scalar-relativistic, and this molecule carries an element '
            'heavier than Kr: the two are in different pictures. Build the '
            'reference with .x2c() (or sfx2c1e) to match them.', RuntimeWarning)
    h_ao = soc_operator(mol, mf.make_rdm1(), two_electron, one_electron)
    mo = np.asarray(mf.mo_coeff)
    return np.einsum('xpq,pi,qj->xij', h_ao, mo, mo, optimize=True)


def _blocks(h_mo, nocc):
    """(h_oo, h_vv, h_ov) of the spin-orbit operator, the only pieces used."""
    norb = np.asarray(h_mo).shape[-1]
    occ, virt = get_occ_virt_indices(np.zeros(norb), nocc)
    return (h_mo[:, occ[:, None], occ[None, :]],
            h_mo[:, virt[:, None], virt[None, :]],
            h_mo[:, occ[:, None], virt[None, :]])


def interstate_element(h_mo, nocc, xs, xt, ys=None, yt=None):
    """T_eta = Tr[h_SO^eta . gamma^IJ], the (3,) real vector, for ONE root pair.

        gamma_ji = - sum_a (X^S_ia X^T_ja + Y^S_ia Y^T_ja)     occupied block
        gamma_ab = + sum_i (X^S_ia X^T_ib + Y^S_ia Y^T_ib)     virtual block

    the interstate generalization of the unrelaxed excited-state difference
    density. The diagonal needs no masking: h_SO is antisymmetric, so h_ii and
    h_aa vanish identically and the i = j, a = b terms cost nothing.

    Outside the Tamm-Dancoff approximation this (XX + YY) form is a choice.
    PySOC instead contracts (X + Y), which is the transition-moment
    normalization rather than the density one; Holzer names exactly that as one
    of the two things previous TD-DFT implementations get wrong.

    xs, xt, ys, yt are single roots, shape (nocc, nvir) or flat (n_ov,).
    """
    h_oo, h_vv, _ = _blocks(np.asarray(h_mo), nocc)
    xs, xt = np.reshape(xs, (nocc, -1)), np.reshape(xt, (nocc, -1))
    ys = np.zeros_like(xs) if ys is None else np.reshape(ys, (nocc, -1))
    yt = np.zeros_like(xt) if yt is None else np.reshape(yt, (nocc, -1))
    return (-np.einsum('xji,ia,ja->x', h_oo, xs, xt, optimize=True)
            - np.einsum('xji,ia,ja->x', h_oo, ys, yt, optimize=True)
            + np.einsum('xab,ia,ib->x', h_vv, xs, xt, optimize=True)
            + np.einsum('xab,ia,ib->x', h_vv, ys, yt, optimize=True))


def ground_state_element(h_mo, nocc, xt, yt=None):
    """T_eta between the ground state and one triplet root, (3,) real.

    Only the occupied-virtual block, and (X - Y) rather than (X + Y): the
    spatial part of h_SO is real antisymmetric, the same class as the magnetic
    dipole and the velocity operator, whose ground-to-excited moments take the
    minus combination. (`oscillator_strengths` takes the plus combination for
    the electric dipole, which is symmetric -- the contrast is the point.)
    """
    _, _, h_ov = _blocks(np.asarray(h_mo), nocc)
    xt = np.reshape(xt, (nocc, -1))
    yt = np.zeros_like(xt) if yt is None else np.reshape(yt, (nocc, -1))
    return np.einsum('xia,ia->x', h_ov, xt - yt, optimize=True)


def sublevel_elements(t_vec, spin_factor):
    """(3,) complex <S|H_SO|T,M> for M = -1, 0, +1.

        M = 0    :  c T_z
        M = +-1  :  -+ (c/sqrt 2) (T_x -+ i T_y)

    The overall i of the Hermitian operator is carried here, since h_SO is the
    IMAGINARY part of it. The relative phase of the M = +-1 pair is convention;
    `coupling` below, which is what the rate is given, does not depend on it.
    """
    tx, ty, tz = (complex(v) for v in t_vec)
    c = spin_factor
    return 1j * np.array([+(c / np.sqrt(2.0)) * (tx + 1j * ty),
                          c * tz,
                          -(c / np.sqrt(2.0)) * (tx - 1j * ty)], complex)


def coupling(t_vec, spin_factor):
    """|V| = sqrt(sum_M |<S|H_SO|T,M>|^2) in Hartree -- what `rates.py` takes.

    A rotational invariant of a real three-vector, so no phase convention and
    no molecular orientation can move it.
    """
    return float(spin_factor * np.linalg.norm(np.asarray(t_vec, float)))


def socme_table(h_mo, nocc, x_s, y_s, x_t, y_t, nroots=None):
    """(n_singlet + 1, n_triplet) table of |<S_I|H_SO|T_J>| in Hartree.

    Row 0 is the ground state. X, Y are the solver's own (n_ov, nroots) arrays,
    from ANY of the BSE routes; `nroots` truncates both manifolds.
    """
    x_s, y_s = check_normalization(x_s, y_s)
    x_t, y_t = check_normalization(x_t, y_t)
    n_s = x_s.shape[1] if nroots is None else min(nroots, x_s.shape[1])
    n_t = x_t.shape[1] if nroots is None else min(nroots, x_t.shape[1])
    out = np.zeros((n_s + 1, n_t))
    for j in range(n_t):
        out[0, j] = coupling(
            ground_state_element(h_mo, nocc, x_t[:, j], y_t[:, j]),
            GROUND_SPIN_FACTOR)
        for i in range(n_s):
            out[i + 1, j] = coupling(
                interstate_element(h_mo, nocc, x_s[:, i], x_t[:, j],
                                   y_s[:, i], y_t[:, j]),
                EXCITED_SPIN_FACTOR)
    return out


def qdpt_hamiltonian(h_mo, nocc, omega_s, x_s, y_s, omega_t, x_t, y_t,
                     nroots=None):
    """The effective relativistic Hamiltonian H_rel = H_0 + H_SOC, complex Hermitian.

    Basis, in order: the ground state, then the singlet roots, then each triplet
    root as its three sublevels M = -1, 0, +1. H_0 is diagonal in the excitation
    energies with the three sublevels still degenerate; H_SOC fills the
    singlet-triplet blocks.

    The singlet-singlet block is zero by Wigner-Eckart -- H_SO is a rank-one
    spin tensor and cannot connect two spin-zero states. The triplet-triplet
    block is NOT zero; it is the zero-field-splitting-like term, it does not
    couple singlets to triplets at first order, and it is omitted here. That
    omission is visible in the spectrum as unsplit sublevels, not hidden.
    """
    x_s, y_s = check_normalization(x_s, y_s)
    x_t, y_t = check_normalization(x_t, y_t)
    n_s = x_s.shape[1] if nroots is None else min(nroots, x_s.shape[1])
    n_t = x_t.shape[1] if nroots is None else min(nroots, x_t.shape[1])
    dim = 1 + n_s + 3 * n_t
    h = np.zeros((dim, dim), complex)
    h[1:1 + n_s, 1:1 + n_s] = np.diag(np.asarray(omega_s, float)[:n_s])
    for j in range(n_t):
        k = 1 + n_s + 3 * j
        h[k:k + 3, k:k + 3] = np.eye(3) * float(omega_t[j])
        h[0, k:k + 3] = sublevel_elements(
            ground_state_element(h_mo, nocc, x_t[:, j], y_t[:, j]),
            GROUND_SPIN_FACTOR)
        for i in range(n_s):
            h[1 + i, k:k + 3] = sublevel_elements(
                interstate_element(h_mo, nocc, x_s[:, i], x_t[:, j],
                                   y_s[:, i], y_t[:, j]),
                EXCITED_SPIN_FACTOR)
    return h + h.conj().T - np.diag(np.diag(h))


def qdpt_spectrum(h_rel):
    """(energies, vectors) of H_rel, through the solver the Casida step uses.

    `diagonalize_matrix` is complex-Hermitian through scipy's `eigh` and picks
    up the distributed path above its own threshold; H_rel is a handful of
    states, so this is always the local solve, and reusing it is what keeps the
    module from carrying an eigensolver of its own.
    """
    h_rel = np.asarray(h_rel)
    hermiticity = float(abs(h_rel - h_rel.conj().T).max())
    if hermiticity > QDPT_HERMITICITY_TOL:
        raise ValueError(f'H_rel is not Hermitian to {hermiticity:.2e}; the '
                         f'spin-orbit blocks were assembled inconsistently')
    energies, vectors, _, _, _ = diagonalize_matrix(h_rel)
    return energies, vectors


def spin_orbit_couplings(mf, mol, nocc, singlet, triplet, nroots=None,
                         two_electron='somf', one_electron='x2c'):
    """|<S_I|H_SO|T_J>| for a singlet and a triplet manifold of the SAME geometry.

    The one-call entry, and the only one a caller normally needs:

        singlet, triplet = isdf_manifolds(mf, mol, nocc, nroots=5)
        out = spin_orbit_couplings(mf, mol, nocc, singlet, triplet)
        k = marcus_levich_jortner_rate(out['socme'][1, 0], ...)

    or, with the manifolds already in hand from any other route,

        out = spin_orbit_couplings(mf, mol, nocc, (om_s, X_s, Y_s),
                                                  (om_t, X_t, Y_t))

    `singlet` and `triplet` are (omega, X, Y) triples exactly as any solver in
    the repo returns them -- Davidson or dense, ISDF or DF or full ERIs, gas
    phase or solvated. Y may be None for a Tamm-Dancoff run.

    Returns a dict: 'socme' the (n_s + 1, n_t) table in Hartree with the ground
    state in row 0, 'h_mo' the operator, 'h_rel' the effective relativistic
    Hamiltonian, and the two energy arrays.
    """
    om_s, x_s, y_s = singlet
    om_t, x_t, y_t = triplet
    h_mo = soc_operator_mo(mf, mol, two_electron, one_electron)
    return {'socme': socme_table(h_mo, nocc, x_s, y_s, x_t, y_t, nroots),
            'h_mo': h_mo,
            'h_rel': qdpt_hamiltonian(h_mo, nocc, om_s, x_s, y_s,
                                      om_t, x_t, y_t, nroots),
            'omega_singlet': np.asarray(om_s, float),
            'omega_triplet': np.asarray(om_t, float)}


def promoting_mode_couplings(mol0, omega, modes, masses, soc_at, i, j, dq=0.1):
    """|dV_ij/dq_k| in Hartree, every real ground-state mode, Herzberg-Teller.

    The correction Samanta, Kim, Coropceanu and Bredas, J. Am. Chem. Soc.
    139, 4042 (2017) carry beside the Condon term: for an El-Sayed-forbidden
    pair the vertical <S_I|H_SO|T_J> is near zero and the rate is carried
    instead by the derivative along a mode that lends the pair the symmetry
    it lacks.

    soc_at: mol -> the (n_s + 1, n_t) socme table `spin_orbit_couplings`
    returns, so this function owns no electronic-structure route and works
    on any of them; `i`, `j` index that table (i = 0 is the ground state
    row). `omega`, `modes`, `masses` are `vibronic.normal_modes`'s, and
    `displace_along_mode` reads the dimensionless step `dq`.

    |V| IS A NORM: for a Condon-forbidden pair it passes through zero
    LINEARLY in q, so |V(+dq)| = |V(-dq)| and a central difference of the
    modulus is identically zero however large the true slope is. V^2 is
    smooth through the zero instead:

        |dV/dq_k|^2 = (|V(+dq_k)|^2 + |V(-dq_k)|^2 - 2|V(0)|^2) / (2 dq^2)

    which is exact for V linear in q_k, Condon term included, and is what is
    actually differentiated below.

    Returns (v0, modes): the Condon term |V_ij(mol0)| in Hartree -- read
    here rather than left for the caller to recompute -- and a list of
    (mode index, omega_k in Hartree, |dV/dq_k| in Hartree) over modes with
    omega_k > 0. An imaginary omega_k is not a displacement direction at a
    minimum and nothing here means anything along it.
    """
    v0 = float(soc_at(mol0)[i, j])
    out = []
    for k in range(modes.shape[1]):
        if omega[k] <= 0:
            continue
        v_pm = [float(soc_at(displace_along_mode(mol0, modes, masses, omega,
                                                  k, sign * dq))[i, j])
               for sign in (+1, -1)]
        dv2 = (v_pm[0] ** 2 + v_pm[1] ** 2 - 2.0 * v0 ** 2) / (2.0 * dq ** 2)
        out.append((k, float(omega[k]), float(np.sqrt(max(0.0, dv2)))))
    return v0, out


def isdf_manifolds(mf, mol, nocc, nroots=5, conv_tol=SOC_MANIFOLD_CONV_TOL,
                   **kwargs):
    """((omega, X, Y) singlet, (omega, X, Y) triplet) from ONE GW and ONE W.

    The adapter for the ISDF matrix-free route. `solve_bse_isdf` is
    singlet-only -- it takes no `spin` -- but it returns the quasiparticle
    energies, the separable factors and W in `info`, and the Davidson driver
    underneath it does take `spin`. So the triplet manifold costs one more
    Davidson and no second GW, and neither driver needs a line changed.

    A triplet BSE is where a singlet/triplet-unstable reference shows up. The
    dense route refuses such a case by name; the Davidson does not probe, so a
    triplet root that comes back below its own Tamm-Dancoff value is the signal
    to re-run with the TDA form rather than to trust the number.
    """
    om_s, x_s, y_s, info = solve_bse_isdf(mf, mol, nocc, nroots=nroots,
                                          conv_tol=conv_tol, **kwargs)
    lr = LinearResponseSolver(info['eps'], spin_mode='restricted')
    om_t, x_t, y_t = solve_casida_davidson(
        lr, nocc, nroots=nroots, polarizability='BSE', W_aux=info['W_aux'],
        isdf_factors=info['factors'], conv_tol=conv_tol, spin='triplet')
    return (om_s, x_s, y_s), (om_t, x_t, y_t)


def chain_manifolds(chain, mol=None, mf=None):
    """((omega, X, Y) singlet, (omega, X, Y) triplet) from ONE GW and ONE W.

    The adapter for `ExcitedStateChain`, whose spin channel is fixed at
    construction. `_forward` computes the quasiparticle set and the screening
    BEFORE the Casida step, and only the Casida step reads `chain.spin` -- so
    both manifolds come off a single pass, and neither the GW code nor the
    chain needs a line changed. The spin-flipped copy is shallow on purpose: it
    shares the mean field, the grids and every setting, and differs in exactly
    the one attribute that selects the kernel.

    Reaching for `_forward` and `_casida` is the deliberate cost of not
    modifying the GW or BSE code; the gate asserts these roots equal an
    independently constructed chain's. This varies SPIN, where `StateManifold`
    varies the root off one forward pass; the two fold together the day an
    interstate quantity needs both axes at once.
    """
    mol, mf = chain.mean_field(mol, mf)
    om_a, pieces = chain._forward(mol, mf)
    x_mo, d, eps_qp, w_aux = pieces[4], pieces[5], pieces[7], pieces[8]
    other = copy.copy(chain)
    other.spin = 'triplet' if chain.spin == 'singlet' else 'singlet'
    om_b, xb, yb, _ = other._casida(x_mo, d, eps_qp, w_aux)
    a = (om_a, pieces[10], pieces[11])
    b = (om_b, xb, yb)
    return (a, b) if chain.spin == 'singlet' else (b, a)
