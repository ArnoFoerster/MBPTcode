"""
Self-contained periodic RI-RPA (direct RPA) correlation energy with k-point
sampling.

Owns the Coulomb kernel explicitly: the interaction enters ONLY through a
`coulG_fn(cell, q, Gv) -> v(q+G)` callback. Passing the bare `tools.get_coulG`
gives standard dRPA; passing a damped kernel (see pbc_rpa_damping) gives the
AUTO-damped RPA of Forster et al., JCTC 2025, 21, 9347, with no other change.

RI-V with the SAME kernel v in both the 2-center metric J and the 3-center B:
    B_{P,ia}(k)   = sum_G  chi_P(q+G)^*  v(q+G)  rho_ia^k(q+G)
    J_{PQ}(q)     = sum_G  chi_P(q+G)^*  v(q+G)  chi_Q(q+G)
    Lhalf         = J^{-1/2} B                          (Coulomb-metric fit)
    Pi(q,iw)      = (2/Nk) sum_k sum_ia Lhalf * [2 e_ia/(w^2+e_ia^2)] * Lhalf^*
    E_c           = (1/Nq) sum_q (1/2pi) int dw [ logdet(1-Pi) + Tr Pi ]

rho_ia^k(q+G) = < i_k | e^{-i(q+G).r} | a_{k+q} > are MO pair densities in G-space
(from ft_aopair on the AO pairs, then AO->MO). Damping is exact by construction
because v is diagonal in G; a damped kernel is just a different multiplier.

The G-sum is blocked for memory. Imaginary-frequency integration uses the repo
grids (src.Base.utils.grids.grids), not pyscf's.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df import ft_ao
from pyscf.df import addons

from src.Base.numerics.constants import PBC_METRIC_EIG_FLOOR, PBC_RPA_NFREQ
from src.Base.utils.grids.grids import (gap_scaled_w0, gauss_legendre_grid,
                                        minimax_frequency_grid, minimax_supported_sizes)
from src.Base.utils.grids.matsubara import beta_from_mf, thermal_e_min
from src.SingleReference.Periodic.pbc_occupations import (is_integer_occupation,
                                                          split_transition_window,
                                                          transition_window)
from src.SingleReference.Periodic.pbc_casida import build_chi0_aux
from src.SingleReference.Periodic.pbc_integrals import (bz_index,
                                                        get_momentum_transfer_map)
from src.SingleReference.Periodic.pbc_rpa_damping import assert_damping_fits
from src.SingleReference.LinearResponse.space_time import dielectric_log_determinant

einsum = lib.einsum


def build_freq_grid(mo_energy, nocc, nw, grid='gauss_legendre', beta=None,
                    mo_occ=None):
    """Imaginary-frequency points/weights from the repo grids.

    grid='gauss_legendre' (default, gap-scaled w0) or 'minimax' (GreenX table;
    nw must be in minimax_supported_sizes()).

    The metallic case. Two things go wrong for a metal and only the second
    one announces itself:

    * `split_transition_window` applies ONE integer occupied count at every k-point,
      so for a metal it measures a gap across an artificial boundary. Depending
      on how ragged the collapse is, that comes out either NEGATIVE -- on a Li
      monolayer with gth-dzvp, where per-k nocc is [8, 9, 11], it gives
      -0.0418 Ha and grids.py raises -- or positive, plausible and fictitious:
      the same system in gth-szv gives +0.0888 Ha against an honest 0.0673, and
      nothing raises at all. Which failure you get is an accident of the basis.
      Pass `mo_occ` and the window comes from the occupation-weighted
      transition set instead (pbc_occupations), which is the honest one.
    * Every grid here is scaled by e_min, and the honest e_min for a metal goes
      to zero, at which point the T = 0 construction is undefined rather than
      inaccurate. Pass `beta` (in inverse Hartree) and e_min is floored at the
      first Matsubara frequency, pi/beta, via `matsubara.thermal_e_min` -- at
      finite temperature that IS the low-energy scale, so the existing
      quadratures become well defined with no change to what consumes them. On
      that system the honest 0.0019 Ha sits below pi/beta = 0.0628, so it is
      the temperature that sets the scale.

    Both default to None, so a gapped system is untouched: `thermal_e_min`
    returns the gap itself whenever the gap exceeds pi/beta.
    """
    if mo_occ is not None and not is_integer_occupation(mo_occ):
        e_min, e_max = transition_window(mo_energy, mo_occ)
    else:
        e_min, e_max = split_transition_window(mo_energy, nocc)
    if beta is not None:
        e_min = thermal_e_min(beta, e_min)
    if grid == 'minimax':
        if nw not in minimax_supported_sizes():
            raise ValueError(f"nw={nw} not in minimax sizes {minimax_supported_sizes()}")
        return minimax_frequency_grid(nw, e_min, e_max)
    elif grid == 'gauss_legendre':
        return gauss_legendre_grid(nw, w0=gap_scaled_w0(gap=e_min))
    raise ValueError(f"unknown grid '{grid}'")


def make_auxcell(cell, auxbasis='weigend'):
    """Build an auxiliary Cell (RI fitting basis) from a molecular auxbasis."""
    from pyscf.pbc import gto as pgto
    auxcell = addons.make_auxmol(cell, auxbasis=auxbasis)
    aux = pgto.Cell()
    aux.a = cell.a
    aux.atom = cell._atom
    aux.basis = auxcell.basis
    aux.unit = 'Bohr'
    aux.dimension = cell.dimension
    aux.low_dim_ft_type = cell.low_dim_ft_type
    aux.precision = cell.precision
    aux.mesh = cell.mesh
    aux.verbose = 0
    aux.build()
    return aux


def coulomb_metric_inv_sqrt(J, tol=PBC_METRIC_EIG_FLOOR):
    """J^{-1/2} for the RI-V Coulomb-metric fit, refusing an indefinite metric.

    J_{PQ}(q) = sum_G chi_P(q+G)^* v(q+G) chi_Q(q+G) is positive SEMIdefinite
    only while the kernel v(q+G) is non-negative for every G, so its only
    admissible small eigenvalues are the linear dependencies of the auxiliary
    basis, which `tol` drops. A genuinely NEGATIVE eigenvalue is a different
    animal: pyscf's get_coulG returns v(G=0) = -pi L_z^2 / 2 for
    cell.dimension == 2 (Sundararaman-Arias) and a negative G=0 value for
    cell.dimension == 1 as well, which tilts J indefinite -- and the offending
    direction is typically the LARGEST in magnitude, not the smallest, so
    filtering it away as null space would silently delete the long-wavelength
    channel of the fit. Raise instead.
    """
    e, U = np.linalg.eigh(J)
    emax = e.max()
    if emax <= 0:
        raise ValueError(f"Coulomb metric J has no positive eigenvalue "
                         f"(max {emax:.6g}); the kernel v(q+G) cannot be a "
                         f"valid RI-V metric.")
    if (e < -tol * emax).any():
        raise ValueError(
            f"Coulomb metric J is not positive definite: {(e < -tol * emax).sum()} "
            f"negative eigenvalue(s), min {e.min():.6g} vs max {emax:.6g}. The RI-V "
            f"fit needs v(q+G) >= 0 for all G; pyscf's get_coulG puts a NEGATIVE "
            f"value at G=0 for cell.dimension < 3. Set "
            f"cell.low_dim_ft_type='inf_vacuum' (truncated, non-negative kernel), "
            f"pass a damped coulG_fn (pbc_rpa_damping.make_coulG_damped, whose "
            f"v(0) = 4 pi int r theta(r) dr > 0), or supply your own non-negative "
            f"kernel.")
    keep = e > tol * emax
    return (U[:, keep] * (e[keep] ** -0.5)) @ U[:, keep].conj().T


def kernel_cell(cell, kpts, coulG_fn=None, mesh=None, check_support=True):
    """(cell, coulG_fn) for a G-space RI build: the cell on `mesh` (a copy if
    one is given), the kernel (bare pyscf get_coulG by default), and the
    check that a damped kernel's support fits the cell at this k-mesh.

    A damped kernel's real-space support must still fit the cell along every
    unsampled direction at THIS k-mesh -- r0 grows with the grid while the
    vacuum does not, so a slab that was fine on a coarse mesh can silently
    stop being fine on a finer one. No-op for any non-damped kernel.
    """
    if mesh is not None:
        cell = cell.copy()
        cell.mesh = mesh
        cell.build(False, False)
    if coulG_fn is None:
        coulG_fn = lambda c, q, Gv: tools.get_coulG(c, k=q, mesh=c.mesh, Gv=Gv)
    if check_support:
        assert_damping_fits(cell, kpts, coulG_fn)
    return cell, coulG_fn


def gspace_ri_factors(cell, aux, mo_coeff, kpts, coulG_fn, bra=slice(None),
                      ket=slice(None), blk=None, extra=None, inv_sqrt=None):
    """Yield (q, Lhalf) per momentum transfer, the RI-V factor of the pairs
    (bra @ k, ket @ k+q) in the Coulomb metric of the kernel v:

        B_{P,pr}(k) = sum_G chi_P(q+G)^* v(q+G) <p_k|e^{-i(q+G)r}|r_{k+q}>
        J_{PQ}(q)   = sum_G chi_P(q+G)^* v(q+G) chi_Q(q+G)
        Lhalf       = J^{-1/2} B,   shape (naux, nkpts, nbra, nket).

    bra / ket select the orbitals (slices of mo_coeff's last axis); blk the
    G block size (None: all G at once). extra(auxG, rho, qtrue) -> (dJ, dB)
    adds a further kernel to J and to B[:, k] for each block (the slab
    reaction field, which needs all G at once); inv_sqrt(J, q) replaces
    `coulomb_metric_inv_sqrt` (to explain a refusal).
    """
    nkpts = len(kpts)
    naux = aux.nao_nr()
    nmo = mo_coeff.shape[-1]
    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]
    b = cell.reciprocal_vectors()
    Gv, _, kws = cell.get_Gv_weights(cell.mesh)          # kws = grid weight
    kws = np.asarray(kws)
    nG = len(Gv)
    blk = nG if blk is None else blk
    nb = len(range(nmo)[bra])
    nk = len(range(nmo)[ket])

    for iq in range(nkpts):
        J = np.zeros((naux, naux), dtype=np.complex128)
        B = np.zeros((naux, nkpts, nb, nk), dtype=np.complex128)
        for ki in range(nkpts):
            kj, G0 = bz_index(kscaled, kscaled[ki] + kscaled[iq])
            qtrue = (kscaled[kj] + G0 - kscaled[ki]).dot(b)     # = qscaled . b
            kptj = kscaled[kj].dot(b)                            # BZ image
            ci = mo_coeff[ki][:, bra].conj()
            cj = mo_coeff[kj][:, ket]
            for p0 in range(0, nG, blk):
                p1 = min(p0 + blk, nG)
                Gblk = Gv[p0:p1]
                kw_blk = kws if kws.ndim == 0 else kws[p0:p1]
                vG = coulG_fn(cell, qtrue, Gblk) * kw_blk       # v(q+G) x weight
                auxG = ft_ao.ft_ao(aux, Gblk, kpt=qtrue)         # chi_P(q+G)
                pqG = ft_ao.ft_aopair(cell, Gblk, aosym='s1',
                                      kpti_kptj=(kpts[ki], kptj), q=qtrue)
                pqG = pqG.reshape(p1 - p0, nmo, nmo)
                rho = einsum('gpq,pi,qa->gia', pqG, ci, cj)
                w_auxG = auxG.conj() * vG[:, None]               # chi_P^* v
                J += einsum('gP,gQ->PQ', w_auxG, auxG)
                B[:, ki] += einsum('gP,gia->Pia', w_auxG, rho)
                if extra is not None:
                    dJ, dB = extra(auxG, rho, qtrue)
                    J += dJ
                    B[:, ki] += dB
        # J(q) depends only on q: each of the nkpts ki-terms adds the identical
        # J(q) (aux/Coulomb are invariant under q -> q+G0), so undo the overcount.
        J /= nkpts
        Jm12 = coulomb_metric_inv_sqrt(J) if inv_sqrt is None else inv_sqrt(J, iq)
        yield iq, einsum('PQ,Qkia->Pkia', Jm12, B)


def build_Pi_all(cell, aux, mo_coeff, mo_energy, nocc, kpts, coulG_fn,
                 freqs, blk=40000):
    """Return a list over q-points of Pi(q, iw), each of shape (nw, naux, naux).

    I - Pi(q, iw) is the RI-V RPA dielectric matrix in the Coulomb-metric
    auxiliary basis (spin factor and 1/nkpts BZ average folded in).
    """
    nkpts = len(kpts)
    naux = aux.nao_nr()
    kconserv = get_momentum_transfer_map(cell, kpts)
    nw = len(freqs)

    Pi_out = []
    for iq, Lhalf in gspace_ri_factors(cell, aux, mo_coeff, kpts, coulG_fn,
                                       bra=slice(0, nocc),
                                       ket=slice(nocc, None), blk=blk):
        Pi_q = np.zeros((nw, naux, naux), dtype=np.complex128)
        for ki in range(nkpts):
            kj = kconserv[iq, ki]
            eia = (mo_energy[ki][:nocc, None] - mo_energy[kj][None, nocc:]).ravel()  # <0
            L = Lhalf[:, ki].reshape(naux, -1)
            for w in range(nw):
                chi = 2.0 * eia / (freqs[w] ** 2 + eia ** 2)     # 2 e_ia/(w^2+e^2)
                Pi_q[w] += (2.0 / nkpts) * (L * chi) @ L.conj().T
        Pi_out.append(Pi_q)
    return Pi_out


def ri_rpa_ecorr_from_dfints(dfints, nw=PBC_RPA_NFREQ, grid='gauss_legendre',
                             mo_energy=None, verbose=False, return_per_q=False,
                             beta=None, mo_occ=None, flags=None):
    """Direct-RPA correlation energy from a prebuilt integrals object, with the
    momentum-transfer sum taken over WHATEVER transfer grid that object carries.

    Same quantity and same formula as ri_rpa_ecorr -- the difference is only that
    the q-sum is a weighted sum over dfints' transfers rather than a hardcoded
    uniform average over the regular mesh:

        E_c = sum_Q w_Q (1/2pi) int dw [ logdet(1 - Pi(Q,iw)) + Tr Pi(Q,iw) ]

    w_Q = 1/nkpts, so this reduces EXACTLY to ri_rpa_ecorr. The weighted form
    is kept because the q->0 integrable Coulomb divergence is what makes RPA
    E_c converge slowly to the thermodynamic limit, and the treatment of that
    limit lives in the KERNEL (pbc_rpa_damping, pbc_wav) rather than in a
    special q-grid -- see pbc_smallq for the head analysis.

    Pi comes from pbc_casida.build_chi0_aux, occupation-weighted, so a metal
    and a gap take the same path. On a gap it is validated against pyscf's
    krgw_ac.get_rho_response and carries the same (4/nkpts) normalization as
    build_Pi_all's (2/nkpts) * [2 e/(w^2+e^2)].

    flags: a dict that receives the numerical flags (`dielectric_log_determinant`:
    'nonpositive_det', the (Q, w) whose det(1 - Pi) was not positive).
    """
    if mo_energy is None:
        mo_energy = dfints.mo_energy
    mo_energy = np.asarray(mo_energy)

    nq = dfints.nkpts
    qw = np.full(nq, 1.0 / nq)

    nocc = int(dfints.nocc[0])
    if mo_occ is None:
        mo_occ = getattr(dfints, 'mo_occ', None)
    freqs, wts = build_freq_grid(list(mo_energy.real), nocc, nw, grid=grid,
                                 beta=beta, mo_occ=mo_occ)

    # The response is occupation-weighted, not just the grid it is sampled on.
    # An integer-occupation Pi on a metal jumps DISCONTINUOUSLY with volume,
    # because the integer nocc changes in steps: measured on BCC Li at 2x2x2,
    # nocc at one k-point goes 3 -> 6 between a = 3.50 and 3.60 A, the
    # transition space 509 -> 536, and E_c jumps by 0.011 Ha in the middle of
    # an equation of state. The occupation-weighted chi0 is continuous there
    # because the weights go to zero smoothly.
    per_q = np.zeros(nq)
    for Q in range(nq):
        kconserv = dfints.kconserv_pair[Q]
        ec_q = 0.0
        for w in range(nw):
            Pi = build_chi0_aux(dfints, kconserv, freqs[w],
                                mo_energy=mo_energy, mo_occ=mo_occ,
                                imaginary=True)
            naux = Pi.shape[0]
            logdet = dielectric_log_determinant(np.eye(naux) - Pi, (int(Q), int(w)), flags)
            ec_q += wts[w] / (2 * np.pi) * (logdet + np.trace(Pi).real)
        per_q[Q] = ec_q.real
        if verbose:
            print(f"  Q={Q:3d}  w={qw[Q]:.6f}  Ec(Q) = {per_q[Q]: .10f}")
    e_corr = float(np.dot(qw, per_q))
    if return_per_q:
        return e_corr, per_q          # per_q is UNWEIGHTED Ec(Q)
    return e_corr


def ri_rpa_ecorr(mf, auxbasis='weigend', nw=PBC_RPA_NFREQ, coulG_fn=None, mesh=None,
                 grid='gauss_legendre', verbose=False, check_support=True,
                 beta=None, flags=None):
    """Direct-RPA correlation energy per unit cell for a KRHF/KRKS mean field.

    coulG_fn(cell, q, Gv) -> v(q+G): the (raw, unweighted) Coulomb kernel.
    Default is the bare pyscf get_coulG; pass a damped kernel for AUTO damping.
    flags: a dict that receives the numerical flags, as in
    `ri_rpa_ecorr_from_dfints`.
    """
    kpts = mf.kpts
    cell, coulG_fn = kernel_cell(mf.cell, kpts, coulG_fn, mesh, check_support)
    nkpts = len(kpts)
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_energy = mf.mo_energy
    if not is_integer_occupation(np.asarray(mf.mo_occ)):
        raise ValueError(
            "ri_rpa_ecorr builds Pi from a fixed occ/virt split "
            "(build_Pi_all), which is not defined for fractional occupations: "
            "the split is an artifact of the integer collapse and changes in "
            "STEPS as the system deforms, so the correlation energy comes out "
            "discontinuous. Build the integrals first and use "
            "ri_rpa_ecorr_from_dfints, which dispatches to the "
            "occupation-weighted response.")
    nocc = cell.nelectron // 2
    aux = make_auxcell(cell, auxbasis)
    if beta is None:
        beta = beta_from_mf(mf)
    freqs, wts = build_freq_grid(mo_energy, nocc, nw, grid=grid, beta=beta,
                                 mo_occ=np.asarray(mf.mo_occ))

    Pi_all = build_Pi_all(cell, aux, mo_coeff, mo_energy, nocc, kpts,
                          coulG_fn, freqs)
    naux = aux.nao_nr()
    e_corr = 0.0
    for iq in range(nkpts):
        ec_q = 0.0
        for w in range(nw):
            Pi = Pi_all[iq][w]
            logdet = dielectric_log_determinant(np.eye(naux) - Pi, (int(iq), int(w)), flags)
            ec_q += wts[w] / (2 * np.pi) * (logdet + np.trace(Pi).real)
        if verbose:
            print(f"  q={iq:3d}  Ec(q) = {ec_q.real: .10f}")
        e_corr += ec_q.real
    return e_corr / nkpts
