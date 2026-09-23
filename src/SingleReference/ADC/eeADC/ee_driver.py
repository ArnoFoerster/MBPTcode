"""Front end for the polarization-propagator (electronic-excitation) ADC
route, mirroring adc_u_driver's split: state and routing here, physics in the
ee_* route modules.

    e, Z = solve_ee_adc(mf, level='adc3', nroots=5)              # spin-free
    e, Z = solve_ee_adc(mf, level='adc3', spin='triplet')
    e, Z = solve_ee_adc(mf, level='adc3', df=True)               # DF (production)
    e, Z = solve_ee_adc(mf, level='adc3', route='spinorbital')   # arbiter

Routes:
  'spinfree'    -- spatial-orbital tensors (ee_r_sigma / ee_r_sigma_df); this
                   is the production path. Both spin channels come out of one
                   operator; `spin` projects onto one via the alpha<->beta
                   involution.
  'spinorbital' -- ee_u_dense_full / ee_u_sigma_full, the validated arbiter.
                   An RHF reference yields singlets AND triplets interleaved;
                   `spin` there goes through the numerical CSF isometry.

level in ('adc1', 'adc2', 'adc2x', 'adc3').
"""
import numpy as np
from pyscf import scf as _scf

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_uhf_spin_orbital_arrays_blockstacked,
    DFIntegrals)
from src.SingleReference.ADC.eeADC import (ee_u_dense_full, ee_u_sigma_full,
                                     ee_r_sigma, ee_r_sigma_df, ee_utils)
from src.Solvers.davidson import davidson

DENSE_LIMIT = 2000      # below this, Davidson's subspace goes linearly
                        # dependent long before it converges; a 59-dim
                        # open-shell sector hit exactly that

_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}
_PROJECT_SHIFT = 1.0e4          # pushes the unwanted channel out of the way


def spin_orbital_arrays(mf, mol=None):
    """(eps, g_anti, nocc) in an occupied-first spin-orbital layout.

    RHF: interleaved alpha/beta, which is occupied-first because the spatial
    orbitals are energy-ordered. UHF: BLOCK-STACKED
    [occ_a, occ_b, virt_a, virt_b], which is the only ordering that keeps the
    occ/virt split contiguous when nocc_a != nocc_b -- and a contiguous split
    is exactly what the ee_u_* modules slice on.

    The ADC equations themselves are spin-orbital and make no closed-shell
    assumption beyond a canonical reference (f diagonal), which UHF satisfies
    in its own spin-orbital basis. So the open-shell route is the same code,
    fed different arrays; tests/test_ee_adc_openshell.py pins that by running
    a closed-shell molecule through BOTH and requiring the same spectrum."""
    mol = mol if mol is not None else mf.mol
    if isinstance(mf, _scf.uhf.UHF):
        return get_uhf_spin_orbital_arrays_blockstacked(mol, mf)
    eps = np.repeat(get_orbital_energies(mf, representation='spatial'), 2)
    g = get_antisymmetrized_spin_eri(
        get_two_electron_integrals_chemist(mol, mf, representation='spatial'))
    return eps, g, mol.nelectron


def solve_ee_adc(mf, mol=None, level='adc3', nroots=5, route='spinfree',
                 df=False, spin=None, matrix_free=True, conv_tol=1e-8,
                 en_dress=None, frozen=0, auxbasis=None,
                 ms_sector='auto', max_subspace=None):
    """(e, Z): the lowest `nroots` excitation energies and their vectors.

    en_dress: Epstein-Nesbet channel dict (ee_en); True means the standard
    hole-hole + particle-particle dressing. It dresses the DOUBLES amplitude
    denominators only -- the singles amplitude and the supermatrix keep their
    MP zeroth order, matching the IP/EA side's en_dress convention."""
    mol = mol if mol is not None else mf.mol
    if spin is not None and spin not in _CHANNEL:
        raise ValueError(f"spin={spin!r}; expected 'singlet', 'triplet' or None")
    from src.SingleReference.ADC.eeADC.ee_en import validate_dress
    en_dress = validate_dress(en_dress)
    if en_dress is not None and level == 'adc1':
        raise ValueError('ADC(1) carries no amplitudes, so en_dress has '
                         'nothing to dress')
    if route == 'unrestricted':
        return _solve_unrestricted(mf, mol, level, nroots, matrix_free,
                                   conv_tol, en_dress, frozen, auxbasis, df)
    if route == 'spinorbital':
        if df:
            # The dense spin-orbital route is the ARBITER -- it forms the full
            # antisymmetrized <pq||rs> over spin orbitals and is what the DF
            # paths are validated against. Density fitting an oracle defeats
            # its purpose, so DF open-shell work goes to route='unrestricted'
            # (ee_u_r_sigma_df), which this route in turn validates.
            raise ValueError('the spin-orbital arbiter route has no DF path; '
                             "use route='unrestricted' for an open-shell "
                             "reference, route='spinfree' for a closed shell")
        if frozen:
            raise NotImplementedError(
                'frozen core is wired on the spin-free route only')
        return _solve_spin_orbital(mf, mol, level, nroots, spin, matrix_free,
                                   conv_tol, en_dress, ms_sector)
    if route != 'spinfree':
        raise ValueError(f"route={route!r}; expected 'spinfree', "
                         "'unrestricted' or 'spinorbital'")
    return _solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                            conv_tol, en_dress, frozen, auxbasis, max_subspace)


def _solve_unrestricted(mf, mol, level, nroots, matrix_free, conv_tol,
                        en_dress=None, frozen=0, auxbasis=None, df=True):
    """Open-shell production route: unrestricted DF sigma, Delta-Ms = 0.

    The sector is enforced by the vector layout rather than by a mask -- there
    is no slot for a spin-flip single -- so unlike the dense spin-orbital
    route this one needs no ms_sector argument."""
    from src.SingleReference.ADC.eeADC import ee_u_r_sigma_df as _u
    if not isinstance(mf, _scf.uhf.UHF):
        raise ValueError("route='unrestricted' needs a UHF reference; use "
                         "route='spinfree' for a closed shell")
    if not df:
        raise ValueError("route='unrestricted' is a density-fitted route; "
                         "for dense open-shell integrals use "
                         "route='spinorbital'")
    eps_a, eps_b = (np.asarray(e) for e in mf.mo_energy)
    no_a, no_b = mf.nelec
    # frozen core: the dropped orbitals enter only through the (diagonal)
    # Fock matrix, so the active problem is the plain slice -- same
    # convention as the spin-free route, applied to each spin separately.
    act_a = slice(frozen, len(eps_a))
    act_b = slice(frozen, len(eps_b))
    eps_a, eps_b = eps_a[act_a], eps_b[act_b]
    no_a, no_b = no_a - frozen, no_b - frozen
    if auxbasis == 'exact':
        dfi = DFIntegrals.from_scf(mol, mf, exact=True)
    else:
        src = mf if (auxbasis is None and getattr(mf, 'with_df', None)) \
            else mf.density_fit(auxbasis=auxbasis)
        dfi = DFIntegrals.from_scf(mol, src)
    Ba = dfi.B_aa[:, act_a, act_a]
    Bb = dfi.B_bb[:, act_b, act_b]
    aop, diag, d = _u.build_operator(eps_a, eps_b, Ba, Bb, no_a, no_b,
                                     level=level, en_dress=en_dress)
    n = d['nH']
    if not matrix_free or n <= DENSE_LIMIT:
        H = np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])
        e, Z = np.linalg.eigh(0.5 * (H + H.T))
        return e[:nroots], Z[:, :nroots]
    e, Z = davidson(aop, diag, k=nroots, tol=conv_tol)
    return np.asarray(e), np.asarray(Z)


def _solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                     conv_tol, en_dress=None, frozen=0, auxbasis=None,
                     max_subspace=None):
    """max_subspace caps the Davidson subspace. It is a MEMORY knob, and at
    scale the dominant one: the doubles vector is no^2 nv^2, so naphthalene at
    aug-cc-pVTZ carries 2.5 GB per trial vector and the default subspace of
    6k+20 would need hundreds of GB of Krylov space alone."""
    if isinstance(mf, _scf.uhf.UHF):
        raise ValueError(
            "route='spinfree' is a closed-shell construction (spatial spin "
            "blocks, singlet/triplet channels); use route='spinorbital' for "
            'a UHF reference')
    eps = get_orbital_energies(mf, representation='spatial')
    no = mol.nelectron // 2
    # frozen core: the dropped orbitals enter only through the Fock matrix,
    # which is already diagonal in eps, so the active problem is the plain
    # slice -- the same convention the benchmark protocols use.
    act = slice(frozen, len(eps))
    eps = eps[act]
    no -= frozen
    nv = len(eps) - no
    if df:
        if auxbasis == 'exact':
            # eigendecomposed B, naux = norb^2: a plumbing check that the DF
            # kernels reproduce the dense route exactly, NOT a production
            # setting -- it is far slower than dense integrals
            B = DFIntegrals.from_scf(mol, mf, exact=True).B_aa[:, act, act]
        else:
            src = mf if (auxbasis is None and getattr(mf, 'with_df', None)) \
                else mf.density_fit(auxbasis=auxbasis)
            B = DFIntegrals.from_scf(mol, src).B_aa[:, act, act]
        aop, diag, dims = ee_r_sigma_df.build_operator(
            eps, B, no, level=level, en_dress=en_dress)
    else:
        V = get_two_electron_integrals_chemist(
            mol, mf, representation='spatial')[act, act, act, act
                                               ].transpose(0, 2, 1, 3)
        aop, diag, dims = ee_r_sigma.build_operator(
            eps, V, no, level=level, en_dress=en_dress)

    n = dims['nH']
    v0 = None
    if spin is not None:
        sgn = _CHANNEL[spin]

        def project(v):
            return 0.5 * (v + sgn * ee_r_sigma.spin_flip_vector(
                np.asarray(v).ravel(), no, nv, level))

        raw = aop

        def aop(v):
            v = np.asarray(v).ravel()
            p = project(v)
            # the complement is shifted far away rather than left at zero, so
            # a lowest-root Davidson cannot converge onto the other channel
            return raw(p) + _PROJECT_SHIFT * (v - p)

        # seed inside the channel: a raw unit vector is half in the shifted
        # complement, which costs the Davidson several iterations to shed
        v0 = _channel_guess(diag, project, n, nroots)

    if not matrix_free:
        H = np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])
        e, Z = np.linalg.eigh(0.5 * (H + H.T))
        return e[:nroots], Z[:, :nroots]
    e, Z = davidson(aop, diag, k=nroots, v0=v0, tol=conv_tol,
                    max_subspace=max_subspace)
    return np.asarray(e), np.asarray(Z)


def _channel_guess(diag, project, n, nroots, pool=4):
    """Orthonormal starting vectors already inside the requested spin
    channel: project the lowest-diagonal unit vectors and re-orthonormalize,
    dropping anything the projector annihilated."""
    order = np.argsort(diag)[:max(pool * nroots, nroots)]
    cols = []
    for k in order:
        e = np.zeros(n); e[k] = 1.0
        cols.append(project(e))
    Q, R = np.linalg.qr(np.column_stack(cols))
    keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    return Q[:, :nroots] if Q.shape[1] >= nroots else None


def _solve_spin_orbital(mf, mol, level, nroots, spin, matrix_free,
                        conv_tol, en_dress=None, ms_sector='auto'):
    eps, g, nocc = spin_orbital_arrays(mf, mol)
    norb = len(eps)
    is_uhf = isinstance(mf, _scf.uhf.UHF)
    if ms_sector == 'auto':
        # an open-shell reference needs the spin-conserving sector imposed;
        # for a closed shell the other sectors only duplicate triplet Ms
        # partners, which is harmless and kept for backwards compatibility
        ms_sector = 0 if is_uhf else None
    if spin is not None and is_uhf:
        raise ValueError(
            'singlet/triplet projection needs a closed-shell reference; an '
            'open-shell spectrum is not spin-separable that way -- use '
            'spin=None and characterize the roots by <S^2>')
    if spin is not None:
        from src.SingleReference.ADC.eeADC.ee_spin_adapt import csf_isometry
        T = csf_isometry(nocc, len(eps), spin=spin, level=level)
        H = ee_u_dense_full.build_supermatrix(eps, g, nocc, level=level,
                                              en_dress=en_dress)
        e, Z = np.linalg.eigh(T.T @ (H @ T))
        return e[:nroots], Z[:, :nroots]
    mask = None
    if ms_sector is not None:
        sz = ee_utils.spin_labels(mf, nocc, norb)
        mask = ee_utils.ms_sector_mask(sz, nocc, norb, ms_sector)
        if level == 'adc1':
            mask = mask[:ee_utils.dimensions(nocc, norb)['n_s']]

    dim = int(mask.sum()) if mask is not None else \
        ee_utils.dimensions(nocc, norb)['nH' if level != 'adc1' else 'n_s']
    if not matrix_free or dim <= DENSE_LIMIT:
        H = ee_u_dense_full.build_supermatrix(eps, g, nocc, level=level,
                                              en_dress=en_dress)
        if mask is None:
            e, Z = np.linalg.eigh(H)
            return e[:nroots], Z[:, :nroots]
        e, Z_sub = np.linalg.eigh(H[np.ix_(mask, mask)])
        Z = np.zeros((len(mask), Z_sub.shape[1]))
        Z[mask, :] = Z_sub          # back to the full configuration basis
        return e[:nroots], Z[:, :nroots]

    aop, diag, _ = ee_u_sigma_full.build_operator(eps, g, nocc, level=level,
                                                  en_dress=en_dress)
    if mask is not None:
        # The Ms sector is a COORDINATE subspace (a subset of configurations),
        # so restrict rather than shift: build the operator on the selected
        # indices only. Shifting the complement to a large value instead makes
        # the Davidson subspace matrix span ~1e4 against ~0.1 and it fails to
        # converge. (The singlet/triplet projection elsewhere is NOT a
        # coordinate subspace -- it mixes the two singles blocks -- so the
        # shift is the right tool there.)
        raw, keep = aop, mask
        n_full = len(mask)

        def aop_sub(v_sub):
            v = np.zeros(n_full)
            v[keep] = np.asarray(v_sub).ravel()
            return raw(v)[keep]

        e, Z_sub = davidson(aop_sub, diag[keep], k=nroots, tol=conv_tol)
        Z_sub = np.asarray(Z_sub)
        if Z_sub.ndim == 1:
            Z_sub = Z_sub[:, None]
        Z = np.zeros((n_full, Z_sub.shape[1]))
        Z[keep, :] = Z_sub          # back to the full configuration basis
        return np.asarray(e), Z
    e, Z = davidson(aop, diag, k=nroots, tol=conv_tol)
    return np.asarray(e), np.asarray(Z)
