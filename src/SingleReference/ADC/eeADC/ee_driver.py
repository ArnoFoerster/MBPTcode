"""The routes behind the electronic-excitation ADC front end,
src.SingleReference.ADC.eeADC.solve.solve_ee_adc (see its docstring): state
and routing here, physics in the ee_* route modules, one dense-or-Davidson
eigensolve (_eigensolve) for every route but the fold.

Every route returns an EEResult: the tuple it always returned, plus
.converged, one flag per root.
"""

import contextlib

import numpy as np
from pyscf import scf as _scf
from pyscf.lib import logger

from src.Base.pyscf.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_uhf_spin_orbital_arrays_blockstacked,
    DFIntegrals)
from src.SingleReference.ADC.eeADC import (ee_u_dense_full, ee_u_sigma_full,
                                     ee_r_sigma, ee_r_sigma_df, ee_utils,
                                     ee_ri_density, ee_state_specific,
                                     ee_fold, ee_ri_fold, ee_u_r_sigma_df)
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.SingleReference.ADC.eeADC.ee_spin_adapt import channel_basis, csf_isometry
from src.Base.numerics.constants import (ADC2_SPIN_SCALING, EE_ADC_DENSE_LIMIT,
                                         EE_FOLD_RESIDUAL_GAP)
from src.Base.environment_seam import attached_environment, resolve_environment
from src.Base.utils.parallel import memory
from src.Base.utils.linear_algebra.diagonalization import eigh_symmetric
from src.Base.utils.parallel.mpi_grid import current_comm
from src.Solvers.davidson import diagonal_seeds, solve_symmetric

CHANNELS = {'singlet': +1.0, 'triplet': -1.0}


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


SS_DENSITIES = ('unrelaxed', 'relaxed', 'relaxed+pcm')


def reaction_field_environment(mf, mol, environment, route):
    """The continuum whose optical reaction field dresses the singles, or None.

    An environment that does not respond (none, point charges) acts through
    the mean field alone and needs nothing here. One that screens must be a
    surface-charge continuum, the route a closed-shell spin-free one, and the
    mean field relaxed in that continuum's eps_static: without it the result
    would be the frozen-polarization half of the shift, not the solvated
    excitation."""
    env = resolve_environment(environment, mf)
    if not getattr(env, 'screens', False):
        return None
    if not hasattr(env, 'ov_grid_potential'):
        raise NotImplementedError(
            f'{env!r}: the EE-ADC routes take a surface-charge continuum '
            '(SolventScreening); this environment has no transition '
            'reaction field here')
    if route != 'spinfree' or isinstance(mf, _scf.uhf.UHF):
        raise NotImplementedError(
            'a continuum is wired on the restricted spin-free routes only')
    if getattr(env, 'eps_static', None) is not None:
        with_solvent = getattr(mf, 'with_solvent', None)
        if with_solvent is None:
            raise ValueError(
                'the continuum has an eps_static but the mean field was not '
                'relaxed in it; build it with env.mean_field(mol, factory)')
        env.check_ground_state_continuum(with_solvent)
    return env


def _bare(mf):
    """A context in which mf hands out the bare interaction: an attached
    environment is taken off for the length of the block. The continuum
    enters the EE-ADC routes through the singles term alone."""
    if getattr(mf, 'with_screening', None) is None:
        return contextlib.nullcontext(mf)
    return attached_environment(mf, None)


def spin_scaling_factors(spin_scaling):
    """(c_os, c_ss) of a spin_scaling argument, or None for ADC(2)."""
    if spin_scaling is None:
        return None
    if isinstance(spin_scaling, str):
        if spin_scaling not in ADC2_SPIN_SCALING:
            raise ValueError(f'spin_scaling={spin_scaling!r}; expected one of '
                             f'{sorted(ADC2_SPIN_SCALING)} or (c_os, c_ss)')
        return ADC2_SPIN_SCALING[spin_scaling]
    c_os, c_ss = (float(c) for c in spin_scaling)
    return c_os, c_ss


def solve_folded_ri(mf, mol, level, nroots, spin, conv_tol, en_dress, frozen,
                    auxbasis, dense_limit, return_fold, env, tile_gb, route,
                    numerics=None, state_specific='off', ss_density='unrelaxed',
                    scaling=None, relaxed_density=False):
    """fold=True: the density-fitted tiled fold of ee_ri_fold."""
    if route != 'spinfree' or isinstance(mf, _scf.uhf.UHF):
        raise ValueError('fold=True is a closed-shell spin-free route')
    if level not in ee_ri_fold.LEVELS:
        raise ValueError(f'fold=True takes level in {ee_ri_fold.LEVELS}; '
                         f'level={level!r}')
    if spin not in CHANNELS:
        raise ValueError("fold=True solves one channel: spin='singlet' or "
                         "'triplet'")
    if en_dress is not None:
        raise ValueError('fold=True has no Epstein-Nesbet dressing')
    if auxbasis == 'exact':
        raise ValueError("fold=True takes a real auxiliary basis, not 'exact'")
    with _bare(mf):
        # over MPI ranks each keeps B_vv in its own auxiliary chunks only
        factors = ee_ri_fold.bare_factors(mol, mf, frozen=frozen,
                                          auxbasis=auxbasis, comm=current_comm())
    rf = (None if env is None else
          ee_ri_fold.TransitionReactionField.from_environment(env, mol, mf,
                                                              frozen))
    # over MPI ranks the operator's tiles are shared out and rank 0 alone
    # iterates (ee_ri_fold.drive); serially this is the plain solve
    op = ee_ri_fold.RIFoldADC2(factors, spin, level=level, reaction_field=rf,
                               tile_gb=tile_gb, numerics=numerics, scaling=scaling)

    def solve(o):
        if level == 'adc2':
            # the roots' error goes as the residual squared
            r = ee_fold.solve_folded(o.pieces(), nroots, spin=spin, tol_omega=conv_tol,
                                     tol_residual=max(conv_tol, np.sqrt(
                                         conv_tol * EE_FOLD_RESIDUAL_GAP)),
                                     dense_limit=dense_limit, solver='block',
                                     verbose=int(mol.verbose >= logger.INFO))
        else:
            r = _solve_cis_ri(o, nroots, conv_tol, dense_limit)
        return dict(omega=r.omega, y=r.y, t1=r.t1, steps=r.steps,
                    loop=list(r.loop), converged=r.converged, level=r.level,
                    flags=list(getattr(r, 'flags', [])))

    out = ee_ri_fold.drive(op, solve)
    embed, _, _ = channel_basis(2 * op.n, op.no, op.nv, 'adc1',
                                CHANNELS[spin])
    res = ee_fold.FoldResult(out['omega'], out['y'], out['t1'], out['steps'],
                             out['loop'], out['converged'], embed, out['level'],
                             out.get('flags'))
    Z = np.column_stack([embed(res.x1[:, k]) for k in range(res.omega.size)])
    res.operator = op
    e = res.omega
    if state_specific != 'off':
        _state_specific(res, op, env, mol, mf, frozen, state_specific, tile_gb,
                        ss_density, auxbasis)
        e = res.omega_ss
    elif relaxed_density:
        res.delta_density_relaxed = _relaxed_densities(res, op, env, mol, mf,
                                                       frozen, tile_gb, False,
                                                       auxbasis)
    return EEResult((e, Z, res) if return_fold else (e, Z), res.converged)


def _relaxed_densities(res, op, env, mol, mf, frozen, tile_gb, pcm_response,
                       auxbasis):
    """Each root's orbital-relaxed difference density (ee_relaxed_density),
    full MO basis; a collective over op's ranks."""
    with _bare(mf):
        Bc = erd.core_factors(mol, mf, frozen, auxbasis) if frozen else None
        return [erd.relaxed_density(
                    erd.tiled_pieces(op, res.y[:, k], res.omega[k], tile_gb),
                    op, mf, mol, frozen, res.t1[k], Bc=Bc, env=env,
                    pcm_response=pcm_response)
                for k in range(res.omega.size)]


def _active_orbitals(mol, mf, frozen, op):
    """The active MO coefficients (nao, no + nv), checked against the operator."""
    nocc = mol.nelectron // 2
    mo_act = np.asarray(mf.mo_coeff)[:, frozen:]
    if mo_act.shape[1] != op.no + op.nv or nocc - frozen != op.no:
        raise RuntimeError('the active orbitals do not match the operator')
    return mo_act


def _state_specific(res, op, env, mol, mf, frozen, mode, tile_gb,
                    ss_density='unrelaxed', auxbasis=None):
    """res with the roots' difference densities and state-specific shifts
    (ee_state_specific); a collective over op's ranks, every rank holding the
    same roots after ee_ri_fold.drive."""
    res.delta_density = [
        ee_ri_density.difference_density(op, res.y[:, k], res.omega[k],
                                         res.t1[k], tile_gb=tile_gb)
        for k in range(res.omega.size)]
    modes = ee_state_specific.shift_modes(env, mode)
    mo_act = _active_orbitals(mol, mf, frozen, op)
    res.ss_shifts = ee_state_specific.state_specific_energies(
        env, mol, mo_act, res.delta_density, modes)
    res.ss_density = ss_density
    if ss_density != 'unrelaxed':
        res.delta_density_relaxed = _relaxed_densities(
            res, op, env, mol, mf, frozen, tile_gb,
            ss_density == 'relaxed+pcm', auxbasis)
        C = np.asarray(mf.mo_coeff)
        res.delta_dipole = ee_state_specific.delta_dipoles(
            mol, mo_act, C, res.delta_density, res.delta_density_relaxed)
        res.ss_shifts_unrelaxed = res.ss_shifts
        res.ss_shifts = ee_state_specific.state_specific_energies(
            env, mol, C, res.delta_density_relaxed, modes)
    res.ss_mode = mode
    res.ss_shift = res.ss_shifts[mode]
    res.omega_ss = res.omega + res.ss_shift


def _solve_cis_ri(op, nroots, conv_tol, dense_limit):
    """ADC(1) on the tiled operator: no doubles, one eigenproblem, as a
    FoldResult with unit singles weights."""
    embed, _, _ = channel_basis(2 * op.n, op.no, op.nv, 'adc1',
                                CHANNELS[op.spin])
    mv = ee_ri_fold.FoldedMatvec(op, None)
    dense = op.n <= dense_limit
    kw = {} if dense else {'x0': _open_seeds(op.diag, nroots, None)}
    e, X, conv = _eigensolve(mv, op.diag, nroots, conv_tol, dense,
                             label='ee-ADC(1) (tiled)', **kw)
    k = len(e)
    return ee_fold.FoldResult(np.asarray(e), np.asarray(X), np.ones(k),
                              np.ones(k, int), ['index'] * k,
                              np.asarray(conv, bool), embed, np.arange(k))


def solve_unrestricted(mf, mol, level, nroots, matrix_free, conv_tol,
                       en_dress=None, frozen=0, auxbasis=None, df=True,
                       dense_limit=EE_ADC_DENSE_LIMIT, threshold=5000):
    """Open-shell production route: unrestricted DF sigma, Delta-Ms = 0.

    The sector is enforced by the vector layout rather than by a mask -- there
    is no slot for a spin-flip single -- so unlike the dense spin-orbital
    route this one needs no ms_sector argument."""
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
    aop, diag, d = ee_u_r_sigma_df.build_operator(eps_a, eps_b, Ba, Bb,
                                                  no_a, no_b, level=level,
                                                  en_dress=en_dress)
    dense = not matrix_free or d['nH'] <= dense_limit
    kw = {} if dense else {'max_memory': _subspace_memory(mf)}
    e, Z, conv = _eigensolve(aop, diag, nroots, conv_tol, dense,
                             threshold=threshold,
                             label='ee-ADC (unrestricted)', **kw)
    return EEResult((e, Z), conv)


def solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                    conv_tol, en_dress=None, frozen=0, auxbasis=None,
                    max_subspace=None, return_parity=False,
                    dense_limit=EE_ADC_DENSE_LIMIT, threshold=5000, env=None):
    """max_subspace caps the Davidson subspace. It is a MEMORY knob, and at
    scale the dominant one: the doubles vector is no^2 nv^2, the solver holds
    the operator's image alongside the subspace (so TWICE that per trial
    vector), and in an augmented triple-zeta basis the default 10*nroots+30
    outgrows any memory.

    Turning it down is not free: the solver restarts by collapsing onto the
    nroots Ritz vectors, so a tight subspace is paid for in matrix-vector
    products and, tight enough, in never converging at all (it says so).
    Below 3*nroots+4 it warns."""
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
    with _bare(mf):
        aop, diag, dims = _spin_free_operator(mf, mol, level, df, spin,
                                              en_dress, act, no, eps,
                                              auxbasis)
    if env is not None:
        aop, diag = _with_reaction_field(aop, diag, env, mol, mf, frozen, no,
                                         nv)
    return _spin_free_solve(aop, diag, dims, no, nv, level, nroots, spin,
                            matrix_free, conv_tol, max_subspace,
                            return_parity, dense_limit, threshold, mf)


def _with_reaction_field(aop, diag, env, mol, mf, frozen, no, nv):
    """aop and diag with the singles block's transition reaction field,
    (ia|vtilde|jb) between every pair of spin-conserving singles: a spin-
    orbital excitation's transition density is a charge density whichever its
    spin, so the alpha and beta singles blocks couple alike (the triplet
    combination cancels)."""
    rf = ee_ri_fold.TransitionReactionField.from_environment(env, mol, mf,
                                                            frozen)
    n_s = no * nv
    raw = aop

    def dressed(vec):
        vec = np.asarray(vec).ravel()
        out = np.array(raw(vec), dtype=float).ravel()
        t = rf.apply((vec[:n_s] + vec[n_s:2 * n_s])[:, None])[:, 0]
        out[:n_s] += t
        out[n_s:2 * n_s] += t
        return out

    d = np.array(diag, dtype=float)
    rd = rf.diagonal()
    d[:n_s] += rd
    d[n_s:2 * n_s] += rd
    return dressed, d


def _spin_free_operator(mf, mol, level, df, spin, en_dress, act, no, eps,
                        auxbasis):
    """(aop, diag, dims) of the spin-free route on the active orbitals."""
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
        # a channel solve hands aop flip eigenvectors only (embed below)
        aop, diag, dims = ee_r_sigma_df.build_operator(
            eps, B, no, level=level, en_dress=en_dress,
            parity=CHANNELS.get(spin))
    else:
        V = get_two_electron_integrals_chemist(
            mol, mf, representation='spatial')[act, act, act, act
                                               ].transpose(0, 2, 1, 3)
        aop, diag, dims = ee_r_sigma.build_operator(
            eps, V, no, level=level, en_dress=en_dress,
            parity=CHANNELS.get(spin))
    return aop, diag, dims


def _spin_free_solve(aop, diag, dims, no, nv, level, nroots, spin,
                     matrix_free, conv_tol, max_subspace, return_parity,
                     dense_limit, threshold, mf):
    """Solve the spin-free operator: dense below dense_limit, else Davidson;
    one spin channel in its flip-pair basis when spin is given."""
    n = dims['nH']
    embed = restrict = None
    if spin is not None:
        # one channel: solve in its flip-pair basis, where it is a coordinate
        # subspace; shifting the other channel away instead stalls the
        # Davidson, as the Ms-sector note in solve_spin_orbital records
        embed, restrict, reps = channel_basis(n, no, nv, level, CHANNELS[spin])
        diag, n = diag[reps], len(reps)
        if n < nroots:
            raise ValueError(f'the {spin} channel holds {n} states, fewer than '
                             f'nroots={nroots}: lower nroots')
    dense = not matrix_free or n <= dense_limit
    kw = {} if dense else {'x0': _open_seeds(diag, nroots, max_subspace),
                           'max_subspace': max_subspace,
                           'max_memory': _subspace_memory(mf)}
    e, Z, conv = _eigensolve(aop, diag, nroots, conv_tol, dense, embed,
                             restrict, threshold=threshold,
                             label='ee-ADC (spin-free)', **kw)
    if Z is None:
        # a distributed dense solve: the eigenvectors are rank 0's alone
        return EEResult((e, None, None) if return_parity else (e, None), conv)
    if not return_parity:
        return EEResult((e, Z), conv)
    parity = np.array([Z[:, k] @ ee_r_sigma.spin_flip_vector(Z[:, k], no, nv, level)
                       for k in range(Z.shape[1])])
    return EEResult((e, Z, parity), conv)


def _open_seeds(diag, nroots, max_subspace):
    """(n, m) start vectors: solve_symmetric's own diagonal seeds, each with a
    random component of norm 1e-2.

    The operator commutes with the molecule's point group, so a search from
    unit vectors stays inside the symmetry species they carry and returns a
    higher root in place of a low state of another species (benzene's
    degenerate third singlet in cc-pVDZ at nroots=3). The admixture reaches
    every species; the fixed rng keeps the solve reproducible. m follows
    solve_symmetric's own seed count, so the seeds never crowd a small
    subspace."""
    n = len(diag)
    cap = n if max_subspace is None else max_subspace // 2
    X = np.column_stack(diagonal_seeds(diag, min(n, 2 * nroots + 4,
                                                 max(nroots, cap))))
    R = np.random.default_rng(0).standard_normal(X.shape)
    return X + 1e-2 * R / np.linalg.norm(R, axis=0)


def _subspace_memory(mf):
    """MB left for the Davidson subspace: mf.max_memory is the budget of the
    whole process (pyscf's convention), less what it holds after the operator
    build. Zero sends the subspace to disk."""
    return max(0, mf.max_memory - memory.current_memory_mb())



def solve_spin_orbital(mf, mol, level, nroots, spin, matrix_free,
                       conv_tol, en_dress=None, ms_sector='auto',
                       dense_limit=EE_ADC_DENSE_LIMIT, threshold=5000):
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

    def supermatrix():
        return ee_u_dense_full.build_supermatrix(eps, g, nocc, level=level,
                                                 en_dress=en_dress)

    if spin is not None:
        # the CSF basis: the eigenvectors stay in it
        T = csf_isometry(nocc, len(eps), spin=spin, level=level)
        e, Z, conv = _eigensolve(None, None, nroots, conv_tol, True,
                                 matrix=lambda: T.T @ (supermatrix() @ T),
                                 threshold=threshold)
        return EEResult((e, Z), conv)
    mask = None
    if ms_sector is not None:
        sz = ee_utils.spin_labels(mf, nocc, norb)
        mask = ee_utils.ms_sector_mask(sz, nocc, norb, ms_sector)
        if level == 'adc1':
            mask = mask[:ee_utils.dimensions(nocc, norb)['n_s']]

    dim = int(mask.sum()) if mask is not None else \
        ee_utils.dimensions(nocc, norb)['nH' if level != 'adc1' else 'n_s']
    dense = not matrix_free or dim <= dense_limit
    # The Ms sector is a COORDINATE subspace (a subset of configurations), so
    # restrict rather than shift: solve on the selected indices only. Shifting
    # the complement to a large value instead makes the Davidson subspace
    # matrix span ~1e4 against ~0.1 and it fails to converge. (The spin-free
    # singlet/triplet channel is not a coordinate subspace of the
    # configurations either, but it is one of the flip-pair basis, which is
    # how channel_basis restricts it.)
    embed, restrict = (None, None) if mask is None else _coordinate_basis(mask)
    if dense:
        aop = diag = kw = None
        matrix = (supermatrix if mask is None else
                  lambda: supermatrix()[np.ix_(mask, mask)])
    else:
        aop, diag, _ = ee_u_sigma_full.build_operator(eps, g, nocc, level=level,
                                                      en_dress=en_dress)
        diag = diag if mask is None else diag[mask]
        matrix = None
        kw = {'max_memory': _subspace_memory(mf)}
    label = ('ee-ADC (spin-orbital)' if mask is None else
             'ee-ADC (spin-orbital, Ms sector)')
    e, Z, conv = _eigensolve(aop, diag, nroots, conv_tol, dense, embed,
                             restrict, matrix=matrix, threshold=threshold,
                             label=label, **(kw or {}))
    return EEResult((e, Z), conv)


def _coordinate_basis(keep):
    """(embed, restrict) of the coordinate subspace a boolean mask keeps."""
    n = len(keep)

    def embed(u):
        v = np.zeros(n)
        v[keep] = u
        return v

    def restrict(v):
        return v[keep]

    return embed, restrict


# ----------------------------------------------------------------------
# the one eigensolve and its result
# ----------------------------------------------------------------------

class EEResult(tuple):
    """What a route returns: the tuple it always returned, (e, Z), (e, Z,
    parity) or (e, Z, FoldResult), so `e, Z = solve_ee_adc(...)` unpacks as
    before; plus .converged, one bool per root of e. A dense solve converges
    every root; a Davidson root that did not converge also warned
    (solve_symmetric), and the fold's flags are FoldResult.converged."""

    def __new__(cls, items, converged):
        out = super().__new__(cls, items)
        out.converged = np.asarray(converged, bool)
        return out

    def __getnewargs__(self):
        return tuple(self), self.converged


def _dense_matrix(aop, n):
    """The (n, n) matrix of aop from its n unit-vector products, symmetrized;
    one unit vector, reset after each product."""
    H = np.empty((n, n))
    u = np.zeros(n)
    for k in range(n):
        u[k] = 1.0
        H[:, k] = aop(u)
        u[k] = 0.0
    return 0.5 * (H + H.T)


def _eigensolve(aop, diag, nroots, conv_tol, dense, embed=None, restrict=None,
                matrix=None, threshold=5000, label='ee-ADC', **davidson):
    """(e, Z, conv): the lowest nroots eigenpairs of the symmetric operator
    aop, dense (eigh_symmetric, distributed above threshold) or by Davidson
    (solve_symmetric, conv_tol its residual tolerance).

    embed, restrict: a subspace to solve in (a spin channel's flip-pair basis,
    an Ms sector's configurations), restrict(embed(u)) == u: the solve acts
    with restrict(aop(embed(u))) on diag, the diagonal in that basis, and Z
    comes back embedded in the full space. matrix: a callable giving the dense
    matrix in that basis, used instead of the unit-vector products of aop.
    davidson: x0, max_subspace, max_memory for solve_symmetric.

    conv has one flag per root, all True for a dense solve. Z is None on the
    ranks that do not own a distributed dense solve's eigenvectors."""
    if embed is not None and aop is not None:
        raw = aop

        def aop(u):
            return restrict(np.asarray(raw(embed(np.asarray(u).ravel()))).ravel())

    if dense:
        H = matrix() if matrix is not None else _dense_matrix(aop, len(diag))
        e, Z, _ = eigh_symmetric(H, threshold=threshold)
        e = e[:nroots]
        Z = None if Z is None else Z[:, :nroots]
        conv = np.ones(len(e), bool)
    else:
        e, Z, conv = solve_symmetric(aop, diag, nroots=nroots,
                                     tol_residual=conv_tol, label=label,
                                     **davidson)
        e, Z = np.asarray(e), np.asarray(Z)
        if Z.ndim == 1:
            Z = Z[:, None]
        conv = np.asarray(conv, bool)
    if Z is not None and embed is not None:
        Z = np.column_stack([embed(Z[:, k]) for k in range(Z.shape[1])])
    return e, Z, conv
