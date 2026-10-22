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

level in ('adc1', 'adc2', 'adc2x', 'adc3', 'gf2').

    e, Z = solve_ee_adc(mf, level='adc2', spin='singlet', fold=True,
                        auxbasis='aug-cc-pvdz-ri', frozen=ncore)   # large molecules

fold=True folds the doubles into the singles problem (ee_fold) and builds
every doubles-shaped quantity a tile at a time from the DF factors
(ee_ri_fold): no array of doubles size is held, so it is the route for
molecules of a thousand basis functions. adc1 and adc2, one spin channel.

environment: a polarizable continuum (src.Base.solvent_screening), given here
or attached to the mean field. The mean field must already be relaxed in its
eps_static continuum (`env.mean_field(mol, factory)`); the excitation's fast
reaction field, at the optical constant, enters the singles block as the
linear-response term 2 (ia|vtilde|jb) of singlets (nothing for triplets),
and every second-order term keeps the bare interaction on the solvated
orbitals -- the PTE convention, ee_ri_fold's module docstring. At adc1 this is
TDA/CIS in linear-response PCM with a non-equilibrium (optical) response.

state_specific ('off', 'vertical', 'equilibrium'; fold=True): the reaction
field of each root's unrelaxed ISR-ADC(2) difference density, (1/2) dV Q dV
(ee_state_specific), added to its eigenvalue: at eps_inf with the ground
state's slow polarization frozen (vertical), or at eps_static (equilibrium);
the LR term stays at eps_inf either way. Triplets, which have no LR term,
take their whole response to the excitation from it.
"""

import numpy as np
from pyscf import scf as _scf
from pyscf.lib import logger

from src.Base.pyscf_interface import (
    get_orbital_energies, get_two_electron_integrals_chemist,
    get_antisymmetrized_spin_eri, get_uhf_spin_orbital_arrays_blockstacked,
    DFIntegrals)
from src.SingleReference.ADC.eeADC import (ee_u_dense_full, ee_u_sigma_full,
                                     ee_r_sigma, ee_r_sigma_df, ee_utils,
                                     ee_ri_density, ee_state_specific)
from src.SingleReference.ADC.eeADC import ee_relaxed_density as erd
from src.Base.constants import (ADC2_SPIN_SCALING, EE_ADC_CONV_TOL,
                                EE_FOLD_RESIDUAL_GAP, EE_RI_TILE_GB)
from src.Base.environment import attached_environment, resolve_environment
from src.Base.utils import memory
from src.Base.utils.linearAlgebra.diagonalization import eigh_symmetric
from src.Base.utils.mpi_grid import current_comm
from src.Solvers.davidson import diagonal_seeds, solve_symmetric

DENSE_LIMIT = 2000      # below this, Davidson's subspace goes linearly
                        # dependent long before it converges; a 59-dim
                        # open-shell sector hit exactly that. Every route
                        # honours it.

FOLD_DENSE_LIMIT = 64   # singles channel size up to which the fold builds
                        # A_eff densely: n products per Newton step against a
                        # Davidson's dozen, so only where Davidson would run
                        # out of space

_CHANNEL = {'singlet': +1.0, 'triplet': -1.0}


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
                 df=False, spin=None, matrix_free=True, conv_tol=EE_ADC_CONV_TOL,
                 en_dress=None, frozen=0, auxbasis=None,
                 ms_sector='auto', max_subspace=None, return_parity=False,
                 dense_limit=None, threshold=5000, fold=False,
                 environment=None, return_fold=False, tile_gb=EE_RI_TILE_GB,
                 device='cpu', state_specific='off', ss_density='unrelaxed',
                 spin_scaling=None, relaxed_density=False):
    """(e, Z): the lowest `nroots` excitation energies and their vectors.

    en_dress: Epstein-Nesbet channel dict (ee_en); True means the standard
    hole-hole + particle-particle dressing. It dresses the DOUBLES amplitude
    denominators only -- the singles amplitude and the supermatrix keep their
    MP zeroth order, matching the IP/EA side's en_dress convention.

    return_parity: spin-free route only; returns (e, Z, parity) with
    parity[k] = <Z_k | F Z_k> under the alpha<->beta flip F: +1 for a singlet
    (or the Ms = 0 part of a quintet, which only the doubles can form), -1 for
    an Ms = 0 triplet. It labels the roots of a spin=None solve.

    dense_limit: configurations below which the supermatrix is built and
    diagonalized instead of iterated on (see DENSE_LIMIT, the default; with
    fold=True the singles channel's size, default FOLD_DENSE_LIMIT, since a
    dense A_eff costs one product per configuration at every Newton step).
    Exposed so a test can force the iterative route onto a system small
    enough to also solve densely -- which is the only way to compare the two.

    threshold: configurations above which a DENSE solve is distributed over
    MPI ranks via ELPA, the same knob and the same backend the charged IP/EA
    route uses (ADCSolver.solve). Below it the solve is local. Distributed,
    the eigenvectors come back on rank 0 only and Z is None elsewhere.

    fold: True solves the folded problem on density-fitted tiles (module
    docstring): Z is then the singles part of each full eigenvector,
    sqrt(T1) y, on the flat spin-orbital singles layout (2 nocc nvir, frozen
    orbitals dropped); conv_tol is both its residual and its frequency
    tolerance (Hartree); tile_gb bounds a tile's working set; device 'gpu' runs
    the folded route's doubles kernel on each rank's GPU. return_fold
    adds the ee_fold.FoldResult (singles weights T1, steps, converged).

    environment: a continuum, or None for the one attached to mf (if any);
    see the module docstring for the convention. Restricted closed-shell
    spin-free routes only.

    state_specific: 'off' (default), 'vertical' or 'equilibrium', fold=True
    in a continuum only. e is then omega plus each root's state-specific
    shift (module docstring); the FoldResult (return_fold) keeps the
    eigenvalues in .omega, the shifts of both modes in .ss_shifts
    ({'vertical', 'equilibrium'}, Hartree; equilibrium needs eps_static),
    the one added in .ss_shift, and each root's active-space spatial
    difference density in .delta_density.

    ss_density: the difference density the state-specific term is built from.
    'unrelaxed' (default) is the ISR-ADC(2) expectation value; 'relaxed' the
    derivative of omega with the orbital response, the ground-state surface
    charges frozen (PTE; ee_relaxed_density); 'relaxed+pcm' the same with the
    continuum responding inside the orbital response, a bound on that choice.
    With a relaxed density the FoldResult also holds .delta_density_relaxed
    (full MO basis, frozen core included), .ss_shifts_unrelaxed, and
    .delta_dipole {'unrelaxed', 'relaxed'} (nroots, 3) in atomic units, the
    electronic dipole change -Tr(Delta rho r).

    spin_scaling: None (ADC(2)), 'scs', 'sos' (ADC2_SPIN_SCALING) or factors
    (c_os, c_ss): spin-component scaled ADC(2) (ee_ri_fold), fold=True at
    level 'adc2' without state_specific (its densities are ADC(2)'s).

    relaxed_density (fold=True, return_fold=True): each root's orbital-relaxed
    difference density d omega / d h in the FoldResult's
    .delta_density_relaxed (full MO basis, spin-summed), in the gas phase or
    in any environment, without the state-specific term; the orbitals respond
    with the mean field's own (gas-like) kernel. What a solvation loop
    (src.Base.solvation_loop) reads."""
    mol = mol if mol is not None else mf.mol
    scaling = _spin_scaling(spin_scaling)
    if scaling is not None and not (fold and level == 'adc2'):
        raise ValueError("spin_scaling is wired on fold=True at level 'adc2'")
    if scaling is not None and state_specific != 'off':
        raise NotImplementedError('spin_scaling with state_specific: the '
                                  'difference densities are unscaled ADC(2)')
    if spin is not None and spin not in _CHANNEL:
        raise ValueError(f"spin={spin!r}; expected 'singlet', 'triplet' or None")
    if return_fold and not fold:
        raise ValueError('return_fold needs fold=True')
    if dense_limit is None:
        dense_limit = FOLD_DENSE_LIMIT if fold else DENSE_LIMIT
    env = _reaction_field_environment(mf, mol, environment, route)
    if ss_density not in SS_DENSITIES:
        raise ValueError(f'ss_density={ss_density!r}; expected one of '
                         f'{SS_DENSITIES}')
    if ss_density != 'unrelaxed' and state_specific == 'off':
        raise ValueError(f'ss_density={ss_density!r} needs state_specific')
    if relaxed_density and not (fold and return_fold and level in ('adc1', 'adc2')
                                and scaling is None):
        raise ValueError('relaxed_density needs fold=True, return_fold=True, '
                         "level 'adc1' or 'adc2' and no spin_scaling")
    if ee_state_specific.check_mode(state_specific) != 'off':
        if not fold:
            raise NotImplementedError('state_specific is wired on fold=True')
        if env is None:
            raise ValueError(f'state_specific={state_specific!r} needs a '
                             'continuum (environment)')
        # 'equilibrium' without eps_static: refused before the solve
        ee_state_specific.shift_modes(env, state_specific)
    if fold:
        return _solve_folded_ri(mf, mol, level, nroots, spin, conv_tol,
                                en_dress, frozen, auxbasis, dense_limit,
                                return_fold, env, tile_gb, route, device,
                                state_specific, ss_density, scaling,
                                relaxed_density)
    if return_parity and route != 'spinfree':
        raise ValueError("return_parity is defined on route='spinfree' only, "
                         "where the alpha<->beta flip acts on the vector layout")
    from src.SingleReference.ADC.eeADC.ee_en import validate_dress
    en_dress = validate_dress(en_dress)
    if en_dress is not None and level in ('adc1', 'gf2'):
        raise ValueError(f'level={level!r} carries no amplitudes, so en_dress has '
                         'nothing to dress')
    if route == 'unrestricted':
        return _solve_unrestricted(mf, mol, level, nroots, matrix_free,
                                   conv_tol, en_dress, frozen, auxbasis, df,
                                   dense_limit, threshold)
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
                                   conv_tol, en_dress, ms_sector, dense_limit,
                                   threshold)
    if route != 'spinfree':
        raise ValueError(f"route={route!r}; expected 'spinfree', "
                         "'unrestricted' or 'spinorbital'")
    return _solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                            conv_tol, en_dress, frozen, auxbasis, max_subspace,
                            return_parity, dense_limit, threshold, env)


SS_DENSITIES = ('unrelaxed', 'relaxed', 'relaxed+pcm')


def _reaction_field_environment(mf, mol, environment, route):
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
    import contextlib
    if getattr(mf, 'with_screening', None) is None:
        return contextlib.nullcontext(mf)
    return attached_environment(mf, None)


def _spin_scaling(spin_scaling):
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


def _solve_folded_ri(mf, mol, level, nroots, spin, conv_tol, en_dress, frozen,
                     auxbasis, dense_limit, return_fold, env, tile_gb, route,
                     device='cpu', state_specific='off', ss_density='unrelaxed',
                     scaling=None, relaxed_density=False):
    """fold=True: the density-fitted tiled fold of ee_ri_fold."""
    from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
    if route != 'spinfree' or isinstance(mf, _scf.uhf.UHF):
        raise ValueError('fold=True is a closed-shell spin-free route')
    if level not in ee_ri_fold.LEVELS:
        raise ValueError(f'fold=True takes level in {ee_ri_fold.LEVELS}; '
                         f'level={level!r}')
    if spin not in _CHANNEL:
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
                               tile_gb=tile_gb, device=device, scaling=scaling)

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
    embed, _, _ = _channel_basis(2 * op.n, op.no, op.nv, 'adc1',
                                 _CHANNEL[spin])
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
    return (e, Z, res) if return_fold else (e, Z)


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
    from src.SingleReference.ADC.eeADC import ee_fold, ee_ri_fold
    embed, _, _ = _channel_basis(2 * op.n, op.no, op.nv, 'adc1',
                                 _CHANNEL[op.spin])
    mv = ee_ri_fold.FoldedMatvec(op, None)
    if op.n <= dense_limit:
        e, X = np.linalg.eigh(ee_fold.dense_effective(mv, op.n))
        e, X, conv = e[:nroots], X[:, :nroots], np.ones(min(nroots, op.n), bool)
    else:
        e, X, conv = solve_symmetric(mv, op.diag, nroots=nroots,
                                     x0=_open_seeds(op.diag, nroots, None),
                                     tol_residual=conv_tol,
                                     label='ee-ADC(1) (tiled)')
    k = len(e)
    return ee_fold.FoldResult(np.asarray(e), np.asarray(X), np.ones(k),
                              np.ones(k, int), ['index'] * k,
                              np.asarray(conv, bool), embed, np.arange(k))


def _solve_unrestricted(mf, mol, level, nroots, matrix_free, conv_tol,
                        en_dress=None, frozen=0, auxbasis=None, df=True,
                        dense_limit=DENSE_LIMIT, threshold=5000):
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
    if not matrix_free or n <= dense_limit:
        H = np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])
        e, Z, _ = eigh_symmetric(0.5 * (H + H.T), threshold=threshold)
        return e[:nroots], None if Z is None else Z[:, :nroots]
    e, Z, _ = solve_symmetric(aop, diag, nroots=nroots, tol_residual=conv_tol,
                              label='ee-ADC (unrestricted)',
                              max_memory=_subspace_memory(mf))
    return np.asarray(e), np.asarray(Z)


def _solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                     conv_tol, en_dress=None, frozen=0, auxbasis=None,
                     max_subspace=None, return_parity=False,
                     dense_limit=DENSE_LIMIT, threshold=5000, env=None):
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
    from src.SingleReference.ADC.eeADC.ee_ri_fold import TransitionReactionField
    rf = TransitionReactionField.from_environment(env, mol, mf, frozen)
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
            parity=_CHANNEL.get(spin))
    else:
        V = get_two_electron_integrals_chemist(
            mol, mf, representation='spatial')[act, act, act, act
                                               ].transpose(0, 2, 1, 3)
        aop, diag, dims = ee_r_sigma.build_operator(
            eps, V, no, level=level, en_dress=en_dress,
            parity=_CHANNEL.get(spin))
    return aop, diag, dims


def _spin_free_solve(aop, diag, dims, no, nv, level, nroots, spin,
                     matrix_free, conv_tol, max_subspace, return_parity,
                     dense_limit, threshold, mf):
    """Solve the spin-free operator: dense below dense_limit, else Davidson;
    one spin channel in its flip-pair basis when spin is given."""
    n = dims['nH']
    embed = None
    if spin is not None:
        # one channel: solve in its flip-pair basis, where it is a coordinate
        # subspace; shifting the other channel away instead stalls the
        # Davidson, as the Ms-sector note in _solve_spin_orbital records
        embed, restrict, reps = _channel_basis(n, no, nv, level, _CHANNEL[spin])
        raw = aop

        def aop(u):
            return restrict(np.asarray(raw(embed(np.asarray(u).ravel()))).ravel())

        diag, n = diag[reps], len(reps)
        if n < nroots:
            raise ValueError(f'the {spin} channel holds {n} states, fewer than '
                             f'nroots={nroots}: lower nroots')

    if not matrix_free or n <= dense_limit:
        H = np.column_stack([aop(np.eye(n)[:, k]) for k in range(n)])
        e, Z, _ = eigh_symmetric(0.5 * (H + H.T), threshold=threshold)
        e, Z = e[:nroots], None if Z is None else Z[:, :nroots]
    else:
        e, Z, _ = solve_symmetric(aop, diag, nroots=nroots,
                                  x0=_open_seeds(diag, nroots, max_subspace),
                                  tol_residual=conv_tol,
                                  max_subspace=max_subspace,
                                  label='ee-ADC (spin-free)',
                                  max_memory=_subspace_memory(mf))
        e, Z = np.asarray(e), np.asarray(Z)
    if Z is None:
        # a distributed dense solve: the eigenvectors are rank 0's alone
        return (e, None, None) if return_parity else (e, None)
    if embed is not None:
        Z = np.column_stack([embed(Z[:, k]) for k in range(Z.shape[1])])
    if not return_parity:
        return e, Z
    parity = np.array([Z[:, k] @ ee_r_sigma.spin_flip_vector(Z[:, k], no, nv, level)
                       for k in range(Z.shape[1])])
    return e, Z, parity


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


def _channel_basis(n, no, nv, level, sgn):
    """(embed, restrict, reps): an orthonormal basis of one spin channel.

    The alpha<->beta flip F permutes the vector's entries, so its sgn
    eigenspace is spanned by (e_k + sgn e_F(k)) / sqrt(2) over the pairs
    k < F(k), plus e_k at every fixed point k = F(k) when sgn = +1. The +1
    space holds the singlets and the Ms = 0 part of the quintets the doubles
    can form; the -1 space holds the Ms = 0 triplets. embed maps coefficients
    in that basis to the full vector, restrict(embed(u)) == u, and reps picks
    one full-vector entry per basis vector (the operator diagonal is the same
    on both entries of a pair)."""
    mate = ee_r_sigma.spin_flip_vector(np.arange(n), no, nv, level)
    k = np.arange(n)
    fixed = mate == k
    reps = k[(k < mate) | (fixed & (sgn > 0))]
    pair = ~fixed[reps]
    mate = mate[reps[pair]]
    w = np.where(pair, np.sqrt(0.5), 1.0)

    def embed(u):
        v = np.zeros(n)
        v[reps] = w * u
        v[mate] = sgn * w[pair] * u[pair]
        return v

    def restrict(v):
        u = w * v[reps]
        u[pair] += sgn * w[pair] * v[mate]
        return u

    return embed, restrict, reps


def _solve_spin_orbital(mf, mol, level, nroots, spin, matrix_free,
                        conv_tol, en_dress=None, ms_sector='auto',
                        dense_limit=DENSE_LIMIT, threshold=5000):
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
        e, Z, _ = eigh_symmetric(T.T @ (H @ T), threshold=threshold)
        return e[:nroots], None if Z is None else Z[:, :nroots]
    mask = None
    if ms_sector is not None:
        sz = ee_utils.spin_labels(mf, nocc, norb)
        mask = ee_utils.ms_sector_mask(sz, nocc, norb, ms_sector)
        if level == 'adc1':
            mask = mask[:ee_utils.dimensions(nocc, norb)['n_s']]

    dim = int(mask.sum()) if mask is not None else \
        ee_utils.dimensions(nocc, norb)['nH' if level != 'adc1' else 'n_s']
    if not matrix_free or dim <= dense_limit:
        H = ee_u_dense_full.build_supermatrix(eps, g, nocc, level=level,
                                              en_dress=en_dress)
        if mask is None:
            e, Z, _ = eigh_symmetric(H, threshold=threshold)
            return e[:nroots], None if Z is None else Z[:, :nroots]
        e, Z_sub, _ = eigh_symmetric(H[np.ix_(mask, mask)], threshold=threshold)
        if Z_sub is None:
            return e[:nroots], None     # not this rank's eigenvectors
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
        # converge. (The spin-free singlet/triplet channel is not a coordinate
        # subspace of the configurations either, but it is one of the flip-pair
        # basis, which is how _channel_basis restricts it.)
        raw, keep = aop, mask
        n_full = len(mask)

        def aop_sub(v_sub):
            v = np.zeros(n_full)
            v[keep] = np.asarray(v_sub).ravel()
            return raw(v)[keep]

        e, Z_sub, _ = solve_symmetric(aop_sub, diag[keep], nroots=nroots,
                                      tol_residual=conv_tol,
                                      label='ee-ADC (spin-orbital, Ms sector)',
                                      max_memory=_subspace_memory(mf))
        Z_sub = np.asarray(Z_sub)
        if Z_sub.ndim == 1:
            Z_sub = Z_sub[:, None]
        Z = np.zeros((n_full, Z_sub.shape[1]))
        Z[keep, :] = Z_sub          # back to the full configuration basis
        return np.asarray(e), Z
    e, Z, _ = solve_symmetric(aop, diag, nroots=nroots, tol_residual=conv_tol,
                              label='ee-ADC (spin-orbital)',
                              max_memory=_subspace_memory(mf))
    return np.asarray(e), np.asarray(Z)
