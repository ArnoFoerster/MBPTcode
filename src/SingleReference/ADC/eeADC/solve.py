"""The front end of the polarization-propagator (electronic-excitation) ADC:
solve_ee_adc checks its arguments and hands them to one route of ee_driver
(state and routing), whose physics is in the ee_* route modules.

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

environment: a polarizable continuum (src.environments.solvent_screening), given here
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
from src.Base.numerics.constants import EE_ADC_CONV_TOL, EE_RI_TILE_GB
from src.SingleReference.ADC.eeADC import ee_driver as _drv
from src.SingleReference.ADC.eeADC import ee_state_specific
from src.SingleReference.ADC.eeADC.ee_driver import EEResult  # noqa: F401  (exported)
from src.SingleReference.ADC.eeADC.ee_en import validate_dress


def solve_ee_adc(mf, mol=None, level='adc3', nroots=5, route='spinfree',
                 df=False, spin=None, matrix_free=True, conv_tol=EE_ADC_CONV_TOL,
                 en_dress=None, frozen=0, auxbasis=None,
                 ms_sector='auto', max_subspace=None, return_parity=False,
                 dense_limit=None, threshold=5000, fold=False,
                 environment=None, return_fold=False, tile_gb=EE_RI_TILE_GB,
                 device='cpu', state_specific='off', ss_density='unrelaxed',
                 spin_scaling=None, relaxed_density=False):
    """(e, Z): the lowest `nroots` excitation energies and their vectors, as
    an EEResult: the tuple, plus .converged, one flag per root (a Davidson or
    fold root that did not converge also warns).

    en_dress: Epstein-Nesbet channel dict (ee_en); True means the standard
    hole-hole + particle-particle dressing. It dresses the DOUBLES amplitude
    denominators only -- the singles amplitude and the supermatrix keep their
    MP zeroth order, matching the IP/EA side's en_dress convention.

    return_parity: spin-free route only; returns (e, Z, parity) with
    parity[k] = <Z_k | F Z_k> under the alpha<->beta flip F: +1 for a singlet
    (or the Ms = 0 part of a quintet, which only the doubles can form), -1 for
    an Ms = 0 triplet. It labels the roots of a spin=None solve.

    dense_limit: configurations below which the supermatrix is built and
    diagonalized instead of iterated on (see ee_driver.DENSE_LIMIT, the default; with
    fold=True the singles channel's size, default ee_driver.FOLD_DENSE_LIMIT, since a
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
    (src.environments.solvation_loop) reads."""
    mol = mol if mol is not None else mf.mol
    scaling = _drv.spin_scaling_factors(spin_scaling)
    if scaling is not None and not (fold and level == 'adc2'):
        raise ValueError("spin_scaling is wired on fold=True at level 'adc2'")
    if scaling is not None and state_specific != 'off':
        raise NotImplementedError('spin_scaling with state_specific: the '
                                  'difference densities are unscaled ADC(2)')
    if spin is not None and spin not in _drv.CHANNELS:
        raise ValueError(f"spin={spin!r}; expected 'singlet', 'triplet' or None")
    if return_fold and not fold:
        raise ValueError('return_fold needs fold=True')
    if dense_limit is None:
        dense_limit = _drv.FOLD_DENSE_LIMIT if fold else _drv.DENSE_LIMIT
    env = _drv.reaction_field_environment(mf, mol, environment, route)
    if ss_density not in _drv.SS_DENSITIES:
        raise ValueError(f'ss_density={ss_density!r}; expected one of '
                         f'{_drv.SS_DENSITIES}')
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
        return _drv.solve_folded_ri(mf, mol, level, nroots, spin, conv_tol,
                                    en_dress, frozen, auxbasis, dense_limit,
                                    return_fold, env, tile_gb, route, device,
                                    state_specific, ss_density, scaling,
                                    relaxed_density)
    if return_parity and route != 'spinfree':
        raise ValueError("return_parity is defined on route='spinfree' only, "
                         "where the alpha<->beta flip acts on the vector layout")
    en_dress = validate_dress(en_dress)
    if en_dress is not None and level in ('adc1', 'gf2'):
        raise ValueError(f'level={level!r} carries no amplitudes, so en_dress has '
                         'nothing to dress')
    if route == 'unrestricted':
        return _drv.solve_unrestricted(mf, mol, level, nroots, matrix_free,
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
        return _drv.solve_spin_orbital(mf, mol, level, nroots, spin, matrix_free,
                                       conv_tol, en_dress, ms_sector, dense_limit,
                                       threshold)
    if route != 'spinfree':
        raise ValueError(f"route={route!r}; expected 'spinfree', "
                         "'unrestricted' or 'spinorbital'")
    return _drv.solve_spin_free(mf, mol, level, nroots, df, spin, matrix_free,
                                conv_tol, en_dress, frozen, auxbasis, max_subspace,
                                return_parity, dense_limit, threshold, env)
