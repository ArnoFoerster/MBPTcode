"""The ISDF-vs-DF acceptance check of one stage of a run.

`Run.accept(stage)` calls `check_stage(run, stage)` after the vertical stage
and each relaxed one when the declaration asks for the check
(`Policies.df_check`). At the stage's geometry the check builds ONE mean field
-- the declared reference functional with J and K fitted on <basis>-jkfit,
relaxed in the run's continuum at eps_static, the continuum attached for the
response, in the span the stage's record names -- and ONE BSE diagonal from the
record's `qp_bookkeeping`: the chain's quasiparticle energies on its explicit
set, the DF orbital energies plus the chain's per-orbital scissor outside it.
Both kernels then solve on it:

    ISDF   `solve_bse_isdf` on factors at the declared level (auxiliary set
           <basis>-ri), the static W at the mean field
    DF     `solve_bse_df`, the static W at the mean field

and the followed root of the record (`props.root`) is compared. ISDF - DF must
lie within ISDF_GRID_ACCURACY_TARGET_EV of the level. Mean field and diagonal
are the same on both sides, so the difference is the factorization's alone; the
off-diagonal quasiparticle Hamiltonian of a run that declares one is left out
of both. The relaxed stages are checked in the continuum at eps_static whatever
their solvation, as the vertical one is: the check measures the factorization,
and a relaxed state's reaction field is the same on both sides.

THE STAGES. 'vert' compares the followed singlet and the followed triplet at R0,
's1' the singlet and 't1' the triplet at their minima.

THE (A - B) PROBE. With `numerics.bse.probe` on and the full kernel the ISDF
solve also converges min eig(A - B) and a non-positive one fails the stage; the
Tamm-Dancoff kernel has none.

REFUSALS. Factors that miss an orbital density of the DF mean field
(`UnrepresentedOrbitals`) fail the stage with the refusal as the reason, as does
a DF mean field whose orbital energies on the explicit set differ from the
chain's by more than DF_CHECK_ORBITAL_MATCH_TOL_EV (the two mean fields would
not order their orbitals alike, and the diagonal would be put on the wrong
orbitals).

The Davidson of this check stops at DF_CHECK_BSE_CONV_TOL, and its mean field
is converged for an energy (`Numerics.derive` of the run's numerics).
"""
import time

import numpy as np

from src.Base.numerics.constants import (DF_CHECK_BSE_CONV_TOL,
                                         DF_CHECK_ORBITAL_MATCH_TOL_EV,
                                         DF_CHECK_ROOTS_ABOVE_FOLLOWED, EV_TO_MEV,
                                         HARTREE_TO_EV,
                                         ISDF_GRID_ACCURACY_TARGET_EV,
                                         SCF_ENERGY_CONV_TOL, SCF_ENERGY_GRAD_TOL)
from src.Base.environment_seam import attach_environment
from src.Base.geometry.geometry import molecule_at
from src.Base.pyscf.pyscf_interface import occupied_count
from src.Base.factorization.separable_ri import UnrepresentedOrbitals
from src.SingleReference.GW.space_time import separable_factors
from src.SingleReference.BSE.solve import solve_bse_df, solve_bse_isdf
from src.photophysics import chain
from src.photophysics.records import label, log
from src.gradients.gwbse.surface_factory import reference_mean_field

#: The spins whose followed root each checked stage compares: the stage's
#: followed state(s) (the vertical stage holds both).
STAGE_SPINS = {'vert': ('singlet', 'triplet'), 's1': ('singlet',),
               't1': ('triplet',)}


def check_numerics(numerics):
    """`numerics` derived for this check: the mean field density-fitted
    (J and K on <basis>-jkfit) and converged for an energy, the BSE Davidson
    stopped at DF_CHECK_BSE_CONV_TOL."""
    return numerics.derive(
        'ISDF-vs-DF acceptance check: one density-fitted mean field converged '
        'for an energy, and a Davidson that settles a pass or fail at the meV '
        'scale',
        scf={'route': 'df', 'grid_level': None,
             'conv_tol': SCF_ENERGY_CONV_TOL,
             'conv_tol_grad': SCF_ENERGY_GRAD_TOL},
        bse={'conv_tol': DF_CHECK_BSE_CONV_TOL})


def with_roots(numerics, root):
    """`numerics` solving up to DF_CHECK_ROOTS_ABOVE_FOLLOWED roots above
    `root`."""
    want = int(root) + DF_CHECK_ROOTS_ABOVE_FOLLOWED
    if numerics.bse.nroots == want:
        return numerics
    return numerics.derive(
        f'the check solves the followed root {root} and '
        f'{DF_CHECK_ROOTS_ABOVE_FOLLOWED} above it', bse={'nroots': want})


def df_mean_field(run, mol, numerics, span):
    """The density-fitted mean field at `mol` for the check: the run's
    reference functional on `numerics` (route 'df'), relaxed in the run's
    continuum at eps_static with the continuum attached for the response, in
    `span` (n_kept, cut; n_kept None: the full basis)."""
    d = run.declaration
    env = chain.continuum(d, mol)
    factory = chain.mean_field_factory(d.reference, numerics, run.max_memory,
                                       [], finish='diis', warm_start=None,
                                       span=span, surface_route='df')
    mf = reference_mean_field(mol, factory, env)
    if env is not None:
        attach_environment(mf, env)
    return mf


def chain_diagonal(mf, bookkeeping):
    """(the chain's quasiparticle energies on the BSE diagonal, Ha; the largest
    |eps_DF - eps_chain| on the explicit set, eV).

    The explicit set carries the chain's roots, the orbitals outside it the DF
    orbital energy plus the chain's per-orbital scissor. The deviation only
    proves that the two mean fields order their orbitals alike."""
    eps = np.asarray(mf.mo_energy, float)
    if int(bookkeeping['n_orbitals']) != len(eps):
        raise RuntimeError(f"the chain has {bookkeeping['n_orbitals']} "
                           f'orbitals, the DF mean field {len(eps)}')
    diag = eps.copy()
    for shifts in (bookkeeping.get('scissor_eV') or {},
                   bookkeeping.get('inside_scissor_eV') or {}):
        for p, s in shifts.items():
            diag[int(p)] = eps[int(p)] + s / HARTREE_TO_EV
    dev = 0.0
    for p, root in bookkeeping['qp_root_eV'].items():
        diag[int(p)] = root / HARTREE_TO_EV
        dev = max(dev, abs(eps[int(p)] * HARTREE_TO_EV
                           - bookkeeping['eps_mean_field_eV'][p]))
    return diag, dev


def check_stage(run, stage):
    """The acceptance check of `stage` ('vert', 's1' or 't1') of `run`.

    Returns {'verdict': 'pass' | 'fail', 'reason', 'target_eV', 'stage',
    'level', 'e_df_scf', 'orbital_match_eV', 'orbital_match_tol_eV',
    'bse_conv_tol', 'spins': {spin: {root, isdf_eV, df_eV,
    isdf_minus_df_meV, chain_omega_eV, min_eig_amb_isdf_Ha, pass,
    davidson_flags}}}; `reason` is None for a pass. A Davidson that left a
    root unconverged is kept as solved and named in the spin's
    `davidson_flags` ({route: `davidson.davidson_flags`}) and, by spin, in
    the result's `davidson_flags`, absent when clean."""
    if stage not in STAGE_SPINS:
        raise ValueError(f'{stage!r} is not a checked stage: '
                         f'{sorted(STAGE_SPINS)}')
    d = run.declaration
    level = d.numerics.isdf.level
    target = ISDF_GRID_ACCURACY_TARGET_EV.get(level)
    out = {'stage': stage, 'level': level, 'target_eV': target,
           'orbital_match_tol_eV': DF_CHECK_ORBITAL_MATCH_TOL_EV,
           'bse_conv_tol': DF_CHECK_BSE_CONV_TOL, 'reason': None, 'spins': {}}
    if target is None:
        out.update(verdict='fail',
                   reason=f'grid level {level!r} has no accuracy target')
        return out
    record = run.read(stage)
    props = record['props']
    mol = molecule_at(run.mol_in, props['coords_bohr'])
    span = record.get('span') or {}
    numerics = check_numerics(d.numerics)
    t0 = time.time()
    mf = df_mean_field(run, mol, numerics,
                       (span.get('n_kept'), span.get('cut'))
                       if span.get('cut') else (None, None))
    nocc = occupied_count(mf)
    diag, dev = chain_diagonal(mf, record['qp_bookkeeping'])
    out.update(e_df_scf=float(mf.e_tot), orbital_match_eV=float(dev))
    log(f'{stage} check: DF SCF {mf.e_tot:.8f} Ha, explicit-set match '
        f'{dev * EV_TO_MEV:.2f} meV', t0)
    if dev > DF_CHECK_ORBITAL_MATCH_TOL_EV:
        out.update(verdict='fail',
                   reason=f'DF orbital energies differ from the chain by '
                          f'{dev * EV_TO_MEV:.1f} meV on the explicit set')
        return out
    tda = d.kernel == 'bse-tda'
    try:
        factors = separable_factors(mf, mol, auxbasis=chain.factor_auxbasis(mol),
                                    grid_accuracy=level, numerics=numerics)
    except UnrepresentedOrbitals as exc:
        out.update(verdict='fail', reason=str(exc))
        log(f'{stage} check: ISDF factors refused: {exc}')
        return out
    followed = {'singlet': ('singlet', d.singlet - 1),
                'triplet': ('triplet', d.triplet - 1)}
    failed = []
    for spin in STAGE_SPINS[stage]:
        name = label(followed[spin])
        root = int(props['root'][name])
        solving = with_roots(numerics, root)
        try:
            w_is, _, _, info = solve_bse_isdf(
                mf, mol, nocc, numerics=solving, qp=diag, factors=factors,
                spin=spin, tda=tda, probe=bool(solving.bse.probe) and not tda)
        except RuntimeError as exc:   # an unstable ISDF kernel fails the stage
            out['spins'][spin] = {'root': root, 'error': str(exc),
                                  'pass': False}
            failed.append(f'{spin}: ISDF refused ({exc})')
            log(f'{stage} {spin}: ISDF refused: {exc}')
            continue
        w_df, _, _, info_df = solve_bse_df(mf, mol, nocc, numerics=solving,
                                           qp=diag, spin=spin, tda=tda)
        # a Davidson root left unconverged is kept as solved and named here
        # (refuse inputs, warn numerics)
        unconverged = {route: flags for route, flags in (
            ('isdf', info.get('davidson_flags')),
            ('df', info_df.get('davidson_flags'))) if flags}
        e_is = float(np.sort(w_is)[root] * HARTREE_TO_EV)
        e_df = float(np.sort(w_df)[root] * HARTREE_TO_EV)
        ok = abs(e_is - e_df) <= target
        if not ok:
            failed.append(f'{spin}: ISDF - DF '
                          f'{EV_TO_MEV * (e_is - e_df):+.2f} meV '
                          f'above {EV_TO_MEV * target:.1f} meV')
        out['spins'][spin] = {
            'state': name, 'root': root, 'isdf_eV': e_is, 'df_eV': e_df,
            'isdf_minus_df_meV': EV_TO_MEV * (e_is - e_df),
            'chain_omega_eV': float(props['omega'][name] * HARTREE_TO_EV),
            'min_eig_amb_isdf_Ha': info.get('min_eig_amb'), 'pass': bool(ok),
            'davidson_flags': unconverged}
        if unconverged:
            out.setdefault('davidson_flags', {})[spin] = unconverged
        log(f'{stage} {spin} root {root}: ISDF {e_is:.4f} DF {e_df:.4f} eV, '
            f'{EV_TO_MEV * (e_is - e_df):+.2f} meV -> '
            f'{"pass" if ok else "FAIL"}')
    out['verdict'] = 'fail' if failed else 'pass'
    out['reason'] = '; '.join(failed) or None
    return out
