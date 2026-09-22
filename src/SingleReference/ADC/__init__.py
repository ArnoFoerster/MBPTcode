"""Dyson IP/EA-ADC, plus electronic-excitation ADC in eeADC/.

    eeADC/                  excitation-energy ADC (polarization propagator,
                            the ee_* modules) -- see its ee_driver.py for
                            solve_ee_adc, the front end
    base.py                ADCSolver dispatcher + the two branch classes
    adc_utils.py           shared DF-gather / chunked-einsum / solver helpers
    adc_r_driver.py        hand-written restricted orchestration
    adc_r_utils.py         restricted CSF helpers + unstreamed U fallback
    adc_r_dense_full.py    restricted dense supermatrix, dense integrals
    adc_r_dense_df.py      restricted dense supermatrix, DF integrals
    adc_r_sigma_full.py    restricted matrix-free operator, dense integrals
    adc_r_sigma_df.py      restricted matrix-free operator, DF (production)
    adc_u_driver.py        hand-written spin-orbital orchestration
    adc_u_utils.py         spin-orbital helpers (EN-dressed t2, ingredients)
    adc_u_dense_full.py    spin-orbital dense supermatrix, dense g
    adc_u_dense_df.py      spin-orbital dense supermatrix, B_spin
    adc_u_sigma_full.py    spin-orbital matrix-free operator, dense g
    adc_u_sigma_df.py      spin-orbital matrix-free operator, g-free B_spin
    spin_adapt.py          CSF isometry and adapter
    static_correction.py   Sigma(infinity) builders + build_static_correction
    solve.py               Davidson/dense solvers, seeds, Lanczos spectral

Screening: build W with static_screened_coulomb_chemist/_aux (in
LinearResponse) and pass it as screening={'W_chemist': W} (dense) or
{'W_aux': W} (DF, matrix-free, adc2x); the C^(1) substitution happens
inside the route modules.

Usage:  solver = ADCSolver(mf, level=..., df=..., ...)
        sc = build_static_correction(mf, mol, kind=..., en_dress=...)
        e, Z = solver.solve(static_correction=sc)

Levels: ADC(2)-X and ADC(3) on the restricted branch, ADC(3) on the
spin-orbital branch.
This file is re-exports only -- no definitions live here.
"""
from src.SingleReference.ADC.base import (ADCSolver, ADCSolverRestricted,
                                          ADCSolverUnrestricted)
from src.SingleReference.ADC.static_correction import (
    diagonalize_static_only, build_static_correction,
    build_mp2_static_correction, build_mp2_static_correction_restricted,
    build_mp2_static_correction_uhf_df,
    build_mp3_static_correction, build_mp3_static_correction_restricted,
    build_ccsd_static_correction, build_ccsd_static_correction_restricted,
    build_ccsdt_static_correction, build_ccsdt_static_correction_restricted,
    build_ks_static_correction, build_ks_static_correction_restricted,
    build_solvent_static_correction, build_solvent_static_correction_restricted,
)
