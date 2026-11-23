"""Static effective Hamiltonians of an active space, downfolded from a
low-level method (HF or RPA).

Usage: one low-level run, then one Hamiltonian per active space.

    low = run_low_level(mf, method='rpa', mode='casida')
    h = build_effective_hamiltonian(low, active_space, screening='crpa',
                                    sigma='srg')
    h.solve(solver)

The options (default first). Details: rpa.run and rpa.downfold in
methods/rpa.py, whose tuples MODES, SCREENINGS, SIGMAS and DIAGNOSTICS are the
valid values; every value is checked by `require_choice`
(src/Base/utils/choices.py).

run_low_level(mf, method, **method_opts)
    method      'rpa' | 'hf' ('hf' takes no further options)
    mode        'casida' | 'imagfrequency' | 'space-time': every pole
                (dense) | density fitted, imaginary frequency, N^4 |
                separable ISDF factors, imaginary time, N^3
    df          True | False (the exact 4-centre integrals, casida only)
    numerics    None (the defaults) | a Numerics (space-time needs one)

build_effective_hamiltonian(low, active_space, spin_free, **method_opts)
    spin_free   True | False: one spatial Hamiltonian, or the spin blocks
    screening   'crpa' | 'bare' | 'crpa_ph' | 'mrpa': cRPA W^r | the bare v
                | W^r on the particle-hole elements only | Scott and Booth's
                moment-conserving interaction (casida only)
    sigma       None | 'srg' | 'averaged': no Sigma_bar, full double
                counting | the static environment self-energy in the SRG
                form | in the averaged form; both with the Hartree-only
                double counting
    diagnostics all of rpa.DIAGNOSTICS | a subset | None
    numerics    None (the run's) | a Numerics whose downfolding section
                (srg_s, pole_margin, continuation_margin) replaces the run's

The general layer (low_level.py, spin.py, core.py, double_counting.py,
zero_body.py, one_body.py, two_body.py and the driver
effective_hamiltonian.py) knows no method; each low-level method is one file
under methods/ and hands the general layer a `DownfoldingTerms`
(low_level.py), the only seam between the two. The driver reaches the
methods through the `LOW_LEVEL` registry of methods/__init__.py. The
particle-hole masks the methods read are functions of the active space
(src/EffectiveHamiltonian/active_space.py).
"""
from .effective_hamiltonian import build_effective_hamiltonian, run_low_level
