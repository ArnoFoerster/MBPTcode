"""GW and vertex-corrected (PSD) self-energies and quasiparticle energies.

    qp_energy.py              calc_qp_energy: the front door
    self_energy.py            Lehmann self-energy from a Casida solution
    transition_amplitudes.py  chi / vertex transition amplitudes (named apart
                              from CC/amplitudes.py, the generated CCSDT
                              equations)
    qp_solve.py               shared QP machinery for the imaginary-axis
                              routes -- Pade continuation, Sigma_x - v_xc
    imaginary_axis.py         Sigma by quadrature on an imaginary frequency
                              grid, O(N^4)
    imaginary_time.py         Sigma = -G W as a pointwise product in
                              imaginary time, and W(i tau) itself
    space_time.py             solve_qp_energy_space_time: the O(N^3) route,
                              imaginary time on separable (ISDF) factors
    contour_deformation.py    quasiparticle energies by contour deformation,
                              no analytic continuation
    sum_over_poles.py         the self-energy as a sum over auxiliary poles
    real_screening.py         the screened interaction at a real frequency
    quasi_boson.py            the quasi-boson dRPA ground state and its EOM
                              G0W0 quasiparticle energies
    auxiliary_bosons.py       AB-G0W0: the dRPA bosons compressed into an RI
                              basis
    cc_polarizability.py      G0W@CC -- the RPA polarizability replaced by an
                              EOM-CC one, through CC/eom.py
    reaction_field.py         Duchemin et al. Eq. (18): the continuum's shift
                              of every quasiparticle energy, and the gauge
                              transform that reaches the bare screening
    evGW.py                   the eigenvalue-self-consistent loop over any of
                              the routes above
    qsGW.py                   quasiparticle self-consistent GW
    qp_states.py              which orbitals carry an explicitly solved
                              quasiparticle energy
    qp_selection.py           which orbitals of the admitted window a surface
                              solves explicitly

Every quasiparticle equation is solved by the one root finder,
`Solvers.qp_equation.find_qp_root`. Every symmetric eigenproblem goes through
the one symmetric Davidson, `Solvers.davidson.solve_symmetric` (ADC, the
Tamm-Dancoff BSE, the (A - B) probe). The exceptions, each because its problem
is not one a symmetric Davidson can take:
  LinearResponse/davidson.py  the full Casida/BSE problem (B != 0): paired, and
                              its symmetric forms need (A - B)^1/2, which a
                              matrix-free action does not give; its trial
                              space is split by pair rows over the ranks
  BSE/bse_upfolded.py         the upfolded BSE: non-symmetric, its roots
                              possibly complex
  CC/eom.py                   EOM-CC: non-Hermitian biorthogonal (pyscf
                              davidson_nosym1)
"""
