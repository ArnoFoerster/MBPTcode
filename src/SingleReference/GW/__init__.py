"""GW and vertex-corrected (PSD) self-energies and quasiparticle energies.

    transition_amplitudes.py  chi / vertex transition amplitudes (named apart
                              from CC/amplitudes.py, the generated CCSDT
                              equations)
    self_energy.py            Lehmann self-energy from a Casida solution
    qp_energy.py              calc_qp_energy: the front door, real axis
    qp_solve.py               shared QP machinery for the imaginary-axis
                              routes -- Pade continuation, Sigma_x - v_xc
    imaginary_axis.py         Sigma by quadrature on an imaginary frequency
                              grid, O(N^4)
    imaginary_time.py         Sigma = -G W as a pointwise product in
                              imaginary time, and W(i tau) itself
    space_time.py             solve_qp_energy_space_time: the O(N^3) route,
                              imaginary time on separable (ISDF) factors
    cc_polarizability.py      G0W@CC -- the RPA polarizability replaced by an
                              EOM-CC one, through CC/eom.py
    reaction_field.py         Duchemin et al. Eq. (18): the continuum's shift
                              of every quasiparticle energy, and the gauge
                              transform that reaches the bare screening
    evGW.py                   the eigenvalue-self-consistent loop over any of
                              the routes above

Davidson note, and it is a RULE: every iterative eigensolver in this tree
delegates its core to pyscf, and there are three of them because there are
three eigenproblems, not three tastes --
  Solvers/davidson.py         real symmetric (pyscf davidson1). BOTH ADC
                              routes: charged IP/EA root-following and
                              neutral ee lowest-k go through the one
                              solve_symmetric call.
  LinearResponse/davidson.py  non-Hermitian paired Casida (pyscf real_eig)
  CC/eom.py                   non-Hermitian biorthogonal (davidson_nosym1)
Neither of the last two reduces to the first: the Casida form needs the
(A-B) Cholesky reduction and X^2-Y^2 normalization, EOM-CC needs left/right
biorthogonalization. A fourth symmetric solver would duplicate the first:
both ADC routes solve the same problem, and one call keeps one meaning of
conv_tol and one answer to whether an unconverged root is reported.
"""
