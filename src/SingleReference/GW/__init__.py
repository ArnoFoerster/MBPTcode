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

Davidson note: the three iterative eigensolvers in this tree --
LinearResponse/davidson.py (symplectic Casida via pyscf real_eig), ADC
(root-following pyscf davidson1), CC/eom.py (non-Hermitian pyscf
davidson_nosym1 + biorthogonalization) -- all delegate the iterative core to
pyscf and differ only in problem-specific setup, so there is no shared
Davidson core worth extracting.
"""
