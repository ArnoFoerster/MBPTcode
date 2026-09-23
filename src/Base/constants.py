"""Numeric defaults and self-energy method registry shared across src/SingleReference/."""

# Physical constants, CODATA 2018. Defined here and nowhere else: import them,
# never re-spell the digits, so every route reports the same number.
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.52917721092

# Lorentzian broadening for self-energy denominators / spectral functions.
DEFAULT_BROADENING_ETA = 1e-3

# Relative eigenvalue floor for inverting an auxiliary-basis metric. The
# LONG-RANGE metric of a range-separated hybrid is numerically singular --
# erf(omega r)/r is smooth, so tight auxiliary functions go linearly dependent
# under it -- so the fit is inverted on its numerical range rather than solved
# through.
AUX_METRIC_LINDEP = 1e-10

# Validated ISDF interpolation grids: {basis: {level: (A1, A2, A3, B1)}}, the
# Lebedev sub-shell replica counts measured to reach an accuracy. THE ONLY
# PLACE A VALIDATED COUNT IS WRITTEN DOWN; a re-measured grid is corrected here
# and nowhere else.
#
# A LEVEL IS AN ACCURACY TARGET, NOT A SIZE: the largest deviation of the three
# lowest BSE roots from `solve_bse_df` at the same mean field, over ten
# molecules spanning H C N O P S Si B with the roots matched between the two
# routes. G1 < 8 meV, G2 < 4 meV, G3 < 2 meV, at 6*A1 + 8*A2 + 12*A3 + 24*B1
# points per atom.
#
# THE LADDER IS NOT MONOTONE, so an entry licenses ITS OWN count and no other.
#
# EVERY GAP IS A REFUSAL, NOT A FALLBACK: a missing level means nobody measured
# it, so a neighbouring level, a larger count and another basis are all
# equally unlicensed.
ISDF_GRID_ACCURACY = {
    'cc-pvdz':     {'G1': (16, 10, 6, 2), 'G2': (16, 10, 6, 2),
                    'G3': (32, 20, 12, 4)},
    'cc-pvtz':     {'G1': (24, 15, 9, 3), 'G2': (24, 15, 9, 3),
                    'G3': (40, 25, 15, 5)},
    'aug-cc-pvdz': {'G1': (24, 15, 9, 3), 'G2': (24, 15, 9, 3)},
    'aug-cc-pvtz': {'G1': (24, 15, 9, 3), 'G2': (40, 25, 15, 5),
                    'G3': (40, 25, 15, 5)},
}

# Multi-start descents behind every grid tabulated above. A shipped radii row is
# keyed on its recipe as well as its counts, so the same counts found from one
# start are a DIFFERENT grid and not the one that was scored.
ISDF_GRID_N_START = 8

# CasidaSolver-only: TDA-shortcut threshold and omega^2 clipping before sqrt().
CASIDA_NUMERICAL_EPS = 1e-6

# Chunk size for blocked exciton contractions (memory/speed tradeoff only).
DEFAULT_BLOCK_SIZE = 512

# Eigenvalue-self-consistent GW (evGW): the quasiparticle energies are
# reinjected into G and P0 and the cycle repeated until the set stops moving.
# The tolerance is on max |delta eps| in Hartree; 1e-5 is 0.27 meV, below the
# basis and grid errors of any quantity built on top.
EVGW_MAX_CYCLE = 30
EVGW_TOL = 1e-5
# Linear mixing eps <- (1 - d) eps_new + d eps_old. Zero is the plain fixed
# point; raise it only for a spectrum that oscillates, which happens when a
# level crosses another between cycles.
EVGW_DAMPING = 0.0
# DIIS subspace for the evGW fixed point, and the cycle it starts on. The first
# cycle is the whole mean-field-to-G0W0 jump, several eV and nothing like the
# later steps, so extrapolating through it hurts; DIIS takes over once the
# iteration is in the linear regime.
EVGW_DIIS_SIZE = 8
EVGW_DIIS_START = 2

# Quasiparticle-self-consistent GW (qsGW). The loop stops when HOMO and LUMO
# move by less than EVGW_TOL and the density by less than QSGW_DM_TOL,
# ||D' - D||_F / nmo, PySCF's criterion. The DIIS space is PySCF's qsGW one;
# QSGW_MIXING_LAMBDA is the new Hamiltonian's weight lambda in Kaplan's linear
# mixing (J. Chem. Theory Comput. 12, 2528 (2016), eq. 21, their 0.3), used
# only when mixing='linear'. QSGW_BLOCK_ELEMS bounds the (chunk, norb, nmo)
# buffers of the static self-energy builder: 2**24 doubles is 128 MB each.
QSGW_DM_TOL = 1e-6
QSGW_DIIS_SIZE = 10
QSGW_MIXING_LAMBDA = 0.3
QSGW_BLOCK_ELEMS = 2**24
# SRG flow parameter s of the qsGW static self-energy, in Hartree^-2 (Marie and
# Loos, JCTC 2023, doi 10.1021/acs.jctc.3c00281, eq. 44). A diagonal term with
# energy denominator a enters with weight 1 - exp(-2 a^2 s), above 0.99 for
# |a| > 4.1 eV at s = 100, so the near-pole terms that stall mode A are damped
# and the rest kept. Marie and Loos find their accuracy plateau from s = 50 and
# recommend 500 or 1000, judged on convergence from a HF start. On water
# cc-pVDZ from s = 200 up the high virtuals carry two self-consistent branches
# and PBE and PBE0 starts end 0.9 to 1.6 meV apart at the frontier; at s = 100
# they agree to 1e-4 meV, with HOMO and LUMO about 1 meV from s = 1000.
QSGW_SRG_FLOW = 100.0
# Relative error bound of the quadrature behind the SRG kernel,
# (1 - exp(-s lam)) / lam = int_0^s exp(-t lam) dt ~ sum_n w_n exp(-t_n lam):
# each term of Sigma~ is off by at most this fraction of itself.
QSGW_SRG_QUAD_TOL = 1e-7

# Energy convergence of the active-space exact diagonalization (pyscf FCI);
# tight because its densities land in a gradient, not only its energy.
FCI_CONV_TOL = 1e-13

# CPHF/CPKS Z-vector solve (GWDensityMatrixSolver.solve_relaxation).
CPHF_MAX_CYCLE = 100
CPHF_TOL = 1e-9

# QP root-finding (src/Solvers/qp_equation.py).
QP_NEWTON_TOL = 1e-6
QP_NEWTON_MAX_ITER = 50
QP_BISECTION_TOL = 1e-6
QP_BISECTION_MAX_ITER = 100
QP_GRAPHICAL_TOL = 1e-8
QP_GRAPHICAL_N_OMEGA = 150
QP_GRAPHICAL_MAX_BISECTION = 100
# Smallest pole strength Z = 1/f'(w) that the 'pole_strength' root selector will
# accept as a quasiparticle; roots below it are satellites. Deep valence/semicore
# states put low-Z satellites nearer to eps_HF than the true QP root, so the
# closest-root rule picks the satellite (Ne 2s: Z=0.03 at -52.5 eV vs Z=0.89 at
# -48.1 eV). Only matters where several roots exist.
QP_Z_MIN = 0.05
# Central-difference step (Hartree) for dSigma/dw when evaluating Z.
QP_Z_DERIV_STEP = 1e-3

# spin factor
GW_DENSITY_SPIN_SUM = 4.0

# Singlet/triplet factor on the bare exchange kernel of a Casida/BSE problem:
# kappa (ia|jb) with kappa = 2 for a singlet and 0 for a triplet.
KAPPA = {'singlet': 2.0, 'triplet': 0.0}

# Working-set budget in GB for the tiled (M, M) intermediates of the ISDF
# routes: the polarizability sweep and the BSE block action. Memory only; the
# flop count is unchanged.
ISDF_TILE_GB = 4.0

# How far a Casida vector may sit from <X|X> - <Y|Y> = 1 before a consumer
# refuses it. Loose enough for a Davidson root at conv_tol 1e-5, tight enough
# that pySCF's 1/2 can never pass: that factor of two is invisible in every
# excitation energy and squared in every oscillator strength.
CASIDA_NORM_TOL = 1e-4

# Closest a classical polarizable site may sit to a QM nucleus, in Bohr. The
# exact folding of W onto the QM region holds only where the two subsystems'
# orbitals do not overlap (Li, D'Avino, Duchemin, Beljonne and Blase, Phys.
# Rev. B 97, 035108 (2018), Sec. II C), and nothing damps the field integral
# between a site and the QM charge: a site the density reaches answers a field
# the induced-dipole model has no physics for and drives v + vtilde indefinite.
# Two heavy atoms in van der Waals contact are about 3.4 Angstrom apart, which
# is the closest an MM centre comes to a QM one in any site list the model
# describes.
MIN_SITE_TO_QM_DISTANCE = 3.4 / BOHR_TO_ANGSTROM

# Uniform field strength in a.u. for the finite-field dipole derivative that
# gives a molecule's polarizability. Central in the field, so the leading error
# is the cubic hyperpolarizability term; 1e-3 keeps that far below the 10 %
# spread between classical site models calibrated against it (Li et al., J.
# Phys. Chem. Lett. 7, 2814 (2016)) and far above the dipole's SCF noise.
POLARIZABILITY_FIELD = 1e-3

# SCF convergence for a calibration polarizability. The finite-field dipole
# derivative divides by 2e-3 a.u., so a dipole converged to 1e-9 already puts
# 1e-6 Bohr^3 of noise on alpha; the same tolerance is held for the analytic
# response so that the two levels of theory differ by their kernel and by
# nothing else.
POLARIZABILITY_SCF_TOL = 1e-12

# Density-direction step for the reaction field's cross term in a correlated
# gradient (src/Base/pcm_derivatives.py). The solvation energy is EXACTLY
# quadratic in the density it is built from, so the central difference this
# scales carries no truncation error and the value is a conditioning choice
# only: measured step-independent to 3e-15 from 1e-1 down to 1e-3.
PCM_CROSS_TERM_STEP = 1e-2

# Contour deformation of the GW self-energy
# (src/SingleReference/GW/contour_deformation.py). A pole of G at
# |omega - eps_q| below RESIDUE_ON_CONTOUR_TOL counts as ON the contour.
# QP_POLE_OFFSET is how far off an orbital energy the quasiparticle iteration
# is kept: at omega = eps_q the imaginary-axis integrand collapses onto nu = 0
# and no quadrature resolves it (water/cc-pVDZ: exact to 3e-14 at 1e-3, only
# 5e-6 at 1e-4 -- a floor, not a tuning knob).
RESIDUE_ON_CONTOUR_TOL = 1e-10
QP_POLE_OFFSET = 1e-3
# The smallest offset the Newton iteration may fall back to when a
# quasiparticle root lies INSIDE the guard band -- the guard then undoes every
# step and the iteration deadlocks at a fixed point that is not the root.
QP_POLE_OFFSET_MIN = 1e-6
# Below this pole strength a converged root is a SATELLITE, not the
# quasiparticle. f(w) = w - eps_p - Sigma(w) diverges at every eps_q, so it has
# a genuine zero just to either side of each one, with Z = 1/(1 - dSigma/dw)
# going to zero there because the slope diverges.
QP_POLE_STRENGTH_MIN = 0.1
# Margin (Hartree) past the residue frequencies of the Newton START that the
# tau grid must carry for the Laplace residue backend to be chosen: the root
# moves by the quasiparticle correction, a few tenths of an eV to 2 eV.
RESIDUE_FREQ_MARGIN = 0.1
# Gauss-Legendre points on the imaginary-frequency integral of the dRPA
# correlation energy in the space-time route; converged to 1e-8 Ha at 40.
RPA_ENERGY_NFREQ = 40
# Gauss-Legendre points on the imaginary-frequency half of a contour
# deformation.
CD_NFREQ = 64
# How far inside the root-to-pole distance the FIRST contour-deformation
# frequency has to sit. A grid whose smallest node is not well inside the
# root-to-pole gap loses the Lorentzian spike the pole of G puts on the
# imaginary-frequency integrand, and the Newton is left on whatever the
# truncated self-energy has a zero at.
CD_POLE_RESOLUTION = 40.0
# Where doubling the contour-deformation grid gives up. A root sitting ON a
# pole of G is resolved by no quadrature, so the growth must stop somewhere
# and say so rather than run the cost up.
CD_NFREQ_MAX = 512
# Imaginary-time points behind the contour-deformation grid. The cosine
# transform onto the imaginary-frequency quadrature would be converged at 18;
# a residue asks the same grid for the cosh transform at a REAL frequency w',
# which reaches down to gap - w' and needs the wider range these points buy.
CD_NTAU = 24

# Newton for the contour-deformation quasiparticle equation
# (src/Solvers/qp_equation.py::solve_qp_equation_newton_guarded). The root is
# converged far tighter than an energy needs because the gradient chain
# differentiates the equation at that root: a residual of 1e-6 leaves Z and
# every adjoint that multiplies it off by the same relative amount.
QP_CD_NEWTON_TOL = 1e-11
QP_CD_NEWTON_MAX_ITER = 100

# Relative cutoff on the eigenvalues of S = C_ov C_ov^T when the auxiliary-boson
# (AB-G0W0) basis is built (src/SingleReference/GW/auxiliary_bosons.py). The
# auxiliary basis is rank-deficient whenever naux exceeds the rank of the
# particle-hole space, and the small eigenvalues are noise that S^{-1/2} would
# amplify.
AB_RCOND = 1e-10

# Below this fraction of the lowest auxiliary pole, a denominator of the
# sum-over-poles self-energy (src/SingleReference/GW/sum_over_poles.py) is
# resonant rather than compressible and the value is not to be believed.
SOP_CLEARANCE_MIN = 0.05
# Least-squares cutoff of the auxiliary-pole fit. F is a Cauchy-like matrix and
# is ill-conditioned by construction once the poles crowd; the cutoff is what
# keeps the amplitudes of a near-degenerate pair finite.
SOP_FIT_RCOND = 1e-12
# Auxiliary poles by default, set by the GRADIENT rather than the energy: the
# energy is converged at 8 and the derivative needs 12.
SOP_N_POLES = 12
# Fit the auxiliary poles on every n-th orbital column. They are COMMON to all
# orbitals, so a subset places them, and the least squares is dense and cubic
# in the columns kept.
SOP_FIT_STRIDE = 8

# Smallest pole strength Z a valence-window orbital must have to keep an
# explicitly solved quasiparticle energy on the BSE diagonal
# (src/SingleReference/GW/qp_states.py). A root below it carries less than half
# the spectral weight of the state, so what was solved is a satellite rather
# than the quasiparticle, and the orbital is better served by the frozen
# scissor.
QP_WINDOW_Z_MIN = 0.5

# The bare Laplace quadrature error a residue frequency must be carried to
# before the cosh transform of proj(tau) may stand in for an explicit chi0(w')
# (src/SingleReference/GW/real_screening.py::LaplaceRealScreening). It gates a
# REPRESENTATION, not an iteration: the transform is exact only while every
# pair energy d -/+ w' still lies inside the grid's fitted 1/y range, and past
# that the residue is not inaccurate but meaningless.
LAPLACE_SCREENING_TOL = 1e-8

# Upfolded BSE: dense diagonalization below this Hamiltonian dimension
# (src/SingleReference/BSE/bse_upfolded.py).
UPFOLDED_BSE_DENSE_LIMIT = 4000

# Largest Remez residual of the self-energy's tau -> omega transform at which a
# downfolded active-space model still holds. The residual, not the point count,
# is what a caller checks: it is set by the interplay of ntau with the
# self-energy's own energy range, which is far wider than the polarizability's.
SIGMA_FIT_ERROR_MAX = 1e-2

# BSE Casida solver: dense below this occupied-virtual pair count, the
# matrix-free ISDF/DF Davidson above (12000 pairs is ~1.2 GB per block and a
# few minutes of eigh). `solve_bse`'s solver='auto' compares BSE_DENSE_MAX_GB
# against 2 * n_ov**2 * 8 bytes, the dense route's own (A, B) storage --
# TDA is NOT exempt, since the dense route builds B whether or not `tda` is
# set and only the eigensolver drops it. BSE_DENSE_MAX_GB is the memory form
# of BSE_DENSE_MAX_NOV so the two spellings of the boundary cannot drift apart.
BSE_DENSE_MAX_NOV = 12000
BSE_DENSE_MAX_GB = 2 * BSE_DENSE_MAX_NOV**2 * 8 / 1e9

# Working-set cap for one block of AO derivative integrals (grad mu nu|lam sig)
# in the analytic gradient assembly (src/gradients/grad_engine.py); memory
# only, the flop count is unchanged.
DERIV_BLOCK_BYTES = 2 << 30

# Ha. The static <p|Sigma_x - v_xc|p> shift vanishes on a Hartree-Fock
# reference, where v_xc IS Sigma_x, but only analytically: evaluated there it
# is round-off, while a Kohn-Sham reference carries tenths of a Hartree. A
# gradient route that cannot differentiate the shift must therefore ask
# whether one is PRESENT by magnitude and not by nonzeroness, a shift this
# size having no force.
XC_SHIFT_GRADIENT_TOL = 1e-10

# Ha/Bohr. Two evaluations of one ISDF gradient agree to this and no better:
# a Newton root re-solved from a frozen seed lands one ulp (1e-16 Ha) from the
# unseeded root, and the interpolative fit's conditioning turns that into
# 4e-9 Ha/Bohr on water. A gate that asks for bitwise equality of two such
# gradients is asking for a property the arithmetic does not have.
ISDF_GRADIENT_FLOOR = 1e-8

# Nuclear finite-difference step (Bohr) for the gradient/derivative-coupling
# layer: the finite-difference gradient of a surface and the derivative
# couplings by eigenvector overlap.
NUCLEAR_FD_STEP = 1e-3

# Convergence of the orbital-response (Z-vector) equation
# antisym(Y_E + Y[fold(Lambda)]) = 0 solved in src/gradients/multipliers.py:
# lgmres runs to an absolute residual of this times max(1, |rhs|).
ORBITAL_MULTIPLIER_TOL = 1e-11

# Iteration ceiling of that lgmres solve. Reaching it is a failure, not a
# truncation: the solver refuses rather than return a half-converged multiplier.
ORBITAL_MULTIPLIER_MAX_ITER = 3000

# Two orbital energies closer than this make a degenerate pair, whose rotation
# the equation cannot determine. The pair is projected out of the solve; its
# right-hand side must vanish by symmetry and is asserted to.
ORBITAL_MULTIPLIER_DEGENERACY_TOL = 1e-8

# Acceptance threshold on the converged residual, relative to max(1, |rhs|).
# It guards against a silently wrong Lagrangian: an lgmres that stagnates
# short of `ORBITAL_MULTIPLIER_TOL` still returns info=0.
ORBITAL_MULTIPLIER_RESIDUAL_TOL = 1e-7

# BSE Casida solver on the gradient chain: the ISDF Davidson's root count and
# residual tolerance. Tighter than the solver's own default because
# Hellmann-Feynman reads the EIGENVECTORS and their convergence lands in the
# force directly.
BSE_DAVIDSON_NROOTS = 5
BSE_DAVIDSON_CONV_TOL = 1e-8

# Coulomb-metric fit error at which an ISDF atomic grid has stopped being a
# coarse grid and become a FAILED FIT; see `separable_ri.optimize_atomic_radii`.
ISDF_FIT_ERROR_FAILED = 1.0

# Working-set cap for one block of the three-centre integral (mu nu|P) when the
# ISDF fit gathers its test-set pairs; memory only, the integrals are unchanged.
THREE_CENTER_BLOCK_BYTES = 2 << 30

# Geometries whose rebuilt environment a gradient chain retains. The reuse is
# WITHIN one geometry -- an energy and a gradient ask for the reaction field
# several times at the point they are evaluated at -- and there is none across
# geometries, since a finite-difference sweep or a relaxation visits each one
# once and never returns.
ENVIRONMENT_CACHE_SIZE = 2

# Orbital-gradient ceiling max |F_ia| for a trustworthy gradient Lagrangian,
# which assumes the occupied-virtual Fock block vanishes. Symmetry hides a
# violation: a symmetric molecule looks converged and is wrong in the fourth
# digit of the force.
SCF_GRAD_TOL = 1e-9

# What a mean field is converged to when it will be DIFFERENTIATED. The
# Lagrangian assumes the occupied-virtual Fock block vanishes, so a loose SCF
# biases the force rather than degrading it gracefully.
SCF_DIFFERENTIABLE_CONV_TOL = 1e-14
SCF_DIFFERENTIABLE_GRAD_TOL = 1e-11

# What a mean field is converged to when only an ENERGY is taken from it. The
# pair above is at or below the noise floor of an exchange-correlation
# quadrature grid, so a Kohn-Sham reference spends its cycles chasing grid
# noise: acrolein/def2-SVP at BHLYP takes 69 cycles and 33 s to reach 1e-14
# against 13 cycles and 7.5 s to reach this, for the same excitation energy.
# Hartree-Fock carries no grid and does not care either way.
SCF_ENERGY_CONV_TOL = 1e-10
SCF_ENERGY_GRAD_TOL = 1e-7

# Overlap-based root following, <Psi(prev)|Psi(now)> from `properties.
# nonadiabatic.follow_state`. Below ROOT_FOLLOW_WEIGHT_MIN the state being
# followed has no counterpart in the displaced manifold: it left the solved
# window, or nroots is too small to hold it. ROOT_FOLLOW_MARGIN_MIN is the gap
# between the best overlap and the runner-up -- a SMALL MARGIN IS NOT A SMALL
# WEIGHT, since two roots that have mixed share the reference character and
# both overlaps are moderate.
ROOT_FOLLOW_WEIGHT_MIN = 0.5
ROOT_FOLLOW_MARGIN_MIN = 0.2

# What an orbital with no explicitly solved quasiparticle energy carries on the
# BSE diagonal: its mean-field eigenvalue, or that eigenvalue plus the frozen
# shift `GW.qp_states.calibrate_scissor` reads off the explicit roots at the
# reference geometry.
OUTSIDE_TREATMENTS = ('scissor', 'mean-field')

# Single-pole energy Omega_p (eV) of a solvent's ELECTRONIC response. Duchemin,
# Amblard and Blase, J. Chem. Theory Comput. 20, 9072 (2024) write
# eps_opt(w)^-1 = 1 + (eps_inf^-1 - 1) f(w; Omega_p) and show that the spatial
# and frequency degrees of freedom of the reaction field then decouple, so
# v_reac(w) = v_reac(0) f(w; Omega_p); on the imaginary axis f continues to
# g(iu) = Omega_p^2 / (u^2 + Omega_p^2), a scalar that damps vtilde above the
# solvent's own plasmon.
#
# 'fit' is a fit to the measured visible-UV response, 'f_sum' the value that
# makes the model's high-frequency tail match the f-sum rule -omega_p^2/u^2
# with omega_p = sqrt(4 pi n_e) the valence plasma frequency, Omega_p =
# omega_p / sqrt(1 - 1/eps_inf).
SOLVENT_PLASMON_EV = {
    'water':            {'fit': 21.0, 'f_sum': 32.5},
    'toluene':          {'fit': None, 'f_sum': 26.6},
    'carbon disulfide': {'fit': None, 'f_sum': 29.0},
}

# ---------------------------------------------------------------------------
# src/properties/: everything computed FROM a potential-energy surface
# ---------------------------------------------------------------------------

# Reciprocal centimetres per Hartree, for a vibrational frequency.
HARTREE_TO_CM = 219474.6313702
# Electron masses per unified atomic mass unit: a chemist's mass into the
# atomic units a Hessian is expressed in.
AMU_TO_ME = 1822.888486209

# Boltzmann constant in Hartree per Kelvin: k_B T of a rate expression, in the
# same energy unit as every gap and reorganization energy it is compared with.
BOLTZMANN_HARTREE_PER_KELVIN = 3.166811563e-6
# The atomic unit of time in seconds; a rate in inverse atomic time becomes s^-1
# by dividing by it.
ATOMIC_TIME_SECONDS = 2.4188843265857e-17
# The speed of light in atomic units, 1/alpha. It enters spontaneous emission
# as c^-3, so a rate is cubic in this number and quoting it to three digits is
# a 0.1% error in every radiative lifetime.
SPEED_OF_LIGHT_AU = 137.035999177

# Largest |H_ixjy - H_jyix| a finite-difference Hessian (src/properties/
# hessian.py) may carry before it is refused, relative to its own largest
# element. A central difference of an analytic force makes the two halves
# differ only through the force's noise divided by the step, so this is a
# measurement of that noise and nothing else. Measured on formaldehyde/
# cc-pVDZ/B3LYP at NUCLEAR_FD_STEP: 2.9e-07 with density-fitted exchange,
# 5.5e-03 with the interpolation. The first falls as h^2 and the second as
# h^1, and the second puts a 1421 cm^-1 mode at 2012 -- so this threshold
# separates a finite difference limited by its own truncation from one whose
# forces do not belong to a single smooth surface.
HESSIAN_FD_ASYMMETRY_TOL = 1e-3

# Minimum share of sum (X+Y)^2 in a BSE root's winning Gamma_i (x) Gamma_a
# channel for the root's point-group irrep label (src/properties/
# characters.py) to be trusted. Below this the transition amplitude is spread
# over more than one symmetry channel -- by a distorted geometry, or by an SCF
# that converged to a symmetry-broken solution -- and the label is not
# meaningful.
PURITY_FLOOR = 0.99

# Above this nuclear charge a scalar-relativistic reference stops being
# optional: the X2C spin-orbit operator and a non-relativistic mean field are
# then in visibly different pictures. Krypton, i.e. anything past the 3d row,
# which is where every phosphorescent emitter sits.
HEAVY_ATOM_Z = 36
# Assembling the singlet-triplet blocks of the effective relativistic
# Hamiltonian inconsistently shows up as non-Hermiticity, so it is refused
# rather than symmetrised away; the blocks are sums of a handful of
# contractions and land far inside this.
QDPT_HERMITICITY_TOL = 1e-12
# Davidson residual for a spin-orbit manifold. Looser than the gradient
# chains' because a coupling is a contraction over the WHOLE vector rather
# than a single eigenvalue, and averages its residual down.
SOC_MANIFOLD_CONV_TOL = 1e-5

# Cartesian geometry optimizer, Hartree/Bohr and Bohr; all four must hold.
# `opt_grad_max` is the largest force component AT THE CONVERGED GEOMETRY, the
# residual the convergence test is applied to. The name is not `grad_max`
# because that word also means the driving force at a fixed input geometry --
# a different number about a different geometry -- and a threshold that can be
# read as either is a threshold nobody can check a record against.
GEOM_OPT_CONV = {'opt_grad_max': 4.5e-4, 'grad_rms': 3.0e-4,
                 'step_max': 1.8e-3, 'step_rms': 1.2e-3}
# The superseded spelling of the same threshold, carried for one release so a
# caller reading GEOM_OPT_CONV['grad_max'] -- or passing a stored `conv` dict
# that spells it that way -- still gets the number it always got. `optimize`
# accepts either spelling and refuses the two disagreeing.
GEOM_OPT_CONV['grad_max'] = GEOM_OPT_CONV['opt_grad_max']

# Conformer search over the soft torsions of a twisted emitter.
# A bond is drawn when the internuclear distance is within this factor of the
# sum of the two covalent radii.
CONFORMER_BOND_SCALE = 1.25
# ... and it counts as SINGLE, hence torsionally soft, only above this fraction
# of the same sum, which stands in for a bond order the connectivity does not
# carry. A C=C at 1.33 A is 0.88 of 2 r_C and an amide C-N at 1.33 A is 0.92 of
# r_C + r_N, while a C-C single bond at 1.53 A is 1.01 and butadiene's central
# bond at 1.47 A is 0.97, so the cut separates the rotatable bonds from the
# rigid ones. Conjugation that shortens a formally single bond below it is read
# as rigid, which is the conservative error: a torsion is missed, never invented.
CONFORMER_SINGLE_BOND_RATIO = 0.93
# Torsion values enumerated per rotatable bond, as offsets from the input
# geometry; 3 is the anti/gauche+/gauche- pattern of an sp3-sp3 bond.
CONFORMER_TORSION_GRID = 3
# Starts the enumeration is truncated to. The product is
# CONFORMER_TORSION_GRID ** n_torsions, so this and not the grid size is what
# bounds the cost: five soft torsions on a donor-acceptor emitter is already
# 243 relaxations, each a full excited-state optimization.
CONFORMER_MAX_STARTS = 64
# Two relaxed structures are ONE conformer when they agree in energy to this
# (Hartree) AND in superposed heavy-atom RMSD to this (Angstrom). 1e-4 Ha is
# 2.7 meV, a tenth of k_B T at room temperature and far below any gap that
# changes a population; 0.15 A sits above the geometric residual of a loose
# relaxation and below the 0.5-1 A that separates a gauche minimum from an anti
# one.
CONFORMER_ENERGY_TOL = 1e-4
CONFORMER_RMSD_TOL = 0.15
# Graph automorphisms enumerated to make that RMSD symmetry-aware. Permuting
# equivalent atoms is what stops the two ends of a symmetric molecule from
# being reported as two conformers; the cap keeps a highly symmetric skeleton
# from enumerating a combinatorial group.
CONFORMER_MAX_AUTOMORPHISMS = 64
# Screening relaxation: GEOM_OPT_CONV loosened by this factor. The pass only has
# to identify which torsional basin a start fell into, and the survivors are
# relaxed again at full convergence, so a threshold an order of magnitude looser
# locates the basin at a fraction of the cycles.
CONFORMER_SCREEN_LOOSENING = 10.0
CONFORMER_SCREEN_CONV = {k: CONFORMER_SCREEN_LOOSENING * v
                         for k, v in GEOM_OPT_CONV.items()}
# Temperature (Kelvin) the conformer populations are reported at.
CONFORMER_TEMPERATURE = 300.0

# The ISDF interpolation grid `properties.surfaces.potential_energy_surface`
# asks for when the caller names none: the level of ISDF_GRID_ACCURACY
# validated to 4 meV on the three lowest BSE roots. A basis or an element with
# no row at it is REFUSED there rather than dropped to a coarser default,
# which is a different factorization and not a coarser one.
SURFACE_GRID_ACCURACY = 'G2'

# Hartree in meV. Derived from HARTREE_TO_EV rather than spelled again, so the
# refreeze drift a relaxation record reports in meV and the excitation energy
# it reports in eV can never be two different conversions.
HARTREE_TO_MEV = 1000.0 * HARTREE_TO_EV

# Seconds `properties.excitations` waits for git to name the commit a record
# was produced on. Provenance is not worth blocking a calculation for: the
# stamp falls back to 'unknown' when the call does not return in time.
GIT_PROVENANCE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Self-energy method registry
# ---------------------------------------------------------------------------
METHOD_REGISTRY = {
    'GW':         {'vertex_mode': 'GW',         'force_rpa_casida': True,  'needs_vertex': False, 'needs_triplet': False},
    'GW@RPA':     {'vertex_mode': 'GW',         'force_rpa_casida': True,  'needs_vertex': False, 'needs_triplet': False},
    'GW@BSE':     {'vertex_mode': 'GW',         'force_rpa_casida': False, 'needs_vertex': False, 'needs_triplet': False},
    'GW@TDHF':    {'vertex_mode': 'GW',         'force_rpa_casida': False, 'needs_vertex': False, 'needs_triplet': False},
    'GWGammaInf': {'vertex_mode': 'GWGammaInf', 'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD1':       {'vertex_mode': 'PSD1',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD2':       {'vertex_mode': 'PSD2',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD4':       {'vertex_mode': 'PSD4',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD5':       {'vertex_mode': 'PSD5',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD6':       {'vertex_mode': 'PSD6',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD7':       {'vertex_mode': 'PSD7',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
    'PSD8':       {'vertex_mode': 'PSD8',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': False},
    'PSD9':       {'vertex_mode': 'PSD9',       'force_rpa_casida': False, 'needs_vertex': True,  'needs_triplet': True},
}


def get_method_info(method):
    """
    Looks up `method` (case-insensitive) in METHOD_REGISTRY
    """
    key = method.upper()
    for registered_key, info in METHOD_REGISTRY.items():
        if registered_key.upper() == key:
            return info
    raise ValueError(
        f"Unknown self-energy method '{method}'. Available: {sorted(METHOD_REGISTRY)}"
    )
