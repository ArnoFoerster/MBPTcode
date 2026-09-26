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

# Relative eigenvalue floor of the Coulomb metric's SQUARE ROOT, the gauge of
# the separable factors (D = M^T V^1/2): a direction below it is the metric's
# numerical null space and is dropped from the root rather than carried at
# the rounding level. Far below AUX_METRIC_LINDEP because the root is never
# inverted: a small eigenvalue enters it as w^1/2, which is harmless.
AUX_METRIC_ROOT_FLOOR = 1e-12

# How negative, relative to the largest eigenvalue, a DRESSED metric v + vtilde
# may be before its root is refused. v + vtilde is a positive kernel, so a
# negative eigenvalue beyond rounding means the discretized reaction field
# over-screens the bare interaction -- an error of the cavity or of eps, not
# a truncation to be dropped.
AUX_METRIC_INDEFINITE_TOL = 1e-10

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

# Grid points per tile of the row-distributed ISDF fit (`fit_rows`): the Gram
# tiles, the Cholesky panels and diagonal blocks, the substitution blocks, the
# three-centre contraction and the projections all run on tiles of this edge,
# owned block-cyclically. FIXED, never derived from the rank count: a GEMM's
# bits depend on its call shape, so the same tile sequence whoever owns it is
# what makes the factor, the solve and D bitwise identical at every rank
# count. 512 keeps the trailing update's inner dimension at two BLAS panels
# and cuts the chlorophyllide hexamer (117762 points) into 231 tiles -- 29 a
# rank at 8 ranks, 0.48 GB a gathered panel -- and pentacene (5328) into 11.
FIT_CHOLESKY_BLOCK = 512

# How many times the replicated fit's own reassociation response a different
# realization of the same fit may sit from it. The response is measured on the
# run: the replicated fit re-run with its three-centre blocks cut per shell and
# accumulated in reverse order, which moves D, the quasiparticle energies, W(0)
# and the BSE roots by what one reordering of an eps-level sum costs once the
# balanced Gram matrix (cond ~ 1e8) amplifies it. The row-distributed fit
# differs from the replicated one by several such reorderings at once (Gram
# tiles, a blocked Cholesky, per-tile contractions), so its distance is a few
# responses; ten covers that and still fails a defect, which moves the fit by
# whole digits.
FIT_REASSOCIATION_K = 10

# How many times a MEASURED repeat or reassociation response of a number that
# number may move when a comparison spans two SCF runs, two threaded pyscf K
# builds or a sum reduced in another order. pyscf's OpenMP GEMM (`lib.ddot`)
# adds its K-split partials in thread-arrival order, so at 16 threads no pyscf
# result repeats bit for bit, and a fixed tolerance gates one machine's BLAS
# rather than the method; the response is measured where the comparison runs
# (the serial force re-associated on one BLAS thread, or the spread of repeated
# evaluations on one mean field). Three puts the BSE@GW excitation force's gate
# at 5.4e-8 Ha/Bohr on water/cc-pVDZ, where that response is 1.8e-8, far under
# the 1.49e-3 a real defect moved it (ranks differentiating two grids).
COMPOSED_GRAD_K = 3

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


# BLAS threads at or above which a pyscf-OpenMP kernel is worth running with
# the BLAS pool held at one thread (`Base.utils.threads.blas_single_threaded`).
# The two pools spin against each other, and the cost of that contention is
# what the wrap removes: on sixteen cores the SCF is 117.5 s with both pools
# wide and 9.4 s with BLAS at one. It scales with the cores there are to
# contend over, so on two or three threads there is nothing to win and the
# entry cost of the wrap is all that is left -- which is what this gates.
BLAS_WRAP_MIN_THREADS = 4


# Fraction of a SLURM allocation's memory that `Base.utils.memory.
# allocation_max_memory_mb` hands to pyscf's own `max_memory`. pyscf's buffers
# (the DF tensor, the numint grid) are not the whole process: the probe that
# found the pentacene cc-pVTZ regression measured 7.8 GB RSS at anthracene
# cc-pVTZ with `max_memory` itself capped at 4000 MB, so mo_coeff, the ISDF
# factors and python's own overhead already run close to as large as the
# capped buffers do. Handing pyscf the whole allocation would leave that other
# half nowhere to go and the job would be killed for memory it never asked
# pyscf for; 0.6 leaves it 40% of the node.
ALLOCATION_MEMORY_FRACTION = 0.6


# Pseudo-inverse cutoff of the minimax transform fits
# (`Base.utils.time_frequency.minimax_transform_weights`), relative to the
# largest singular value of that row's design matrix. The fit is a per-point
# least squares solved through an SVD, and below `_REGULARIZATION_ABOVE` points
# it carries NO Tikhonov term, so the filter is 1/S: an exactly zero singular
# value gives 0/0 and one below 1.5e-162 squares to zero and gives an
# infinity. Either fills a row of the transform with non-numbers that then
# propagate into W(i.tau) and the self-energy. Both are reached in production
# -- a minimax tau point large enough that exp(-x tau) underflows over the
# whole node range leaves an exactly zero COLUMN in the design matrix.
# This separates arithmetically zero from small: the smallest relative
# singular value any grid in this code reaches is 2e-17, and GreenX inverts
# those deliberately (conditioning is the regularization's business, not this
# cutoff's), so the value sits far below anything a fit uses and drops only
# what the arithmetic itself cannot invert.
TRANSFORM_FIT_RCOND = 1e-100


# Lebedev order of the cavity surface: 11 is 50 points per sphere against
# pyscf's own default of 29, which is 302. The continuum's cost is the SURFACE
# POTENTIAL of the density, n_ao^2 x n_surf, so this is the only knob that
# moves it -- and it buys almost nothing to refine. Measured: formaldehyde's
# solvation energy in toluene moves 0.4 meV between order 11 and order 29
# while the potential costs four times more at 29, and biphenyl's relaxed
# inter-ring torsion is IDENTICAL at orders 11 and 17 to a hundredth of a
# degree. The worry that a gradient might be more sensitive than an energy,
# the surface moving with the atoms, does not survive a full relaxation.
PCM_LEBEDEV_ORDER = 11


# The smallest residual the BSE/Casida Davidson (`LinearResponse.davidson`) may
# be asked for, as a multiple of machine epsilon times the largest particle-hole
# energy difference. The residual A x - omega x comes from block actions good to
# eps * ||A||, and ||A|| is that diagonal: the ISDF action at
# naphthalene/cc-pVDZ matches the dense blocks to 1.3 eps max(d). On
# Casida-shaped model problems (500 to 12000 pairs, max(d) 14 and 63 Ha) the
# solver reached 5e-14 to 1.5e-12 Ha, 15 to 110 eps max(d), and asked for less it
# stopped for want of a new direction. A tolerance under this -- 3e-12 Ha at
# max(d) = 15 Ha -- is refused before the first cycle.
DAVIDSON_FLOOR_EPS_MULTIPLE = 1e3
# pyscf `real_eig`'s own linear-dependence threshold, passed explicitly because
# the Davidson's preconditioner sizes corrections against it.
DAVIDSON_LINDEP = 1e-12
# Hartree. The residual below which a Davidson correction is SIZED for
# real_eig's linear-dependence test rather than handed over at its raw length
# r / (d - omega). Above it the raw length clears pyscf's absolute test -- the
# fabric probe converges anthracene/cc-pVTZ at 1e-5 on it in 18-22 cycles --
# and leaving it raw there keeps every solve at conv_tol >= 1e-5 bitwise what
# it was, since a root is only corrected while |r| > conv_tol.
DAVIDSON_SIZED_RESIDUAL = 1e-5
# The smallest part of a sized Davidson correction that may lie outside the
# trial subspace, as a fraction of the correction, for it to be added; nearly
# dependent directions fill the subspace and its projected (A-B) block loses
# definiteness. With every correction sized, 1e-6 broke down on 1 of 24
# Casida-shaped model problems at conv_tol 1e-8 and 2 at 1e-10, 1e-4 and 1e-3
# on none of 48. As used, sized below DAVIDSON_SIZED_RESIDUAL, 1e-3 converges
# 48 of 48 at 1e-8 and 47 at 1e-10.
DAVIDSON_MIN_NEW_FRACTION = 1e-3


# Share of a rank's own slice of the fitted tensor that the distributed DF
# build (`Base.distributed_df`) may hold in the transients of its exchange.
# The build cuts the three-centre integrals by AO-pair column and stores them
# by auxiliary row, so every round holds four buffers of (naux, block width):
# the two integral buffers, the piece this rank sends and the pieces it
# receives. Sizing the block against the SLICE rather than against free memory
# is what keeps the peak near the slice the split exists to fit -- at 0.5 the
# build peaks at 1.5x the slice, where a single exchange of the whole column
# layout would peak at 3x and give back what dividing the auxiliary index
# bought.
DF_EXCHANGE_TRANSIENT_FRACTION = 0.5

# The digest `mpi_grid.agreement` compares across ranks: an array's C-ordered
# bytes as 64-bit words u_i, summed mod 2^64 against W[i mod L] c[i div L],
# with W and c odd words of one PCG64 stream drawn from this seed and L the
# block length below. One changed word always changes the sum (an odd weight is
# invertible mod 2^64); changes in several words cancel only where their
# differences stand in the ratio of pseudo-random weights, ~2^-64. Any fixed
# seed serves: it changes every digest, never a verdict between ranks running
# one tree.
AGREEMENT_DIGEST_SEED = 1
# 64-bit words per block of that digest. The uint64 matrix-vector product over
# the blocks measured 18 GB/s at 4096 on the laptop (26 GB in 1.4 s) against
# 12 GB/s at 16384 and 65536, where blake2b over the same bytes runs 0.29 GB/s;
# the whole digest ran 10 to 15 GB/s with other jobs on the machine.
AGREEMENT_DIGEST_BLOCK = 4096

# The (A - B) probe replicated over ranks (`LinearResponse.davidson`) runs its
# own Lanczos where serially it runs ARPACK: eigsh holds one process-wide lock
# across its whole iteration, matvecs included, so rank threads cannot all be
# inside it at once. Sized by eigsh's own defaults: a basis of
# max(2k + 1, AMB_LANCZOS_NCV) vectors between thick restarts, and at most
# AMB_LANCZOS_MAXITER_PER_DIM operator applications per pair.
AMB_LANCZOS_NCV = 20
AMB_LANCZOS_MAXITER_PER_DIM = 10

# How the Hellmann-Feynman adjoint of a BSE root is realized. 'explicit'
# contracts the Casida vectors against the three-index blocks B[P, i, a] of
# `gradients.bse_isdf.bse_cache`, (naux, nocc, nvir) each, ten of them alive at
# once; 'grid' is the reverse of the ISDF block action
# (`LinearResponse.isdf_bse_adjoint`), which closes over the grid and forms no
# three-index block at all. The same derivative to rounding, not the same
# bits, so the vocabulary is spelled once for the chain and the record.
BSE_ADJOINTS = ('explicit', 'grid')

# Grid points per row tile of the grid BSE adjoint
# (`LinearResponse.isdf_bse_adjoint`). FIXED, never derived from a memory
# budget or a rank count: a GEMM's bits depend on its call shape, so one tile
# sequence is what lets the rows be handed to their owners unchanged. Each
# rank holds its own row tiles and streams the other ranks' column tiles,
# so a pass keeps a handful of (256, M) and (256, naux) tiles alive: a few
# GB at the chlorophyllide hexamer (M 117762), inside ISDF_TILE_GB.
BSE_ADJOINT_TILE_ROWS = 256

# GB of float64: the largest particle-hole block C_ov = B[:, occ, virt],
# naux * nocc * nvir * 8 bytes, that the explicit residue backend of the
# quasiparticle solves may build. The backend holds C_ov, the adjoint Cov_bar
# of the state in its reverse pass and one more block of that size inside each
# residue evaluation or push, three at once, whole on EVERY rank: at this limit
# 768 GB a rank, a 1 TB node with nothing beside it. Above it the route cannot
# run at any rank count -- the chlorophyllide hexamer's block is 2197 GB -- and
# the answer is the Laplace backend (residues below the particle-hole gap, from
# proj(tau)) or the pole model, which build no block.
EXPLICIT_RESIDUE_MAX_GB = 256.0

# Auxiliary rows per slab of W_bar in the distributed grid BSE adjoint
# (`LinearResponse.isdf_bse_adjoint`): slab s is summed by rank s % nranks
# over the grid tiles in tile order, and the slabs are then gathered verbatim.
# FIXED, like BSE_ADJOINT_TILE_ROWS and for the same reason: a slab is a
# GEMM's row count, so it must not follow the rank count. The grid index is
# cut in BSE_ADJOINT_TILE_ROWS tiles on both sides of every M^2 product, so
# one pass holds (2 nT + 4) (tile, tile) blocks, 6.3 MB, not (rows, M) tiles.
BSE_ADJOINT_AUX_ROWS = 256

# The largest count one MPI call may carry, in elements of the call's
# datatype. mpi4py hands the library an MPI_Count only where the library has
# the MPI-4 large-count routines (MPI_Allreduce_c, ...), which Open MPI 5.0.x
# does not provide; its fallback narrows the count to a C int and raises
# MPI_ERR_ARG past 2^31 - 1. proj(tau) at the chlorophyllide hexamer is 1.9e10
# doubles and D 3.3e9, so every collective of `Base.utils.mpi_grid` moves at
# most this many float64 per call, in windows; below it a call is one window,
# the unchunked call itself.
MPI_COUNT_MAX = 2 ** 31 - 1

# Fraction of what max_memory leaves that the distributed ISDF-K SCF
# (`Base.distributed_isdf_jk`) lets this rank's rows of Z take, for the two
# operators a range-separated functional asks for; above it Z's blocks are
# formed per K build from G = M^T V instead ('factored'). The serial ISDFJK's
# 'auto' mode takes a third, and so does this: the SCF's own buffers, the
# DFT grid and the fit's peak need the rest. Rank 0's decision.
ISDF_SCF_KERNEL_MEMORY_FRACTION = 0.33

# Auxiliary functions per slab of the metric V (or its attenuated form) that
# the distributed ISDF-K SCF evaluates at a time to form G = M^T V: a slab is
# a GEMM's column count, so it is fixed by the basis and not by the rank
# count, and no rank holds V whole (7.8 GB at the chlorophyllide hexamer,
# naux 31260; a slab is 128 MB).
ISDF_SCF_METRIC_SLAB = 512

# Bytes of one thread's temporary in the element-wise work of the ISDF fit's
# three-centre pass (`Base.separable_ri`): a shell block's test co-densities
# are screened and built a slab of grid rows at a time, and its kept
# (mu nu|P) rows gathered a slab of auxiliary functions at a time, so no
# block is ever whole (3 GB of co-densities at the chlorophyllide dimer) and
# each slab is made, scaled and reduced while it is still in cache. A cost
# knob only: every element is the same product whatever the slab.
FIT_ROW_CHUNK_BYTES = 4 << 20

# Edge, in grid points, of the square tiles in which the replicated ISDF fit
# (`Base.separable_ri`) writes the transpose of its Gram matrix's lower block
# triangle into the upper one while balancing it: a tile's transposed read
# touches one cache line and one page per source row, 256 of each, and 256
# measured fastest of 32..512 at two threads. A cost knob only: every element
# is the same two products whatever the tile.
FIT_TRANSPOSE_TILE = 256

# The trial space the Casida/BSE Davidson (`LinearResponse.davidson`) holds
# before pyscf's `real_eig` collapses it. real_eig sizes that space from its
# process-wide MAX_MEMORY (4000 MB), 103 trial pairs at the chlorophyllide
# dimer/cc-pVDZ (504,804 pairs), and a collapse keeps only the nroots Ritz
# vectors and applies the action to them again: under that bound 12 roots of
# anthracene/cc-pVDZ took 49 cycles and 1294 block actions where the solve
# never collapsed took 21 and 716, its top roots converging last. The space is
# raised to DAVIDSON_SPACE_CYCLES of real_eig's per-cycle increment (converged
# 12- and 16-root solves there held 353 and 379 pairs, 18 and 19 increments),
# never past DAVIDSON_SPACE_GB of its four pair-space-long holders per trial
# pair (990 pairs at the dimer) and never below real_eig's own bound, so every
# solve pyscf already held whole runs exactly as it did.
DAVIDSON_SPACE_CYCLES = 50
DAVIDSON_SPACE_GB = 16


# Bohr within which an explicit set of ISDF shell radii counts as THE shipped
# table row for the same (element, basis, auxbasis, counts). Both sides are
# float64 -- one read back from the JSON table, one held by the caller -- so
# agreement is either exact to the last bit or the two are different grids:
# neighbouring rows differ in the second decimal, and a run-time
# re-optimization onto another local minimum lands 1e-3 Bohr or further away.
# This tolerance absorbs decimal round-tripping and nothing else; it is not a
# statement that two grids this close are interchangeable.
ISDF_RADII_MATCH_TOL = 1e-10

# Bytes of W(i.omega) - I per chunk of the omega -> tau transform of the
# space-time self-energy (`GW.imaginary_time`): the frequencies are folded into
# Wt(i.tau) a chunk at a time, two at the chlorophyllide dimer/cc-pVTZ (0.87 GB
# a frequency). The chunk is also the association of that sum -- one GEMM per
# chunk, added in chunk order -- which the row-distributed transform keeps, so
# it is fixed by naux alone and never by the rank count.
SCREENED_CHUNK_BYTES = 2 << 30
