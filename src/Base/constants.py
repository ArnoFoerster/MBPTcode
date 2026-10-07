"""Numeric defaults and self-energy method registry shared across src/SingleReference/."""
import math

# Physical constants, CODATA 2018. Defined here and nowhere else: import them,
# never re-spell the digits, so every route reports the same number.
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.52917721092

# Lorentzian broadening for self-energy denominators / spectral functions.
DEFAULT_BROADENING_ETA = 1e-3

# Relative eigenvalue floor for inverting an auxiliary-basis metric. The
# long-range metric of a range-separated hybrid is numerically singular
# (erf(omega r)/r is smooth, so tight auxiliary functions go linearly dependent
# under it), so the fit is inverted on its numerical range rather than solved
# through.
AUX_METRIC_LINDEP = 1e-10

# Relative eigenvalue floor of the Coulomb metric's square root, the gauge of
# the separable factors (D = M^T V^1/2): a direction below it is the metric's
# numerical null space and is dropped from the root rather than carried at
# the rounding level. Far below AUX_METRIC_LINDEP because the root is never
# inverted: a small eigenvalue enters it as w^1/2, which is harmless.
AUX_METRIC_ROOT_FLOOR = 1e-12

# How negative, relative to the largest eigenvalue, a dressed metric v + vtilde
# may be before its root is refused. v + vtilde is a positive kernel, so a
# negative eigenvalue beyond rounding means the discretized reaction field
# over-screens the bare interaction: an error of the cavity or of eps, not a
# truncation to be dropped.
AUX_METRIC_INDEFINITE_TOL = 1e-10

# Lebedev shell counts per atom of the default separable-RI / ISDF grid,
# 148 points per atom.
ISDF_DEFAULT_COUNTS = {'A1': 8, 'A2': 5, 'A3': 3, 'B1': 1}

# Validated ISDF interpolation grids: {basis: {level: (A1, A2, A3, B1)}}, the
# Lebedev sub-shell replica counts measured to reach an accuracy; the only
# place a validated count is written down.
#
# A level is an accuracy target, not a size: the largest deviation of the three
# lowest BSE roots from `solve_bse_df` at the same mean field, over ten
# molecules spanning H C N O P S Si B with the roots matched between the two
# routes. G1 < 8 meV, G2 < 4 meV, G3 < 2 meV, at 6*A1 + 8*A2 + 12*A3 + 24*B1
# points per atom.
#
# The ladder is not monotone, so an entry licenses its own count and no other.
# A missing level is a refusal, not a fallback: nobody measured it, so a
# neighbouring level, a larger count and another basis are all equally
# unlicensed.
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
# start are a different grid and not the one that was scored.
ISDF_GRID_N_START = 8

# Highest auxiliary angular momentum the ISDF interpolation grid represents.
ISDF_MAX_AUX_L = 5

# Auxiliary set standing in for an element the augmented RI family lacks, keyed
# (element, orbital basis). Mg has no aug-cc-pV{D,T}Z-ri. Lowest five BSE singlets
# of MgH2, MgF2, Mg(OH)2 against the autoaux set, Mg set varied, others genuine:
# cc-pVQZ-ri stays within 1.0 meV (cc-pVDZ-ri up to 3.9, cc-pVTZ-ri 1.7 at aug-DZ),
# and it reaches l = 5 = ISDF_MAX_AUX_L (src/Base/basis/ri_fallback.py).
ISDF_AUX_FALLBACK = {('Mg', 'aug-cc-pvdz'): 'cc-pvqz-ri',
                     ('Mg', 'aug-cc-pvtz'): 'cc-pvqz-ri'}

# CasidaSolver-only: TDA-shortcut threshold and omega^2 clipping before sqrt().
CASIDA_NUMERICAL_EPS = 1e-6

# Newton solve of the pair-channel Riccati equation B^+ + D T + T A + T B T = 0
# (ring-CCD / ladder-CCD amplitudes, LinearResponse/riccati.py): converged when
# max|residual| falls below the tolerance. Newton is quadratic from T = 0 for a
# stable channel, so the cap is a safeguard, not a budget.
RICCATI_CONV_TOL = 1e-10
RICCATI_MAX_ITER = 50
# The DF-streamed ladder amplitudes iterate Jacobi steps with DIIS instead of
# Newton (no pair-space Sylvester solve at scale): linear, not quadratic.
RICCATI_MAX_ITER_JACOBI = 200

# N^{-1/2} v by Lanczos (ADC/solve.py apply_inverse_sqrt), the Faddeev-ADC(3)
# metric turned into a plain symmetric eigenproblem vector by vector: stop when
# two successive Krylov approximations agree to tol*|v|. The metric is 1 + dN
# with dN small, so a handful of steps settle it; the cap is a safeguard.
INV_SQRT_LANCZOS_TOL = 1e-12
INV_SQRT_LANCZOS_MAX_STEPS = 100

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
# |a| > 4.1 eV at s = 100. Off the diagonal the kernel K(a, b) is at most
# (1 + sqrt 2) / (2 max(|a|, |b|)) at every s, so a virtual's near-pole terms
# stay out of its couplings to the occupied orbitals, the block that rotates the
# density; mode A carries them there at size 1/eta. Marie and Loos recommend
# 500 or 1000 (accuracy plateau from s = 50, judged from a HF start). On water
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
# <S^2> above S(S+1) by more than this on an unrestricted reference is warned:
# the quasiparticle energies of a spin-contaminated determinant describe a
# mixture of spin states, which no correction downstream undoes.
UHF_SPIN_CONTAMINATION_WARN = 0.1

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
# owned block-cyclically. Fixed, never derived from the rank count: a GEMM's
# bits depend on its call shape, so the same tile sequence whoever owns it is
# what makes the factor, the solve and D bitwise identical at every rank
# count. 512 keeps the trailing update's inner dimension at two BLAS panels.
FIT_CHOLESKY_BLOCK = 512

# How an ISDF fit is realized: 'replicated', the whole fit on every rank
# (`fit_M_streaming`), or 'rows', `separable_ri.fit_rows` by the grid-row tiles
# above. A realization, not a functional: both solve one estimator, the Gram
# matrix over every product pair and F D^T over the screened pairs.
FIT_REALIZATIONS = ('replicated', 'rows')

# GB of float64 the fit adjoint formed whole may hold on one rank: the test set
# over every product pair and the auxiliaries, (M, nao*n2 + naux), three times
# (D_test, its adjoint, one temporary), F of the same width and the whole
# (nao, nao, naux) three-centre tensor with its adjoint, estimated before any
# is allocated (`isdf_derivatives.whole_fit_adjoint_gb`). The size grows as
# M * nao^2 and does not divide by the rank count, since every rank forms it.
# Above the cap the whole form refuses and names the row fit, whose adjoint
# runs in the fit's grid-row tiles (`fit='rows'`).
WHOLE_FIT_ADJOINT_MAX_GB = 256.0

# The most a composed force may move, in Ha/Bohr, when the Casida-level seeds
# its fold reads change by one ulp per element: 1e-9 of a 0.1 Ha/Bohr force.
# A fit adjoint formed from single solves of the Gram matrix (cond ~ 2e8)
# carries such a change to ~1e-12 on water/cc-pVDZ, whole fit and row fit
# alike; applying G^-1 twice to an (nk, nk) product would carry it to ~1e-8
# (tests/test_fit_adjoint_stability.py).
FIT_ADJOINT_ULP_RESPONSE_MAX = 1e-10

# How far, in Ha/Bohr and Ha, a composed force and its energy over ranks may
# sit from the same calculation serially on the same mean field. A rank split
# re-associates the reductions -- chi0(0) behind the static W, the tau and
# frequency sweeps, the Davidson's products -- which moves their results by a
# few ulp; the chain carries an ulp change of what reaches its fold to at most
# FIT_ADJOINT_ULP_RESPONSE_MAX (water/cc-pVDZ: 8e-13 to 1.5e-12 Ha/Bohr at 2
# and 3 ranks), and a total energy of ~100 Ha by a few of its ulp (1.4e-14).
# It assumes the factors themselves are not re-associated: the fit's
# three-centre sum is one block on the molecules gated here, and where a rank
# split cuts it, its last bits reach the force through the fit's Gram matrix
# (cond ~ 2e8) at ~1e-8, as a one-BLAS-thread repeat does (FIT_REALIZATION_
# FORCE_TOL).
RANK_SPLIT_FORCE_TOL = 1e-10
RANK_SPLIT_ENERGY_TOL = 1e-12

# How far a force may move between two mean fields each converged to
# SCF_DIFFERENTIABLE_GRAD_TOL (two SCF runs, or the SCF divided over ranks
# against the serial one). Their orbitals differ by at most ~that / gap,
# 4e-11 for gaps above 0.25 Ha; the factors D depend on the geometry alone and
# do not move; the force is linear in the orbital difference with
# coefficients the size of the one-electron derivative integrals, below
# 10 Ha/Bohr per unit density on these molecules: 4e-10, with headroom.
CONVERGED_SCF_FORCE_TOL = 1e-9

# How far a force may move between two realizations of the one ISDF fit
# (the replicated fit_M_stable and the tiled row fit, or the fit with its
# arithmetic re-associated). Both solve the Gram matrix of cond ~ 2e8, so
# they agree in D only to cond(G) u ~ 3e-8 relative, in its near-null space;
# a force reads D through branches below 1 Ha/Bohr here, so it moves by at
# most ~3e-8 (water/cc-pVDZ: 1.5e-8 row fit against whole, 7.9e-9 under a
# one-BLAS-thread repeat). A different estimator -- the Gram matrix over the
# screened pairs alone -- moves it by 1e-3.
FIT_REALIZATION_FORCE_TOL = 1e-7
# The same for a total energy, in Ha. An energy reads D through the products
# the fit reproduces (Z = D D^T contracted with densities), which the Gram
# matrix's near-null space does not reach to first order: the two
# realizations' 3e-8 relative in D moves a ~100 Ha energy by ~1e-12 (water/
# cc-pVDZ: 9.1e-13 row fit against whole on a shape-skewed BLAS).
FIT_REALIZATION_ENERGY_TOL = 1e-10
# The same, relative to its norm, for D itself and for the fit branch
# contracted with RANDOM adjoints, both of which reach the Gram matrix's
# near-null space, where the two blockings agree only to cond(G) u ~ 3e-8 per
# solve; the branch applies the solve on both sides of G_bar: 6e-8, here
# with headroom for the denser grids' cond(G) (water/cc-pVDZ: D 2.4e-8, the
# branch 2.5e-8). A different estimator moves either by ~1e-3.
FIT_REALIZATION_REL_TOL = 1e-6
# The same, relative, for what reads D through the products the fit
# reproduces -- W(0), the quasiparticle and the BSE energies: second order in
# the near-null-space difference, plus their solvers' own convergence
# (W(0) 1.5e-13, the BSE roots 4.8e-11 on ethylene/cc-pVDZ).
FIT_REALIZATION_OBSERVABLE_REL_TOL = 1e-9

# How far a sum of well-conditioned terms (one sign or a few orders apart),
# re-associated over grid tiles, may move relative to its result: N u for
# N ~ 1e3 grid points, ~1e-13; 1e-12 with headroom (1e-14 measured on the
# collocation adjoint and X_mo^T X_bar).
REASSOCIATED_SUM_REL_TOL = 1e-12

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

# How many times a measured repeat or reassociation response of a number that
# number may move when a comparison spans two SCF runs, two threaded pyscf K
# builds or a sum reduced in another order. pyscf's OpenMP GEMM (`lib.ddot`)
# adds its K-split partials in thread-arrival order, so at 16 threads no pyscf
# result repeats bit for bit, and a fixed tolerance gates one machine's BLAS
# rather than the method; the response is measured where the comparison runs
# (the serial force re-associated on one BLAS thread, or the spread of repeated
# evaluations on one mean field). Three puts the BSE@GW excitation force's gate
# at 5.4e-8 Ha/Bohr on water/cc-pVDZ, where that response is 1.8e-8.
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
# gradient (src/Base/pcm_derivatives.py). The solvation energy is quadratic in
# the density it is built from, so the central difference this scales carries
# no truncation error and the value is a conditioning choice only (measured
# step-independent to 3e-15 from 1e-1 down to 1e-3).
PCM_CROSS_TERM_STEP = 1e-2

# Contour deformation of the GW self-energy
# (src/SingleReference/GW/contour_deformation.py). A pole of G at
# |omega - eps_q| below RESIDUE_ON_CONTOUR_TOL counts as on the contour.
# QP_POLE_OFFSET is how far off an orbital energy the quasiparticle iteration
# is kept. The integral term takes its singular part in closed form
# (`cd_integral_weights`), so Sigma is continuous through every orbital energy
# and the offset sets only the Newton's start and the band a frozen path
# records.
RESIDUE_ON_CONTOUR_TOL = 1e-10
QP_POLE_OFFSET = 1e-3
# The smallest band the Newton may relax to when a quasiparticle root lies
# inside the guard, where the guard undoes every step and the iteration
# deadlocks at a fixed point that is not the root. Ten times the on-contour
# tolerance: Sigma is exact at that distance, and a dense spectrum can put a
# real root within 1e-6 Ha of a neighbouring level.
QP_POLE_OFFSET_MIN = 1e-9
# Below this pole strength a converged root is a satellite, not the
# quasiparticle: Z = 1/(1 - dSigma/dw) collapses beside a pole of Sigma at
# eps_q -/+ Omega_s.
QP_POLE_STRENGTH_MIN = 0.1
# Margin (Hartree) past the residue frequencies of the Newton start that the
# tau grid must carry for the Laplace residue backend to be chosen: the root
# moves by the quasiparticle correction, a few tenths of an eV to 2 eV.
RESIDUE_FREQ_MARGIN = 0.1
# Gauss-Legendre points on the imaginary-frequency integral of the dRPA
# correlation energy in the space-time route; converged to 1e-8 Ha at 40.
RPA_ENERGY_NFREQ = 40
# Gauss-Legendre points on the imaginary-frequency half of a contour
# deformation.
CD_NFREQ = 64
# Imaginary-time points behind the contour-deformation grid. The cosine
# transform onto the imaginary-frequency quadrature would be converged at 18;
# a residue asks the same grid for the cosh transform at a real frequency w',
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
# Auxiliary poles by default, set by the gradient rather than the energy: the
# energy is converged at 8 and the derivative needs 12.
SOP_N_POLES = 12
# Fit the auxiliary poles on every n-th orbital column. They are common to all
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
# representation, not an iteration: the transform is exact only while every
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
# matrix-free ISDF/DF Davidson above (12000 pairs is ~1.2 GB per block).
# `solve_bse`'s solver='auto' compares BSE_DENSE_MAX_GB against
# 2 * n_ov**2 * 8 bytes, the dense route's own (A, B) storage; TDA is not
# exempt, since the dense route builds B whether or not `tda` is set and only
# the eigensolver drops it. BSE_DENSE_MAX_GB is the memory form
# of BSE_DENSE_MAX_NOV so the two spellings of the boundary cannot drift apart.
BSE_DENSE_MAX_NOV = 12000
BSE_DENSE_MAX_GB = 2 * BSE_DENSE_MAX_NOV**2 * 8 / 1e9

# Working-set cap for one block of AO derivative integrals (grad mu nu|lam sig)
# in the analytic gradient assembly (src/gradients/grad_engine.py); memory
# only, the flop count is unchanged.
DERIV_BLOCK_BYTES = 2 << 30

# Ha. The static <p|Sigma_x - v_xc|p> shift vanishes on a Hartree-Fock
# reference, where v_xc is Sigma_x, but only analytically: evaluated there it
# is round-off, while a Kohn-Sham reference carries tenths of a Hartree. A
# gradient route that cannot differentiate the shift must therefore ask
# whether one is present by magnitude and not by nonzeroness, a shift this
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
# Hellmann-Feynman reads the eigenvectors and their convergence lands in the
# force directly.
BSE_DAVIDSON_NROOTS = 5
BSE_DAVIDSON_CONV_TOL = 1e-8
# Residual |r| a root the force reads needs, where the Davidson stopped above
# BSE_DAVIDSON_CONV_TOL: its eigenvector error is |r| over the gap to the
# roots outside the trial space, and the force moves by it at first order.
# On C1 ethylene/cc-pVDZ (SOP chain, 5 roots spanning 0.073 Ha) the S1 force
# moves by 7e-3 |r| Ha/Bohr down to the fit's floor, 3e-9 at |r| = 1e-7; a
# spectrum ten times denser keeps 1e-7 under ISDF_GRADIENT_FLOOR.
BSE_FORCE_RESIDUAL_TOL = 1e-7

# Coulomb-metric fit error at which an ISDF atomic grid has stopped being a
# coarse grid and become a failed fit; see `separable_ri.optimize_atomic_radii`.
ISDF_FIT_ERROR_FAILED = 1.0

# Working-set cap for one block of the three-centre integral (mu nu|P) when the
# ISDF fit gathers its test-set pairs; memory only, the integrals are unchanged.
THREE_CENTER_BLOCK_BYTES = 2 << 30

# Auxiliary functions per tile of the fitted Fock skeleton
# (`Base.skeleton_tiles`): tile t of consecutive auxiliary shells is rank
# t % size's, and its (mu nu|P) and derivative integrals are made, contracted
# and dropped in slabs of the first AO index under THREE_CENTER_BLOCK_BYTES.
# Fixed by the basis, never by the rank count, so each tile's contribution is
# the same bits whoever computes it. A tile of the exchange's
# L = (mu nu|P) C_occ holds nao * nocc * SKELETON_AUX_TILE doubles, and the
# auxiliary set splits into naux / SKELETON_AUX_TILE tiles, several a rank on
# a large system.
SKELETON_AUX_TILE = 32

# Points per tile of the xc skeleton's Becke-grid sum (`Base.skeleton_tiles`,
# `xc_grid_skeleton`): tile t of the mean field's own sorted grid is rank
# t % size's, a multiple of pyscf's 56-point screening block. Fixed by the
# grid, never by the rank count or the free memory, so each tile's addend is
# the same bits whoever computes it. One tile holds the AO values to second
# order, 10 * nao * XC_SKELETON_TILE doubles.
XC_SKELETON_TILE = 1008

# Geometries whose rebuilt environment a gradient chain retains. The reuse is
# within one geometry (an energy and a gradient ask for the reaction field
# several times at the point they are evaluated at); there is none across
# geometries, since a finite-difference sweep or a relaxation visits each one
# once.
ENVIRONMENT_CACHE_SIZE = 2

# Orbital-gradient ceiling max |F_ia| for a trustworthy gradient Lagrangian,
# which assumes the occupied-virtual Fock block vanishes. Symmetry hides a
# violation: a symmetric molecule looks converged and is wrong in the fourth
# digit of the force.
SCF_GRAD_TOL = 1e-9

# What a mean field is converged to when only an energy is taken from it. An
# energy test at 1e-14 is at or below the noise floor of an
# exchange-correlation quadrature grid, so a Kohn-Sham reference spends its
# cycles chasing grid noise -- several times the cycles this pair takes, for
# the same excitation energy. Hartree-Fock carries no grid and does not care
# either way.
SCF_ENERGY_CONV_TOL = 1e-10
SCF_ENERGY_GRAD_TOL = 1e-7

# Orbital-gradient norms |g| = ||2 F_vo||_F at which an SCF's record notes the
# first cycle below: the energy-only threshold, the force path's, and two
# decades between.
SCF_HISTORY_GRAD_MARKS = (1e-7, 1e-9, 1e-11)

# Bohr. Largest atomic displacement over which an SCF along a walk starts
# from the previous geometry's converged density rather than the atomic
# guess: optimizer steps and finite-difference displacements are below the
# optimizer's trust ceiling of 0.5 Bohr; a reoriented copy of the molecule is
# not, and starts afresh.
SCF_WARM_START_MAX_SHIFT = 0.5

# The second-order finish of an SCF (`scf_convergence.newton_finished`):
# pyscf's DIIS loop is run to an orbital gradient of SCF_NEWTON_FINISH_FROM
# with the energy-only conv_tol, then exact Newton steps on the orbital
# Hessian take it to the mean field's own conv_tol_grad. Each step solves
# H x = -g by preconditioned conjugate gradients to a residual of
# SCF_NEWTON_RESIDUAL_FRACTION x conv_tol_grad (the next gradient is that
# residual plus O(|x|^2), so one step from 1e-7 reaches a 1e-11 target),
# with at most SCF_NEWTON_MAX_HOPS Hessian products (one response Fock build
# each) per step and SCF_NEWTON_MAX_STEPS steps.
SCF_NEWTON_FINISH_FROM = SCF_ENERGY_GRAD_TOL
SCF_NEWTON_DIIS_CONV_TOL = SCF_ENERGY_CONV_TOL
SCF_NEWTON_RESIDUAL_FRACTION = 0.1
SCF_NEWTON_MAX_HOPS = 40
SCF_NEWTON_MAX_STEPS = 4
# Ha. Smallest |2 (F_aa - F_ii)| the Newton solve's diagonal preconditioner
# divides by: a near-degenerate pair across the gap is not divided by zero.
SCF_NEWTON_PRECOND_FLOOR = 1e-2

# What a mean field is converged to when it will be differentiated. The
# Lagrangian assumes the occupied-virtual Fock block vanishes, so a loose SCF
# biases the force rather than degrading it gracefully: the force moves by
# |dF| <= 1.2 |g| (|g| = ||2 F_vo||_F; S1 forces of formaldehyde, ethylene
# and acetaldehyde/cc-pVDZ at HF, PBE0 and LRC-wPBEh on ISDF-K), so the
# orbital gradient is what this pair converges. The energy test is made
# non-binding: an energy at |g| is converged to O(|g|^2 / gap), and one ulp
# of a total energy is 2.2e-16 |E|, so |dE| < 1e-14 is met above ~50 Ha only
# by a cycle that repeats the density bit for bit, which on a large system
# need never happen. At 1e-10 the forces sit within 5e-11 Ha/Bohr of the
# 1e-14 runs', their own reproducibility floor between two SCF runs.
SCF_DIFFERENTIABLE_CONV_TOL = SCF_ENERGY_CONV_TOL
SCF_DIFFERENTIABLE_GRAD_TOL = 1e-11

# Overlap-based root following, <Psi(prev)|Psi(now)> from `properties.
# nonadiabatic.follow_state`. Below ROOT_FOLLOW_WEIGHT_MIN the state being
# followed has no counterpart in the displaced manifold: it left the solved
# window, or nroots is too small to hold it. ROOT_FOLLOW_MARGIN_MIN is the gap
# between the best overlap and the runner-up; a small margin is not a small
# weight, since two roots that have mixed share the reference character and
# both overlaps are moderate.
ROOT_FOLLOW_WEIGHT_MIN = 0.5
ROOT_FOLLOW_MARGIN_MIN = 0.2

# What an orbital with no explicitly solved quasiparticle energy carries on the
# BSE diagonal: its mean-field eigenvalue, or that eigenvalue plus the frozen
# shift `GW.qp_states.calibrate_scissor` reads off the explicit roots at the
# reference geometry.
OUTSIDE_TREATMENTS = ('scissor', 'mean-field')

# Single-pole energy Omega_p (eV) of a solvent's electronic response. Duchemin,
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
    'dichloromethane':  {'fit': None, 'f_sum': 32.8},
}

# ---------------------------------------------------------------------------
# src/properties/: everything computed from a potential-energy surface
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

# hc in Hartree nm: a photon of E Hartree has the vacuum wavelength
# HARTREE_WAVELENGTH_NM / E nm (lambda = 2 pi c / E in Bohr).
HARTREE_WAVELENGTH_NM = 2.0 * math.pi * SPEED_OF_LIGHT_AU * BOHR_TO_ANGSTROM / 10.0
# FWHM of a Gaussian per standard deviation, 2 sqrt(2 ln 2).
GAUSSIAN_FWHM_PER_SIGMA = math.sqrt(8.0 * math.log(2.0))
# Exponent F below which a Franck-Condon density e^F times its bell integral is
# zero in double precision: ln of the smallest normal float (-708) less a
# margin of e^40 for a bell prefactor 1/sqrt(2 pi F''), which only a band
# narrower than 1e-17 Hartree could exceed.
FC_UNDERFLOW_EXPONENT = -748.0

# Largest |H_ixjy - H_jyix| a finite-difference Hessian (src/properties/
# hessian.py) may carry before it is refused, relative to its own largest
# element. The two halves of a central difference of an exact force differ by
# the force noise over h plus the O(h^2) truncation, so a large value means
# the step is outside the surface's Taylor radius or the forces are noisy.
# Formaldehyde/cc-pVDZ/B3LYP at NUCLEAR_FD_STEP: 4.0e-07 density-fitted,
# 7.6e-02 on the 148-point interpolation grid.
HESSIAN_FD_ASYMMETRY_TOL = 1e-3

# Relative asymmetry below which a finite-difference Hessian is at its force
# noise floor and its step scaling is not tested: an ISDF force reproduces to
# ~1e-8 Ha/Bohr, which over h = 1e-3 is ~1e-5 of a force constant.
HESSIAN_FD_NOISE_REL = 1e-5

# Range the asymmetry ratio between steps 2h and h must fall in above the
# noise floor: h^2 truncation gives 4 (3.85 on formaldehyde's G3 ISDF grid). A
# surface still rough on the step's scale gives less (2.38 on its 148-point
# grid).
HESSIAN_FD_STEP_RATIO = (2.5, 6.0)

# Lowest level of ISDF_GRID_ACCURACY a finite-difference Hessian of an ISDF-K
# mean field is built on. The ISDF energy depends on the orientation of each
# atom's interpolation cloud (1e-5 to 1e-4 Ha within a few degrees on
# formaldehyde), and the covariant atomic frames turn the clouds at 1-20 rad
# per Bohr of nuclear motion, so on coarse grids the surface is rough on a
# 1e-3 Bohr scale and its force constants are wrong even in the h -> 0 limit
# although the force is exact. Formaldehyde/cc-pVDZ/B3LYP against density
# fitting, largest frequency error: 1362 cm^-1 at 148 points per atom, 405 at
# G2, 11 at G3 (35 on Hartree-Fock).
ISDF_HESSIAN_MIN_GRID = 'G3'

# Minimum share of sum (X+Y)^2 in a BSE root's winning Gamma_i (x) Gamma_a
# channel for the root's point-group irrep label (src/properties/
# characters.py) to be trusted. Below this the transition amplitude is spread
# over more than one symmetry channel -- by a distorted geometry, or by an SCF
# that converged to a symmetry-broken solution -- and the label is not
# meaningful.
PURITY_FLOOR = 0.99

# Quasiparticle-orbital character (`properties.characters`). The window of
# canonical orbitals Pipek-Mezey localizes, on the orbital's own side of the
# gap and starting at the frontier; a localized orbital whose largest fragment
# population is below the floor is reported as not cleanly assigned; two
# candidates whose fingerprint similarities are closer than the margin make a
# geometry-to-geometry track ambiguous.
CHARACTER_WINDOW = 6
LOCALIZED_ASSIGNMENT_FLOOR = 0.8
TRACKING_MARGIN = 0.1

# Fragment localization (`Base.fragment_localization`): the Pipek-Mezey
# functional tolerance. Tighter than pyscf's 1e-6 default because the local
# orbitals are differentiated -- a finite difference of a diabatic element
# over h = 1e-3 Bohr needs the localization converged well past h^2.
FRAGMENT_PM_CONV_TOL = 1e-12
FRAGMENT_PM_CONV_TOL_GRAD = 1e-9
FRAGMENT_PM_POLISH_MAX = 2000

# Fragment-partitioned BSE (`properties.fragment_bse`). Up to
# FRAGMENT_DENSE_MAX rows every block is diagonalized and every resolvent
# solved densely; above it, by Davidson and conjugate gradients to
# FRAGMENT_SOLVE_TOL. The pole guard asks Omega_0 to sit FRAGMENT_POLE_MARGIN
# Hartree (0.27 eV) below the lowest eigenvalue of A_QQ, so Sigma(Omega) is
# smooth over the diabats' range.
FRAGMENT_DENSE_MAX = 3000
# The same limit for a matrix-free (ISDF) operator: forming a block there
# costs one whole-system action per row, against the few hundred an
# iterative solve takes.
FRAGMENT_MATRIX_FREE_DENSE_MAX = 256
FRAGMENT_SOLVE_TOL = 1e-10
FRAGMENT_POLE_MARGIN = 0.01

# Finite-difference diabats (`properties.diabatic`): a displaced diabat that
# overlaps its reference by less than this has mixed within its block, and its
# difference would follow a different state.
DIABAT_OVERLAP_FLOOR = 0.9

# Analytic diabatic gradients (`gradients.fragment_diabatic`): a diabat whose
# block holds another state closer than this (Hartree) has no well-defined
# eigenvector response, which divides by the gap.
DIABAT_GAP_MIN = 1e-4
# Roots carried beyond those asked for, and the smallest subspace before a
# restart, in the full-BSE block Davidson of the fragment partition: with
# only the quasiparticle diagonal as preconditioner, a root just above the
# last one asked for otherwise leaves the subspace at each restart.
FRAGMENT_DAVIDSON_EXTRA_ROOTS = 4
FRAGMENT_DAVIDSON_MIN_SPACE = 80
# Newton step (Hartree) at which the pole guard stops locating the lowest
# positive eigenvalue on Q. It sharpens the reported value only, not the
# guard's verdict.
FRAGMENT_GUARD_NEWTON_TOL = 1e-6
# Seed of the two generic AO vectors a, b that fix each diabat's sign: the
# sign of a^T T b, T its AO transition density (`fragment_bse.diabat_phase`).
FRAGMENT_PHASE_SEED = 20261006

# How many orbitals past the frontier are solved to find the lowest-energy
# attachment or removal: G0W0 reorders states relative to the mean field, so
# the quasiparticle LUMO is not always the first virtual, but it is never far.
QP_ORDER_SEARCH = 4

# Eq. (18) arrays kept per mean field (`reaction_field`): callers alternate
# between the optical response and its static partner, per spin layout.
REACTION_FIELD_CACHE_SIZE = 4

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
# chains' because a coupling is a contraction over the whole vector rather
# than a single eigenvalue, and averages its residual down.
SOC_MANIFOLD_CONV_TOL = 1e-5

# Cartesian geometry optimizer, Hartree/Bohr and Bohr; all four must hold.
# `opt_grad_max` is the largest force component at the converged geometry, the
# residual the convergence test is applied to. The name is not `grad_max`
# because that word also means the driving force at a fixed input geometry, a
# different number about a different geometry.
GEOM_OPT_CONV = {'opt_grad_max': 4.5e-4, 'grad_rms': 3.0e-4,
                 'step_max': 1.8e-3, 'step_rms': 1.2e-3}
# The deprecated spelling of the same threshold. `optimize` accepts either
# spelling and refuses the two disagreeing.
GEOM_OPT_CONV['grad_max'] = GEOM_OPT_CONV['opt_grad_max']

# Bohr. geomeTRIC carries a geometry in Angstrom through its own Bohr radius
# (CODATA 2018, 3.2e-11 relative from BOHR_TO_ANGSTROM), so the start it
# hands back is the given one displaced by ~1e-10 Bohr. A requested geometry
# within this of the walk's start is the start, evaluated at its own bits,
# which is what lets a surface's first point (`FirstPoint`) answer it.
GEOMETRIC_START_TOL = 1e-8

# The body frame (`body_frame.BodyFrame`): a frozen convention that carries a
# direction turns with the Kabsch rotation of the geometry onto the one it was
# frozen at. That rotation is undetermined where the atoms lie on a line: a
# reference whose second principal spread is below this fraction of the first
# keeps its conventions in the lab frame, and a geometry whose superposition
# Sylvester denominators fall below it has no rotation derivative.
BODY_FRAME_MIN_SPREAD = 1e-6

# Conformer search over the soft torsions of a twisted emitter.
# A bond is drawn when the internuclear distance is within this factor of the
# sum of the two covalent radii.
CONFORMER_BOND_SCALE = 1.25
# ... and it counts as single, hence torsionally soft, only above this fraction
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
# Two relaxed structures are one conformer when they agree in energy to this
# (Hartree) and in superposed heavy-atom RMSD to this (Angstrom). 1e-4 Ha is
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
# no row at it is refused there rather than dropped to a coarser default,
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
# what the wrap removes. It grows with the threads there are to contend over,
# so on two or three threads there is nothing to win and the entry cost of
# the wrap is all that is left -- which is what this gates.
BLAS_WRAP_MIN_THREADS = 4


# Fraction of the job allocation's memory that `Base.utils.memory.
# allocation_max_memory_mb` hands to pyscf's own `max_memory`. pyscf's buffers
# (the DF tensor, the numint grid) are not the whole process: mo_coeff, the
# ISDF factors and python's own overhead can run close to as large as the
# capped buffers do. Handing pyscf the whole allocation would leave that other
# half nowhere to go and the job would be killed for memory it never asked
# pyscf for; 0.6 leaves it 40% of the allocation.
ALLOCATION_MEMORY_FRACTION = 0.6


# Pseudo-inverse cutoff of the minimax transform fits
# (`Base.utils.time_frequency.minimax_transform_weights`), relative to the
# largest singular value of that row's design matrix. The fit is a per-point
# least squares solved through an SVD, and below `_REGULARIZATION_ABOVE` points
# it carries no Tikhonov term, so the filter is 1/S: an exactly zero singular
# value gives 0/0 and one below 1.5e-162 squares to zero and gives an
# infinity, either filling a row of the transform with non-numbers that
# propagate into W(i.tau) and the self-energy. Both occur: a minimax tau point
# large enough that exp(-x tau) underflows over the whole node range leaves an
# exactly zero column in the design matrix. The cutoff separates
# arithmetically zero from small: the smallest relative singular value any
# grid here reaches is 2e-17, and GreenX inverts those deliberately
# (conditioning is the regularization's business), so it drops only what the
# arithmetic cannot invert.
TRANSFORM_FIT_RCOND = 1e-100


# Lebedev order of the cavity surface: 11 is 50 points per sphere against
# pyscf's own default of 29, which is 302. The continuum's cost is the surface
# potential of the density, n_ao^2 x n_surf, so this is the only knob that
# moves it, and refining it buys almost nothing: formaldehyde's solvation
# energy in toluene moves 0.4 meV between orders 11 and 29, and biphenyl's
# relaxed inter-ring torsion agrees at orders 11 and 17 to a hundredth of a
# degree.
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
# Hartree. The residual below which a Davidson correction is sized for
# real_eig's linear-dependence test rather than handed over at its raw length
# r / (d - omega). Above it the raw length clears pyscf's absolute test and is
# left raw, so a solve at conv_tol >= 1e-5 (a root is only corrected while
# |r| > conv_tol) is pyscf's own unsized iteration.
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
# receives. Sizing the block against the slice rather than against free memory
# keeps the peak near the slice the split exists to fit: at 0.5 the build
# peaks at 1.5x the slice, where a single exchange of the whole column layout
# would peak at 3x.
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
# 64-bit words per block of that digest; a throughput knob only (4096 is the
# fastest block length measured for the uint64 matrix-vector product over the
# blocks, which runs far faster than blake2b over the same bytes).
AGREEMENT_DIGEST_BLOCK = 4096

# The (A - B) probe replicated over ranks (`LinearResponse.davidson`) runs its
# own Lanczos where serially it runs ARPACK: eigsh holds one process-wide lock
# across its whole iteration, matvecs included, so rank threads cannot all be
# inside it at once. Sized by eigsh's own defaults: a basis of
# max(2k + 1, AMB_LANCZOS_NCV) vectors between thick restarts, and at most
# AMB_LANCZOS_MAXITER_PER_DIM operator applications per pair.
AMB_LANCZOS_NCV = 20
AMB_LANCZOS_MAXITER_PER_DIM = 10
# Weight of the normalized cold start 1/d added to the lowest converged Casida
# root's normalized X - Y when the (A - B) Lanczos starts after the Davidson
# (`LinearResponse.davidson`). (A - B) is block diagonal over the pair irreps
# and the root lies in one of them, so the root alone would confine the
# Krylov space there; 1/d has weight in every irrep. At 0.1 a tenth of the
# start reaches the other irreps while the root still carries most of it, so
# the warm start reaches the sign certificate in fewer block actions than 1/d
# alone.
PROBE_START_MIX = 0.1

# How the Hellmann-Feynman adjoint of a BSE root is realized. 'explicit'
# contracts the Casida vectors against the three-index blocks B[P, i, a] of
# `gradients.bse_isdf.bse_cache`, (naux, nocc, nvir) each, ten of them alive at
# once; 'grid' is the reverse of the ISDF block action
# (`LinearResponse.isdf_bse_adjoint`), which closes over the grid and forms no
# three-index block at all. The same derivative to rounding, not the same
# bits, so the vocabulary is spelled once for the chain and the record.
BSE_ADJOINTS = ('explicit', 'grid')

# Grid points per row tile of the grid BSE adjoint
# (`LinearResponse.isdf_bse_adjoint`). Fixed, never derived from a memory
# budget or a rank count: a GEMM's bits depend on its call shape, so one tile
# sequence is what lets the rows be handed to their owners unchanged. Each
# rank holds its own row tiles and streams the other ranks' column tiles,
# so a pass keeps a handful of (256, M) and (256, naux) tiles alive.
BSE_ADJOINT_TILE_ROWS = 256

# GB of float64: the largest particle-hole block C_ov = B[:, occ, virt],
# naux * nocc * nvir * 8 bytes, that the explicit residue backend of the
# quasiparticle solves may build. The backend holds C_ov, the adjoint Cov_bar
# of the state in its reverse pass and one more block of that size inside each
# residue evaluation or push, three at once, whole on every rank: at this limit
# 768 GB a rank. Above it the route cannot run at any rank count, and the
# answer is the Laplace backend (residues below the particle-hole gap, from
# proj(tau)) or the pole model, which build no block.
EXPLICIT_RESIDUE_MAX_GB = 256.0

# Auxiliary rows per slab of W_bar in the distributed grid BSE adjoint
# (`LinearResponse.isdf_bse_adjoint`): slab s is summed by rank s % nranks
# over the grid tiles in tile order, and the slabs are then gathered verbatim.
# Fixed, like BSE_ADJOINT_TILE_ROWS and for the same reason: a slab is a
# GEMM's row count, so it must not follow the rank count. The grid index is
# cut in BSE_ADJOINT_TILE_ROWS tiles on both sides of every M^2 product, so
# one pass holds (2 nT + 4) (tile, tile) blocks, 6.3 MB, not (rows, M) tiles.
BSE_ADJOINT_AUX_ROWS = 256

# The largest count one MPI call may carry, in elements of the call's
# datatype. mpi4py hands the library an MPI_Count only where the library has
# the MPI-4 large-count routines (MPI_Allreduce_c, ...), which Open MPI 5.0.x
# does not provide; its fallback narrows the count to a C int and raises
# MPI_ERR_ARG past 2^31 - 1, a count proj(tau) and D pass at large M, so every
# collective of `Base.utils.mpi_grid` moves at most this many float64 per
# call, in windows; below it a call is one window, the unchunked call itself.
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
# count, and no rank holds V whole: naux^2 doubles, where a slab is 512 naux.
ISDF_SCF_METRIC_SLAB = 512

# Bytes of one thread's temporary in the element-wise work of the ISDF fit's
# three-centre pass (`Base.separable_ri`): a shell block's test co-densities
# are screened and built a slab of grid rows at a time, and its kept
# (mu nu|P) rows gathered a slab of auxiliary functions at a time, so no
# block is ever whole and each slab is made, scaled and reduced while it is
# still in cache. A cost knob only: every element is the same product whatever
# the slab.
FIT_ROW_CHUNK_BYTES = 4 << 20

# Edge, in grid points, of the square tiles in which the replicated ISDF fit
# (`Base.separable_ri`) writes the transpose of its Gram matrix's lower block
# triangle into the upper one while balancing it: a tile's transposed read
# touches one cache line and one page per source row, 256 of each. A cost
# knob only: every element is the same two products whatever the tile.
FIT_TRANSPOSE_TILE = 256

# The trial space the Casida/BSE Davidson (`LinearResponse.davidson`) holds
# before pyscf's `real_eig` collapses it. real_eig sizes that space from its
# process-wide MAX_MEMORY (4000 MB), so the trial pairs it holds fall as the
# pair space grows, and a collapse keeps only the nroots Ritz vectors and
# applies the action to them again: a solve that collapses takes more cycles
# and block actions than one that never does, its top roots converging last.
# The space is raised to DAVIDSON_SPACE_CYCLES of real_eig's per-cycle
# increment, never past what this rank's memory holds of its four
# pair-space-long holders per trial pair and never below real_eig's own bound,
# so a solve that fits within real_eig's own space runs pyscf's iteration.
# Where no allocation says what this rank's memory is (a single machine
# outside SLURM, whose max_memory is pyscf's process default), the whole
# holders are held within DAVIDSON_SPACE_GB instead.
DAVIDSON_SPACE_CYCLES = 50
DAVIDSON_SPACE_GB = 16
# Cycle cap of the gradient chain's Davidson: three trial spaces of
# DAVIDSON_SPACE_CYCLES increments, so a solve whose space collapses twice
# still reaches its tolerance. It differs from the solver's default cap of 100
# cycles only for a solve that needs more.
BSE_FORCE_MAX_CYCLE = 3 * DAVIDSON_SPACE_CYCLES

# Fraction of this rank's max_memory (the allocation's share a job hands pyscf,
# `Base.utils.memory.allocation_max_memory_mb`) that the Davidson's trial space
# (`LinearResponse.davidson`) may take together with the block action's own
# working set (its kernel rows and tiles and one batch's buffers), counted
# on the pair rows this rank holds. One half is real_eig's own rule, which
# gives its holders half of max_memory; the action's arrays come off that half
# rather than out of the other, which the factors, W, the mean field and the
# setup's gathered D hold. On a large system the half left beside the action
# holds more trial pairs than DAVIDSON_SPACE_CYCLES asks for, so the cycle
# bound binds and the iteration path does not depend on the rank count.
DAVIDSON_SPACE_FRACTION = 0.5

# Pair rows per tile of the Casida/BSE Davidson's trial space distributed over
# ranks (`LinearResponse.trial_space`): a rank holds the rows of its
# contiguous run of tiles of V, W, U1 and U2. Fixed, never derived from a rank
# count: a GEMM's bits depend on its call shape, so one tile sequence is what
# makes the row products the same bits at every rank count. A cost knob
# otherwise: a tile of this height costs about what the untiled rows do, a
# smaller one more per row, and the tile count bounds how unevenly the ranks'
# runs of tiles can split the pair space.
DAVIDSON_PAIR_TILE = 1024

# The diagonal the Casida/BSE Davidson (`LinearResponse.davidson`) divides its
# residuals by: 'bare', the pair energies d = eps_a - eps_i, or 'screened',
# d - (ii|W|aa), the diagonal of the screened direct term in the fit's own
# gauge (the bare Coulomb's for TDHF, nothing for RPA). The singlet's Hartree
# diagonal 2(ia|ia) is left out: added, it converged slower. The screened
# diagonal is the default: it is closer to the diagonal of A where the
# screened attraction binds the pairs, so a BSE solve takes fewer cycles and
# block actions to the same roots, and holds whole a trial space the bare one
# collapses. RPA's diagonal is d either way; 'bare' stays selectable, and it is
# what a solve that must reproduce a bare-preconditioned iteration asks for.
DAVIDSON_PRECONDITIONER = 'screened'
DAVIDSON_PRECONDITIONERS = ('bare', 'screened')

# Grid rows per tile of the screened preconditioner's fitted densities
# D^T (X_o o X_o) and D^T (X_v o X_v) (`LinearResponse.davidson`): tile t is
# rows [4096 t, 4096 (t + 1)) of the grid, formed whole by the rank holding
# its first row, and every tile's (naux, n_occ + n_vir) addend is added in
# tile order (`ordered_chain_sum`). Fixed by the grid, never by the rank
# count; a tile's squares are 4096 (n_occ + n_vir) doubles beside the
# (naux, n_occ + n_vir) sum.
DAVIDSON_DIAGONAL_TILE = 4096

# Bohr within which an explicit set of ISDF shell radii counts as the shipped
# table row for the same (element, basis, auxbasis, counts). Both sides are
# float64 (one read back from the JSON table, one held by the caller), so
# agreement is either exact to the last bit or the two are different grids:
# neighbouring rows differ in the second decimal, and a run-time
# re-optimization onto another local minimum lands 1e-3 Bohr or further away.
# This tolerance absorbs decimal round-tripping and nothing else; it is not a
# statement that two grids this close are interchangeable.
ISDF_RADII_MATCH_TOL = 1e-10

# Bytes of W(i.omega) - I per chunk of the omega -> tau transform of the
# space-time self-energy (`GW.imaginary_time`): the frequencies are folded into
# Wt(i.tau) a chunk at a time, naux^2 doubles a frequency. The chunk is also
# the association of that sum (one GEMM per chunk, added in chunk order),
# which the row-distributed transform keeps, so it is fixed by naux alone and
# never by the rank count.
SCREENED_CHUNK_BYTES = 2 << 30
