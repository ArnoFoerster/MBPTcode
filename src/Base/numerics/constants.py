"""Numeric defaults shared by every layer: physical constants, tolerances, grid
sizes and memory knobs, each with the reason for its value."""
import math

# Physical constants. Defined here and nowhere else: import them, never re-spell
# the digits, so every route reports the same number. Each line says the CODATA
# adjustment its value is from. pyscf.data.nist holds older values for most of
# them (HARTREE2EV = 27.21138602); BOHR_TO_ANGSTROM is CODATA 2010, equal to
# nist.BOHR bitwise, so a geometry passes between pyscf and this code unchanged
# (GEOMETRIC_START_TOL relies on it).
HARTREE_TO_EV = 27.211386245988          # CODATA 2018
HARTREE_TO_KCAL = 627.509474             # CODATA 2018
BOHR_TO_ANGSTROM = 0.52917721092         # CODATA 2010 = pyscf nist.BOHR

# Bytes in one GB and MB in one GB (decimal, as pyscf's max_memory MB is 1e6
# bytes), for every memory estimate.
BYTES_PER_GB = 1e9
MB_PER_GB = 1000.0

# Auxiliary functions per chunk of a pyscf DF loop turned into MO factors
# (`pyscf_interface.mo_factor_block`): bounds the chunk's (nb, nao, nao) copy.
DF_MO_BLOCK = 200

# Imaginary frequencies of the O(N^4) imaginary-axis GW route
# (`GW.imaginary_axis`): its W(i omega) quadrature, minimax by default.
IMAGFREQ_NFREQ = 20

# Lorentzian broadening for self-energy denominators / spectral functions.
DEFAULT_BROADENING_ETA = 1e-3

# Relative eigenvalue floor of the Coulomb metric's square root, the gauge of
# the separable factors (D = M^T V^1/2): a direction below it is the metric's
# numerical null space and is dropped from the root rather than carried at
# the rounding level. It can sit this low because the root is never
# inverted: a small eigenvalue enters it as w^1/2, which is harmless.
AUX_METRIC_ROOT_FLOOR = 1e-12

# How negative, relative to the largest eigenvalue, a dressed metric v + vtilde
# may be before its root is refused. v + vtilde is a positive kernel, so a
# negative eigenvalue beyond rounding means the discretized reaction field
# over-screens the bare interaction: an error of the cavity or of eps, not a
# truncation to be dropped.
AUX_METRIC_INDEFINITE_TOL = 1e-10

# The bytes of auxiliary-metric eigensystems and roots `separable_ri` keeps,
# keyed by the metric's content (`metric_eigensystem`): a chain asks for the
# same (V + vtilde)^1/2 several times per geometry -- each factor, the
# reaction field's congruence, the D branch's root and its adjoint -- and the
# eigendecomposition is the cost. A metric too large for the budget is not
# kept; the row fit's `metric_root` never is.
METRIC_ROOT_CACHE_BYTES = 512 * 2**20

# The largest OPENBLAS_NUM_THREADS a stock OpenBLAS build accepts. Its thread
# pool is sized at compile time (NUM_THREADS); asking for more can segfault
# mid-SCF. Every launcher that sets the thread environment clamps OpenBLAS to
# this while leaving OMP free to use the whole node.
OPENBLAS_MAX_THREADS = 128

# Validated ISDF interpolation grids: {basis: {level: (A1, A2, A3, B1)}}, the
# Lebedev sub-shell replica counts measured to reach an accuracy; the only
# place a validated count is written down.
#
# A level is an accuracy target, not a size: the largest deviation of the lowest
# BSE roots from `solve_bse_df` at the same mean field, over sixteen molecules
# spanning H B C N O Si P S (fifteen at cc-pVQZ), singlets and triplets, with
# the roots matched between the two routes. G1 < 8 meV, G2 < 4 meV, G3 < 2 meV,
# at 6*A1 + 8*A2 + 12*A3 + 24*B1 points per atom. Each level is the cheapest
# rung of the ladder 8, 16, ... 72 (A1 = rung) that meets its target on both
# spins.
#
# The ladder is not monotone, so an entry licenses its own count and no other.
# A missing level is a refusal, not a fallback: nobody measured it, so a
# neighbouring level, a larger count and another basis are all equally
# unlicensed. aug-cc-pVQZ has no entry; cc-pVTZ G3 is the best rung although
# it measures 2.02 meV (glyoxal triplet) against 2 meV. Worst root error at the
# entry, meV: cc-pVDZ 16 6.79 / 24 1.94; aug-cc-pVDZ 16 6.66 / 24 1.86; cc-pVTZ
# 16 4.40 / 32 3.18 / 40 2.02; aug-cc-pVTZ 24 3.88 / 32 1.43; cc-pVQZ 32 3.81 /
# 64 1.41.
#
# 'GH' is not an accuracy target: it is the grid a finite-difference Hessian of
# an ISDF-K mean field is built on (`ISDF_HESSIAN_MIN_GRID`), set by the
# smoothness of the energy surface and not by the BSE roots: cc-pVDZ
# formaldehyde frequencies against density fitting are off by up to 93 cm^-1
# (HF) and 24 cm^-1 (B3LYP) at (24, 15, 9, 3), by 35 and 11 at (32, 20, 12, 4).
# The 'GH' radii are the ones the Hessian was validated on; do not re-optimize
# them for the BSE. A basis without a 'GH' entry has no validated Hessian grid.
# The target each level is validated to, in eV: G1 < 8, G2 < 4, G3 < 2 meV.
ISDF_GRID_ACCURACY_TARGET_EV = {'G1': 8e-3, 'G2': 4e-3, 'G3': 2e-3}

# The ISDF-vs-DF acceptance check of a geometry (`photophysics.df_check`).
# Its Davidson residual, Ha: the check decides a pass or a fail at the meV
# scale of the targets above, and two kernels are solved per spin per stage.
DF_CHECK_BSE_CONV_TOL = 1e-6
# The largest |eps_DF - eps_ISDF| on the explicit set, eV, that still proves
# the two mean fields order their orbitals alike (same functional; the ISDF-K
# and the J/K fit differ by a few meV).
DF_CHECK_ORBITAL_MATCH_TOL_EV = 5e-2
# Roots solved above the compared one, so the Davidson's last root is not it.
DF_CHECK_ROOTS_ABOVE_FOLLOWED = 3

# Orbitals listed on each side of the gap in a vertical record's
# quasiparticle spectrum, and the largest quasiparticle shifts (or the worst
# unrepresented orbital densities) a vertical record reports.
VERTICAL_FRONTIER_LISTED = 3
VERTICAL_WORST_LISTED = 8

# The production TADF run's command-line defaults (`photophysics.cli`), each
# named in --help. The functional and the factors' grid level of the chain;
# the BSE roots solved (above every reported one) and the roots reported per
# spin; the frontier half-width of a declared frontier set; the shape of the
# frozen quasiparticle shift beyond the explicit set.
TADF_REFERENCE = 'lrc-wpbeh'
TADF_GRID_ACCURACY = 'G1'
TADF_NROOTS = 6
TADF_N_STATES = 3
TADF_QP_HALF_WIDTH = 2
TADF_OUTSIDE_SCISSOR = 'curve'
# Threads per rank, and the memory (MB) of a launch whose scheduler names no
# allocation.
TADF_THREADS = 2
TADF_ALLOCATION_MB = 100000
# The UISC rate model's defaults (`photophysics.declaration.UiscModel`): the
# promoting modes differenced, the dimensionless step each way, the lowest
# singlets and triplets of the coupling block, and the reorganization
# energies (eV) Samanta et al., J. Am. Chem. Soc. 139, 4042 (2017), Sec. 2
# scan with S = 0.
UISC_N_MODES = 12
UISC_DQ = 0.1
UISC_BLOCKS = (2, 2)
UISC_LAMBDA_SAMANTA_EV = (0.10, 0.20)
ISDF_GRID_ACCURACY = {
    'cc-pvdz':     {'G1': (16, 10, 6, 2), 'G2': (24, 15, 9, 3),
                    'G3': (24, 15, 9, 3), 'GH': (32, 20, 12, 4)},
    'cc-pvtz':     {'G1': (16, 10, 6, 2), 'G2': (32, 20, 12, 4),
                    'G3': (40, 25, 15, 5), 'GH': (40, 25, 15, 5)},
    'cc-pvqz':     {'G1': (32, 20, 12, 4), 'G2': (32, 20, 12, 4),
                    'G3': (64, 40, 24, 8)},
    'aug-cc-pvdz': {'G1': (16, 10, 6, 2), 'G2': (24, 15, 9, 3),
                    'G3': (24, 15, 9, 3)},
    'aug-cc-pvtz': {'G1': (24, 15, 9, 3), 'G2': (24, 15, 9, 3),
                    'G3': (32, 20, 12, 4), 'GH': (40, 25, 15, 5)},
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

# The active-space DMRG (block2): bond dimension, sweep count, the noise added
# in the sweeps and the tolerance of its Davidson.
DMRG_BOND_DIM = 500
DMRG_N_SWEEPS = 20
DMRG_NOISE = 1e-4
DMRG_DAVIDSON_TOL = 1e-12

# |element| below which an active-space vertex counts as absent, or a spin or
# Hermiticity residual of the effective Hamiltonian as round-off.
ACTIVE_SPACE_VERTEX_TOL = 1e-10

# The direct DF exchange diagonal (`qp_solve._df_direct_exchange_diagonal`):
# natural occupations of the density below DF_EXCHANGE_OCCUPATION_FLOOR times
# the largest, and Coulomb-metric eigenvalues below DF_EXCHANGE_METRIC_FLOOR
# times the largest, are dropped.
DF_EXCHANGE_OCCUPATION_FLOOR = 1e-10
DF_EXCHANGE_METRIC_FLOOR = 1e-12

# The (A - B) probe (`LinearResponse.probe`): its residual tolerance, and the
# looser tolerances a sign-only probe tries first, stopping at the first that
# proves the sign (theta -+ |r| clear of zero).
AMB_PROBE_TOL = 1e-6
AMB_PROBE_SIGN_TOLS = (1e-2, 1e-3, 1e-4)

# CPHF/CPKS Z-vector solve (GWDensityMatrixSolver.solve_relaxation).
CPHF_MAX_CYCLE = 100
CPHF_TOL = 1e-9
# src.Base.pyscf.cphf.solve: rounds of solving again for the true residual
CPHF_REFINEMENTS = 10
# ... and pyscf's lib.krylov inside one round, on a unit right-hand side: its
# own tolerance and the dependence threshold that ends it (the round stops on
# the dependence test, so both sit below any true-residual tolerance)
CPHF_KRYLOV_TOL = 1e-14
CPHF_KRYLOV_LINDEP = 1e-20
# Z-vector of a correlated force or relaxed excited-state density
# (df_correlation_force, ee_relaxed_density), tighter than CPHF_TOL: the
# force is first order in the multipliers' error
CORRELATION_CPHF_TOL = 1e-10

# QP root-finding (src/Solvers/qp_equation.py).
QP_NEWTON_TOL = 1e-6
QP_NEWTON_MAX_ITER = 50
QP_GRAPHICAL_TOL = 1e-8
QP_GRAPHICAL_N_OMEGA = 150
QP_GRAPHICAL_MAX_BISECTION = 100
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
# Floor on the measured reassociation response the distance is divided by: a
# grid on which reordering a sum happens to move a number by an ulp would
# otherwise turn a 1e-12 realization difference into a ratio of thousands.
# The row fit and the replicated fit differ by 1.3e-12 relative on the BSE root
# there; a defect moves the fit by whole digits.
FIT_REASSOCIATION_FLOOR = 2e-13

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
# gradient (src/environments/pcm_derivatives.py). The solvation energy is quadratic in
# the density it is built from, so the central difference this scales carries
# no truncation error and the value is a conditioning choice only (measured
# step-independent to 3e-15 from 1e-1 down to 1e-3).
PCM_CROSS_TERM_STEP = 1e-2

# Contour deformation of the GW self-energy
# (src/SingleReference/GW/contour_deformation.py). A pole of G at
# |omega - eps_q| below RESIDUE_ON_CONTOUR_TOL counts as on the contour.
# QP_POLE_OFFSET is how far toward the chemical potential from its orbital
# energy a state's start point lies, where Eq. (27) of the pole model and the
# residue backend's frequency margin are read before any root is known
# (`qp_equation.qp_start`). The integral term takes its singular part in
# closed form (`cd_integral_weights`), so Sigma is continuous through every
# orbital energy and the root search needs no distance from them.
RESIDUE_ON_CONTOUR_TOL = 1e-10
QP_POLE_OFFSET = 1e-3
# Margin (Hartree) past the residue frequencies of the Newton start that the
# tau grid must carry for the Laplace residue backend to be chosen: the root
# moves by the quasiparticle correction, a few tenths of an eV to 2 eV.
RESIDUE_FREQ_MARGIN = 0.1
# Gauss-Legendre points on the imaginary-frequency integral of the dRPA
# correlation energy in the space-time route; converged to 1e-8 Ha at 40.
RPA_ENERGY_NFREQ = 40
# Gauss-Legendre points on the imaginary-frequency half of a contour
# deformation, for a quasiparticle set with a state on the quadrature routes.
# The grid is never resized: the Lorentzian a pole of G puts at nu = 0 is
# integrated in closed form.
CD_NFREQ = 64
# Contour-deformation points for a quasiparticle set the pole model carries
# whole (every state on the sum-over-poles route or a frozen scissor): that
# Sigma is analytic in omega, so the grid only feeds the fit. Against 512
# points (formaldehyde, acrolein and DMABN, gas and toluene, at
# LRC-wPBEh/cc-pVDZ; ethylene HF with 24 poles; admitted window): from 32
# points up every S1 and T1 is within 0.034 meV and the frontier
# quasiparticle within 0.009 meV; at 24, acrolein T1 is 0.12 meV off. The
# deepest explicit state (the occupied scissor probe) converges slower, 44 meV
# off at 32 and 15 meV at 64.
CD_NFREQ_SOP = 32
# Imaginary-time points behind the contour-deformation grid. The cosine
# transform onto the imaginary-frequency quadrature would be converged at 18;
# a residue asks the same grid for the cosh transform at a real frequency w',
# which reaches down to gap - w' and needs the wider range these points buy.
CD_NTAU = 24

# Root tolerance (Ha) and step cap of the bracket-safeguarded refinement of a
# bracketed quasiparticle root with a closed-form slope, on every route of the
# one root finder (`Solvers.qp_equation.find_qp_root`). The root is
# converged far tighter than an energy needs because the gradient chain
# differentiates the equation at that root: a residual of 1e-6 leaves Z and
# every adjoint that multiplies it off by the same relative amount.
QP_ROOT_TOL = 1e-11
QP_ROOT_MAX_ITER = 100

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
# Relative distance below which the vector fit's poles are merged into one
# (`sum_over_poles.merge_poles`), before every relocation pass and after the
# last. Zeros of the fit's weight that fall below the particle-hole gap are
# clipped onto it, and a complex pair keeps its real part, so poles pile onto
# one position and the pole basis F gets duplicate columns (cond ~1e16 on
# water/cc-pVDZ at CD_NFREQ_SOP points): the least squares is rank-deficient,
# the next relocation is decided by rounding, and a one-ulp change of W_c can
# move a pole by 43%. Merged, F has cond <= 1.4e6 and the same change moves a
# pole by at most 1.3e-8 (100 draws on each of four states).
# Two poles a relative delta apart have kernel columns a relative delta apart,
# so the data cannot tell them apart below ~delta; one pole at their mean
# u = Om^2 changes the model by O(delta^2) of their amplitude, 1e-6 at 1e-3,
# below the 32-point model's own error. At 32 points S1 and T1 sit within
# 0.006 meV of 512 (formaldehyde, acrolein; LRC-wPBEh/cc-pVDZ, admitted
# window); without the merge formaldehyde is 3.1 and 16.9 meV off, a branch of
# the fit. The merged set keeps fewer poles (8-12 of 12) and is not refilled
# to M: thioformaldehyde's HOMO-1 quasiparticle sits 0.038 meV from the
# contour at 64 points.
SOP_POLE_MERGE_TOL = 1e-3

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

# Root tolerance (Ha) of `GW.quasi_boson.QPqb.quasiparticle_root`, the dense
# route's largest-weight root of w = eps_p + Sigma_pp(w): the Newton step it
# stops at, and the budget of couplings it may drop before bracketing. By
# Weyl's inequality, removing couplings of total norm sqrt(sum W^2) from the
# arrowhead supermatrix moves every root by at most that norm, so dropping the
# weakest poles while the norm stays below this moves the answer by no more
# than the solve itself resolves. What goes is the symmetry-forbidden poles,
# |W| ~ 1e-17, whose roots sit closer to their pole than w can resolve. The
# same 1e-12 as `QPqb.solve_diag`'s Newton, so both rules converge alike.
QB_QP_ROOT_TOL = 1e-12

# Bytes of one (intervals x poles) block of `quasi_boson.end_value_bounds`,
# the pruning pass of the largest-weight root search: an orbital can hold
# hundreds of thousands of poles, so a block is a few rows of them.
QB_QP_END_BLOCK_BYTES = 64 << 20

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

# BSE Casida solver: dense below this occupied-virtual pair count, the
# matrix-free ISDF/DF Davidson above (12000 pairs is ~1.2 GB per block).
# `solve_bse`'s solver='auto' compares BSE_DENSE_MAX_GB against
# 2 * n_ov**2 * 8 bytes, the dense route's own (A, B) storage; TDA is not
# exempt, since the dense route builds B whether or not `tda` is set and only
# the eigensolver drops it. BSE_DENSE_MAX_GB is the memory form
# of BSE_DENSE_MAX_NOV so the two spellings of the boundary cannot drift apart.
BSE_DENSE_MAX_NOV = 12000
BSE_DENSE_MAX_GB = 2 * BSE_DENSE_MAX_NOV**2 * 8 / BYTES_PER_GB

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
# Ha/Bohr. A force against the finite difference of its own energy on
# formaldehyde/cc-pVDZ at the default and G1 counts, on the BSE-selected radii
# of the shipped table: measured 1.2e-8 to 2.9e-8, where the radii of the table
# before gave below ISDF_GRADIENT_FLOOR. Read only by the tests that measure it
# (test_one_fit_forces_follow_their_energy, test_mean_field_force_tiles); every
# other gate stays at the floor.
ISDF_ROW_FORCE_ENERGY_TOL = 4e-8

# Nuclear finite-difference step (Bohr) for the gradient/derivative-coupling
# layer: the finite-difference gradient of a surface and the derivative
# couplings by eigenvector overlap.
NUCLEAR_FD_STEP = 1e-3

# Convergence of the orbital-response (Z-vector) equation
# antisym(Y_E + Y[fold(Lambda)]) = 0 solved in src/gradients/multipliers.py:
# lgmres runs to an absolute residual of this times max(1, |rhs|).
ORBITAL_MULTIPLIER_TOL = 1e-11

# Iteration ceiling of that lgmres solve (`ScfNumerics.multiplier_max_iter`).
# Reaching it is a numerical event: the last iterate is returned, warned about
# and flagged 'orbital_response_not_converged'.
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
# A Casida eigenvector's sign is fixed by its largest |X| element, positive;
# elements within this relative distance of the largest are a tie, broken by
# the lowest index, so that a symmetry-equal pair whose magnitudes differ only
# by rounding does not decide the sign by its last bits.
CASIDA_PHASE_TIE_TOL = 1e-8

# Coulomb-metric fit error at which an ISDF atomic grid has stopped being a
# coarse grid and become a failed fit; see `separable_ri.optimize_atomic_radii`.
ISDF_FIT_ERROR_FAILED = 1.0

# Working-set cap for one block of the three-centre integral (mu nu|P) when the
# ISDF fit gathers its test-set pairs; memory only, the integrals are unchanged.
THREE_CENTER_BLOCK_BYTES = 2 << 30

# Auxiliary functions per tile of the fitted Fock skeleton
# (`gradients.skeleton_tiles`): tile t of consecutive auxiliary shells is rank
# t % size's, and its (mu nu|P) and derivative integrals are made, contracted
# and dropped in slabs of the first AO index under THREE_CENTER_BLOCK_BYTES.
# Fixed by the basis, never by the rank count, so each tile's contribution is
# the same bits whoever computes it. A tile of the exchange's
# L = (mu nu|P) C_occ holds nao * nocc * SKELETON_AUX_TILE doubles, and the
# auxiliary set splits into naux / SKELETON_AUX_TILE tiles, several a rank on
# a large system.
SKELETON_AUX_TILE = 32

# Points per tile of the xc skeleton's Becke-grid sum (`gradients.skeleton_tiles`,
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
# which assumes the occupied-virtual Fock block vanishes, and for a state
# model's relaxed density and Hellmann-Feynman force (STATE_MODEL_SCF_GRAD_TOL
# is this ceiling under its callers' name). Symmetry hides a violation: a
# symmetric molecule looks converged and is wrong in the fourth digit of the
# force.
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
# with the energy-only conv_tol (SCF_ENERGY_CONV_TOL), then exact Newton steps on the orbital
# Hessian take it to the mean field's own conv_tol_grad. Each step solves
# H x = -g by preconditioned conjugate gradients to a residual of
# SCF_NEWTON_RESIDUAL_FRACTION x conv_tol_grad (the next gradient is that
# residual plus O(|x|^2), so one step from 1e-7 reaches a 1e-11 target),
# with at most SCF_NEWTON_MAX_HOPS Hessian products (one response Fock build
# each) per step and SCF_NEWTON_MAX_STEPS steps.
SCF_NEWTON_FINISH_FROM = SCF_ENERGY_GRAD_TOL
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
# The energy test of a reference mean field held to the last bits a total
# energy has (one ulp is 2.2e-16 |E|): what the dense reference surfaces and
# the gates that compare against them converge, with SCF_DIFFERENTIABLE_GRAD_TOL
# on the orbital gradient.
SCF_TIGHT_CONV_TOL = 1e-14

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

# The adaptive explicit set (`GW.qp_selection`): an analytic-continuation GW at
# the reference geometry selects which admitted orbitals are solved, and the
# rest of the window borrows a solved orbital's shift. Its first-order error
# in each target excitation, B = sum_p |n_p| (|delta_p| + u_p), is held below
# this many meV per target state, well under the ISDF grid's own error.
ADAPTIVE_QP_TOL_MEV = 1.0
# The fraction of that budget the selection aims for before the explicit roots
# exist, so that the check on the production eigenvectors rarely has to add a
# state (each addition costs a second Casida at the reference geometry).
ADAPTIVE_SELECT_MARGIN = 0.5
# Largest residual AC minus explicit shift on the frontier pair, eV, above
# which the continuation is unfit to select and the whole window is solved.
ADAPTIVE_AC_CALIBRATION_MAX_EV = 0.1
# Largest |AC shift| the continuation is trusted to judge, eV; beyond it the
# state is solved explicitly. Explicit shifts seen on HF and range-separated
# references stay below 1.6 eV.
ADAPTIVE_AC_SHIFT_MAX_EV = 3.0
# Roots within this many eV of a target, on the selection spectrum, become
# targets too: along a walk they can mix with or swap onto the followed state.
ADAPTIVE_ROOT_MIX_EV = 0.1
# Largest |delta_p| + u_p of a hole, eV, whatever its weight in the targets: a
# hole further from its own root than this is solved. A single excitation has
# sum_p |n_p| ~ 2, so every root, targeted or not, then sits within twice this
# of its whole-window energy, i.e. within ADAPTIVE_ROOT_MIX_EV: a root too far
# from a target at R0 to be one cannot be pushed onto it by the scissors.
ADAPTIVE_HOLE_MAX_EV = 0.5 * ADAPTIVE_ROOT_MIX_EV
# Rounds of additions after the reference-geometry check on the production
# eigenvectors before the whole window is solved instead.
ADAPTIVE_MAX_ROUNDS = 2

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
HARTREE_TO_CM = 219474.6313702          # CODATA 2014
# Electron masses per unified atomic mass unit: a chemist's mass into the
# atomic units a Hessian is expressed in.
AMU_TO_ME = 1822.888486209               # CODATA 2018

# Boltzmann constant in Hartree per Kelvin: k_B T of a rate expression, in the
# same energy unit as every gap and reorganization energy it is compared with.
BOLTZMANN_HARTREE_PER_KELVIN = 3.166811563e-6
# The atomic unit of time in seconds; a rate in inverse atomic time becomes s^-1
# by dividing by it.
ATOMIC_TIME_SECONDS = 2.4188843265857e-17
# The speed of light in atomic units, 1/alpha. It enters spontaneous emission
# as c^-3, so a rate is cubic in this number and quoting it to three digits is
# a 0.1% error in every radiative lifetime.
SPEED_OF_LIGHT_AU = 137.035999177        # CODATA 2022

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

# Level of ISDF_GRID_ACCURACY ('GH', see there) a finite-difference Hessian of
# an ISDF-K mean field is built on. The ISDF energy depends on the orientation of each
# atom's interpolation cloud (1e-5 to 1e-4 Ha within a few degrees on
# formaldehyde), and the covariant atomic frames turn the clouds at 1-20 rad
# per Bohr of nuclear motion, so on coarse grids the surface is rough on a
# 1e-3 Bohr scale and its force constants are wrong even in the h -> 0 limit
# although the force is exact. Formaldehyde/cc-pVDZ/B3LYP against density
# fitting, largest frequency error: 1362 cm^-1 at 148 points per atom, 405 at
# the (16, 10, 6, 2) grid, 24 at (24, 15, 9, 3), 11 at (32, 20, 12, 4); on
# Hartree-Fock 93 at (24, 15, 9, 3) and 35 at (32, 20, 12, 4).
ISDF_HESSIAN_MIN_GRID = 'GH'

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
# population is below the floor is reported as not cleanly assigned.
CHARACTER_WINDOW = 6
LOCALIZED_ASSIGNMENT_FLOOR = 0.8

# Fragment localization (`fragments.localization`): the Pipek-Mezey
# functional tolerance. Tighter than pyscf's 1e-6 default because the local
# orbitals are differentiated -- a finite difference of a diabatic element
# over h = 1e-3 Bohr needs the localization converged well past h^2.
FRAGMENT_PM_CONV_TOL = 1e-12
FRAGMENT_PM_CONV_TOL_GRAD = 1e-9
FRAGMENT_PM_POLISH_MAX = 2000

# Fragment-partitioned BSE (`gradients.gwbse.fragment_bse`). Up to
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

# Finite-difference diabats (`tests/fragment_diabatic_fd.py`): a displaced diabat that
# overlaps its reference by less than this has mixed within its block, and its
# difference would follow a different state.
DIABAT_OVERLAP_FLOOR = 0.9

# Analytic diabatic gradients (`gradients.gwbse.fragment_diabatic`): a diabat whose
# block holds another state closer than this (Hartree) has no well-defined
# eigenvector response, which divides by the gap.
DIABAT_GAP_MIN = 1e-4
# Roots carried beyond those asked for in the full-BSE block Davidson of the
# fragment partition (`davidson.run_davidson`'s extra_roots): with only the
# quasiparticle diagonal as preconditioner, a root just above the last one
# asked for otherwise leaves the subspace at each restart.
FRAGMENT_DAVIDSON_EXTRA_ROOTS = 4
# Hartree. The smallest magnitude the diagonal preconditioner of the
# partition's resolvent solve may take: no correction is divided by a number
# closer to zero.
FRAGMENT_PRECONDITIONER_FLOOR = 1e-3
# Newton step (Hartree) at which the pole guard stops locating the lowest
# positive eigenvalue on Q. It sharpens the reported value only, not the
# guard's verdict.
FRAGMENT_GUARD_NEWTON_TOL = 1e-6
# Residual norm and cycle budget of the pole guard's Davidson on Q. The guard
# decides a sign against FRAGMENT_POLE_MARGIN, and a Ritz value's error goes
# as |r|^2 over the gap to the next eigenvalue, so 1e-5 decides it with
# orders of magnitude to spare. Q holds every pair but the diabats' and a dense
# cluster of states at its low end, where one root converges slowly.
FRAGMENT_GUARD_TOL = 1e-5
FRAGMENT_GUARD_MAX_CYCLE = 1000
# Seed of the two generic AO vectors a, b that fix each diabat's sign: the
# sign of a^T T b, T its AO transition density (`fragment_bse.diabat_phase`).
# Any integer: it fixes a convention, and |couplings| do not depend on it.
FRAGMENT_PHASE_SEED = 1

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

# The refreeze of both optimizers, an outer loop around the walk. After a walk
# converges at R*, its frozen conventions are rebuilt there -- the ones this
# geometry would choose -- and the drift E_new(R*) - E_old(R*) is measured at
# that one geometry (the rebuilt surface's first evaluation against the old
# surface's last), so it holds the change of the conventions and none of the
# walk's descent. The state is relaxed on the rebuilt surface; a drift above
# REFREEZE_TOL_MEV refreezes again at that walk's minimum, at most
# REFREEZE_MAX_PASSES times, and a drift still above it is warned about and
# flagged (`optimize.RefreezeNotConverged`). A few meV is far below the accuracy of GW-BSE
# excitation energies and gaps.
REFREEZE_TOL_MEV = 3.0
REFREEZE_MAX_PASSES = 4

# The start of every excited-state walk (`optimize.start_nudge`). A surface
# invariant to rigid motion keeps the point group of the geometry a walk
# starts at, so a walk from a symmetric geometry stops at the symmetric
# stationary point even where it is a saddle. The start is moved by a fixed
# random internal displacement (rigid motion projected out), scaled so that
# its largest atomic displacement is START_NUDGE_BOHR: a tenth of the first
# trust radius, worth ~1e-4 Ha on a stiff mode, and above what a walk
# converges to (GEOM_OPT_CONV's step_max 1.8e-3 Bohr). START_NUDGE_SEED makes
# it the same vector at every launch for a given geometry.
START_NUDGE_BOHR = 0.02
START_NUDGE_SEED = 1729

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

# The ISDF interpolation grid `gradients.gwbse.surface_factory.potential_energy_surface`
# asks for when the caller names none: the level of ISDF_GRID_ACCURACY
# validated to 4 meV on the three lowest BSE roots. A basis or an element with
# no row at it is refused there rather than dropped to a coarser default,
# which is a different factorization and not a coarser one.
SURFACE_GRID_ACCURACY = 'G2'

# eV in meV, for a difference already taken in eV.
EV_TO_MEV = 1000.0

# Hartree in meV. Derived from HARTREE_TO_EV rather than spelled again, so the
# refreeze drift a relaxation record reports in meV and the excitation energy
# it reports in eV can never be two different conversions.
HARTREE_TO_MEV = EV_TO_MEV * HARTREE_TO_EV

# The saddle test of a minimum's Hessian (`optimize.escape_saddle`). A walk's
# end is a saddle when the lowest curvature of the internal Cartesian Hessian
# (rigid motion projected out, Ha/Bohr^2) is below SADDLE_CURVATURE_TOL; it is
# then left along that mode by SADDLE_ESCAPE_STEP_BOHR (the first trust radius
# of `optimize`) and walked again. The tolerance is the curvature whose
# energy gain over that step, |k| s^2 / 2, is SADDLE_ENERGY_TOL_MEV: a softer
# negative mode lowers the energy by less than 0.1 meV over the escape step,
# far inside the refreeze tolerance. It sits five times above the noise of a
# central-difference Hessian of the excited-state force at NUCLEAR_FD_STEP
# (asymmetry 1.5e-4 Ha/Bohr^2 at the formaldehyde S1 minimum, LRC-wPBEh/
# cc-pVDZ, ISDF-K mean field converged to SCF_OPTIMIZATION_GRAD_TOL).
SADDLE_ESCAPE_STEP_BOHR = 0.1
SADDLE_ENERGY_TOL_MEV = 0.1
SADDLE_CURVATURE_TOL = (-2.0 * SADDLE_ENERGY_TOL_MEV / HARTREE_TO_MEV
                        / SADDLE_ESCAPE_STEP_BOHR ** 2)

# Seconds `properties.excitations` waits for git to name the commit a record
# was produced on. Provenance is not worth blocking a calculation for: the
# stamp falls back to 'unknown' when the call does not return in time.
GIT_PROVENANCE_TIMEOUT = 10.0


# BLAS threads at or above which a pyscf-OpenMP kernel is worth running with
# the BLAS pool held at one thread (`Base.utils.parallel.threads.blas_single_threaded`).
# The two pools spin against each other, and the cost of that contention is
# what the wrap removes. It grows with the threads there are to contend over,
# so on two or three threads there is nothing to win and the entry cost of
# the wrap is all that is left -- which is what this gates.
BLAS_WRAP_MIN_THREADS = 4


# Fraction of the job allocation's memory that `Base.utils.parallel.memory.
# allocation_max_memory_mb` hands to pyscf's own `max_memory`. pyscf's buffers
# (the DF tensor, the numint grid) are not the whole process: mo_coeff, the
# ISDF factors and python's own overhead can run close to as large as the
# capped buffers do. Handing pyscf the whole allocation would leave that other
# half nowhere to go and the job would be killed for memory it never asked
# pyscf for; 0.6 leaves it 40% of the allocation.
ALLOCATION_MEMORY_FRACTION = 0.6


# Pseudo-inverse cutoff of the minimax transform fits
# (`Base.utils.grids.time_frequency.minimax_transform_weights`), relative to the
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
# The one symmetric Davidson (`Solvers.davidson.solve_symmetric`): its default
# residual tolerance and cycle cap; the bounds its davidson1 linear-dependence
# threshold is held between (1e-2 |r|^2, never looser than pyscf's own 1e-14
# and floored well above round-off: a vector orthogonalized to a squared norm
# of 1e-22 still has ~6 honest digits left); and the norm below which a
# direction projected off the converged set counts as null.
DAVIDSON_TOL_RESIDUAL = 1e-6
DAVIDSON_MAX_CYCLE = 200
DAVIDSON_LINDEP_FLOOR = 1e-22
DAVIDSON_LINDEP_CEILING = 1e-14
DAVIDSON_NULL_DIRECTION = 1e-8


# Share of a rank's own slice of the fitted tensor that the distributed DF
# build (`Base.factorization.distributed_df`) may hold in the transients of its exchange.
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
# `gradients.gwbse.bse_isdf.bse_cache`, (naux, nocc, nvir) each, ten of them alive at
# once; 'grid' is the reverse of the ISDF block action
# (`gradients.gwbse.isdf_bse_adjoint`), which closes over the grid and forms no
# three-index block at all. The same derivative to rounding, not the same
# bits, so the vocabulary is spelled once for the chain and the record.
BSE_ADJOINTS = ('explicit', 'grid')

# Grid points per row tile of the grid BSE adjoint
# (`gradients.gwbse.isdf_bse_adjoint`). Fixed, never derived from a memory
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
# (`gradients.gwbse.isdf_bse_adjoint`): slab s is summed by rank s % nranks
# over the grid tiles in tile order, and the slabs are then gathered verbatim.
# Fixed, like BSE_ADJOINT_TILE_ROWS and for the same reason: a slab is a
# GEMM's row count, so it must not follow the rank count. The grid index is
# cut in BSE_ADJOINT_TILE_ROWS tiles on both sides of every M^2 product, so
# one pass holds (2 nT + 4) (tile, tile) blocks, 6.3 MB, not (rows, M) tiles.
BSE_ADJOINT_AUX_ROWS = 256

# The largest C int: a count a 32-bit LAPACK, MPI or libcint argument holds.
INT32_MAX = 2 ** 31 - 1

# The largest count one MPI call may carry, in elements of the call's
# datatype. mpi4py hands the library an MPI_Count only where the library has
# the MPI-4 large-count routines (MPI_Allreduce_c, ...), which Open MPI 5.0.x
# does not provide; its fallback narrows the count to a C int and raises
# MPI_ERR_ARG past 2^31 - 1, a count proj(tau) and D pass at large M, so every
# collective of `Base.utils.parallel.mpi_grid` moves at most this many float64 per
# call, in windows; below it a call is one window, the unchunked call itself.
MPI_COUNT_MAX = INT32_MAX

# Relative Coulomb-energy error above which the ISDF-K mean field
# (`Base.factorization.isdf_jk.ISDFJK.check`) warns that its grid is weak, on the first density the SCF
# asks about: benzene/cc-pVDZ on `optimize_atomic_radii`'s default single
# descent puts 0.31 Ha of error into a single J element and collapses the SCF
# by 129 eV, while K stays good to 5e-3.
ISDF_JK_CHECK_TOL = 1e-3

# Fraction of what max_memory leaves that the distributed ISDF-K SCF
# (`Base.factorization.distributed_isdf_jk`) lets this rank's rows of Z take, for the two
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
# three-centre pass (`Base.factorization.separable_ri`): a shell block's test co-densities
# are screened and built a slab of grid rows at a time, and its kept
# (mu nu|P) rows gathered a slab of auxiliary functions at a time, so no
# block is ever whole and each slab is made, scaled and reduced while it is
# still in cache. A cost knob only: every element is the same product whatever
# the slab.
FIT_ROW_CHUNK_BYTES = 4 << 20

# Edge, in grid points, of the square tiles in which the replicated ISDF fit
# (`Base.factorization.separable_ri`) writes the transpose of its Gram matrix's lower block
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

# Bohr. Largest atomic displacement over which a Casida Davidson along a walk
# starts from the previous geometry's converged vectors, carried into this
# geometry's orbitals (`davidson.DavidsonWarmStart`), rather than from unit
# vectors alone: the SCF warm start's rule, for the same steps.
DAVIDSON_WARM_START_MAX_SHIFT = SCF_WARM_START_MAX_SHIFT

# Fraction of this rank's max_memory (the allocation's share a job hands pyscf,
# `Base.utils.parallel.memory.allocation_max_memory_mb`) that the Davidson's trial space
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

# Working set, in GB, of one occupied tile of the density-fitted folded
# ADC(2) operator (`eeADC.ee_ri_fold`): the tile holds a handful of
# (n_tile nvir, nocc nvir) doubles-shaped blocks, and its size moves only the
# memory and the GEMM shapes, never a number beyond the summation order.
EE_RI_TILE_GB = 8.0

# Spin-component scaling (c_os, c_ss) of the doubles in ADC(2): SCS (Grimme,
# J. Chem. Phys. 118, 9095 (2003); Hellweg, Gruen and Haettig, Phys. Chem.
# Chem. Phys. 10, 4119 (2008)) and SOS (Jung, Lochan, Dutoi and Head-Gordon,
# J. Chem. Phys. 121, 9793 (2004)), the factors of Turbomole's ricc2.
ADC2_SPIN_SCALING = {'scs': (6.0 / 5.0, 1.0 / 3.0), 'sos': (1.3, 0.0)}

# Auxiliary functions per chunk while the occupied/virtual factors of the
# folded ADC(2) operator are transformed from the AO three-index tensor; the
# chunk holds (n_aux_chunk, nao, nao) doubles.
EE_RI_AUX_CHUNK = 128

# ---------------------------------------------------------------------------
# Root finders, static screening, ISDF span and grid identity, reaction-field fit
# ---------------------------------------------------------------------------

# Separable-RI factors on near-null AO combinations (`separable_ri`): the
# relative Coulomb residual of an orbital density against its RI-V fit beyond
# which the factors do not represent it. ISDF_DENSITY_RESIDUAL_TOL selects the
# S-normalized overlap eigenfunctions of `represented_span`, which also drops
# every eigenvalue below ISDF_SPAN_OVERLAP_FLOOR (coefficient norms above
# 1e3); ISDF_ORBITAL_RESIDUAL_TOL refuses a set of orbitals
# (`refuse_unrepresented_orbitals`). Measured on CBP gas aug-cc-pVDZ at
# (24,15,9,3): worst virtual density 1.9 and min eig(A - B) -1.7 Ha.
ISDF_DENSITY_RESIDUAL_TOL = 1e-2

ISDF_ORBITAL_RESIDUAL_TOL = 5e-2

ISDF_SPAN_OVERLAP_FLOOR = 1e-6

# A refusal names at most ISDF_REFUSED_ORBITALS_NAMED orbitals, worst first;
# the fitted densities D^T X_mo^2 are summed in tiles of
# ISDF_FITTED_DENSITY_ROWS grid rows.
ISDF_REFUSED_ORBITALS_NAMED = 12

ISDF_FITTED_DENSITY_ROWS = 4096

# A UHF Casida exciton whose charge amplitude chi_a for orbital p has norm below
# this is an M_s = 0 triplet component, which the restricted singlet spectrum
# does not hold: it is dropped from the singlet-only PSD sums (PSD1/5/6/8) so a
# closed shell gives the RHF self-energy (agreement measured at 3e-9 eV on
# water/6-31G). The triplet modes PSD2/4/7/9 keep it, as the spin-orbital eh
# pair does.
UHF_PSD_CHARGE_AMPLITUDE_MIN = 1e-5

# Ha. Scan windows of the quasiparticle root search (`qp_equation.
# find_qp_root`) on a self-energy known pointwise, eps -+ half width: (|eps|
# above which the tier applies, half width). Deeper states have
# larger corrections and their satellites sit further out.
QP_SEARCH_WINDOW_TIERS = ((0.0, 0.5), (1.5, 1.0), (4.0, 2.0))

# Floor on |Z| below which the periodic root scan (`pbc_self_energy.
# qp_energy_g0w0`) counts a root as a pole artifact (Z -> 0 beside a pole of
# Sigma_c) when it reports how many roots a metal's band has. Not a
# quasiparticle threshold: that is QP_WINDOW_Z_MIN.
PBC_QP_Z_MIN = 0.05

# Ha. Bisection tolerance of the periodic root scan's sign changes.
PBC_QP_ROOT_TOL = 1e-10

# Periodic GW and RPA (Periodic/):
# PBC_RPA_NFREQ: Gauss-Legendre points of the periodic dRPA frequency
#   quadrature (RI-V and THC routes).
# PBC_ISDF_CHOLESKY_STOP: the pivoted Cholesky point selection stops once the
#   largest Schur-complement diagonal falls to this times the first.
# PBC_ISDF_REGULARIZATION: ridge on the ISDF normal equations, relative to
#   trace(Z)/n. Z is Hermitian PSD by construction but is deliberately driven
#   towards rank deficiency -- more interpolation points means a better fit AND
#   a worse-conditioned Z -- so the solve is regularized rather than assumed
#   safe.
# PBC_KERNEL_NEGATIVE_TOL: a Coulomb kernel value below -this times
#   max(its largest, 1) is negative and refused (it would make V^q indefinite).
# PBC_METRIC_EIG_FLOOR: an eigenvalue of the RI-V Coulomb metric below -this
#   times the largest is negative and refused; one below +this times the
#   largest is a linear dependency of the auxiliary basis and is dropped.
# PBC_K_HEAD_TOL: |k| below which the 2pi/k head is replaced by its
#   Brillouin-zone-cell average.
PBC_RPA_NFREQ = 32

PBC_ISDF_CHOLESKY_STOP = 1e-10

PBC_ISDF_REGULARIZATION = 1e-10

PBC_KERNEL_NEGATIVE_TOL = 1e-10

PBC_METRIC_EIG_FLOOR = 1e-10

PBC_K_HEAD_TOL = 1e-8

# Occupation-weighted response (`pbc_occupations`): pairs whose occupation
# difference falls below PBC_OCCUPATION_THRESHOLD carry no useful weight and
# would contribute a zero row to A - B (0/0 for a degenerate pair), so they are
# dropped; measured cost on a Li monolayer: 0.0008% of the weight. Occupations
# are integer when every mo_occ is within INTEGER_OCCUPATION_TOL of 0 or 2.
PBC_OCCUPATION_THRESHOLD = 1e-4

INTEGER_OCCUPATION_TOL = 1e-8

# k-mesh symmetry (pbc_symmetry): an operation is used only if its AO rotation
# carries every overlap S^k onto S^{Sk} to this; Si gth-dzvp meets it at 4e-11,
# the precision of the lattice sums.
KMESH_SYMMETRY_TOL = 1e-8

# A walk step whose evaluation raises `StepFailed` is retried from the last
# accepted geometry with the radius times WALK_STEP_SHRINK, by the Cartesian
# walker and by geomeTRIC alike; below WALK_TRUST_MIN_BOHR (Bohr, the norm of
# the 3N-vector step) the radius has collapsed and the walk stops.
WALK_STEP_SHRINK = 0.25

WALK_TRUST_MIN_BOHR = 2e-3

# The walk's step cap and its trust radius (Bohr, the norm of the 3N-vector
# step). The first step is taken on a guessed Hessian, and a longer one can
# carry a quasiparticle root past where the frozen Newton seed finds it.
WALK_MAX_CYCLE = 50
WALK_TRUST_BOHR = 0.1
WALK_TRUST_MAX_BOHR = 0.5
# ... and those of a mean-field ground state's own relaxation
# (`optimize.relax_ground_state`): no quasiparticle root to carry, so the
# first step may be as long as a ground-state optimizer takes
WALK_GROUND_MAX_CYCLE = 100
WALK_GROUND_TRUST_BOHR = 0.3

# Largest relative error of the auxiliary representation of the reaction field
# a solvated run accepts: || G_fit - G || / || G || over the pair densities of
# the SOLVENT_RI_PROBE_ORBITALS highest occupied x lowest virtual orbitals, G
# the cavity potential of a pair density and G_fit that of its Coulomb fit
# (`SolventScreening.fit_residual`). vtilde_PQ itself is the exact cavity
# contraction; the residual is the fit's missing tail on the cavity, and it
# converges with the auxiliary set's diffuse functions (an even-tempered set
# reaches the exact contraction to 0.02 meV). Measured, IEF-PCM toluene,
# PBE0, static-W BSE: the basis's own -ri set gives 3.9-6.1% at cc-pVDZ
# (water 6.1%: S1 +10.0, T1 -4.7, dEST +14.6 meV; formaldehyde and planar
# DMABN: <= 1.3 meV) and 1.4-1.5% at aug-cc-pVDZ (<= 5 meV); cc-pvdz-ri on
# aug-cc-pVDZ gives 12.6-16.8% and 20-44 meV.
SOLVENT_RI_RESIDUAL_MAX = 0.08

SOLVENT_RI_PROBE_ORBITALS = 20

# Turning rate (rad per Bohr of one nuclear coordinate) above which an atom's
# covariant ISDF frame is reported (`separable_ri.frame_turning_record`). The
# rate is 1/gap of the atom's neighbour-tensor eigenvalues times its slope:
# 0.3-3 rad/Bohr on ordinary atoms (water 0.9, DMABN twisted amine N 3.0),
# 16.2 on DMABN's planar amine N and 18.9 on the ladder formaldehyde's carbon,
# where turning frames made the GH Hessian 2.7 percent asymmetric. Reported,
# not refused: Hessians and walks freeze the frames (`isdf_jk.FrozenFrames`).
FRAME_TURNING_RATE_BAR = 5.0

# The upfolded BSE's non-symmetric Davidson: residual norm, cycle cap, the
# floor on |theta - H_ii| of the diagonal preconditioner, and the norm below
# which a new direction counts as already spanned.
UPFOLDED_BSE_DAVIDSON_TOL = 1e-7

UPFOLDED_BSE_DAVIDSON_MAX_CYCLE = 100

UPFOLDED_BSE_PRECOND_FLOOR = 1e-8

UPFOLDED_BSE_LINDEP_TOL = 1e-8

# Relative imaginary part (|Im| / max(1, |Re|max) Ha) above which a returned
# upfolded-BSE root is complex: the dynamical BSE has no real solution there.
UPFOLDED_BSE_IMAG_TOL = 1e-6

# Orbital-energy separation (Ha), at the geometry the explicit set is decided
# at, below which neighbouring members on one side of the gap share a
# subspace of the quasiparticle Hamiltonian (`ExcitedStateChain(
# qp_hamiltonian=True)`), frozen with the set. Generous: H is invariant to
# rotations inside a subspace, so a pair that later crosses keeps a smooth
# surface, and one that never mixes keeps its own corrections to second order
# in Sigma_pq / gap. DMABN/cc-pVDZ LUMO and LUMO+1 are 0.43 eV apart at R0
# and cross on the S1 walk.
QP_HAMILTONIAN_WINDOW = 0.03

# The quasiparticle Hamiltonian of one subspace solved to self-consistency
# (`sum_over_poles.subspace_hamiltonian`): the largest element (Ha) of the
# residual e0 + Sigma^c[H] - H at which the Newton takes its last step, and
# the cap on its steps (from the frozen seed one or two steps suffice).
QP_SUBSPACE_TOL = 1e-13

QP_SUBSPACE_MAX_ITER = 200

# Interior samples of an interval between neighbouring poles that no bound
# certifies as holding exactly one root (`qp_equation.PoleSum`): a negative
# fitted amplitude at one of its ends, or negative ones nearby strong enough
# to bend f. The sign of f next to a pole is known from its amplitude, and a
# turning point of f between two samples is located on the closed-form slope,
# so the 0 or 2 crossings an interval with mixed-sign amplitudes holds are
# found. An unbounded outer interval is sampled on QP_GRAPHICAL_N_OMEGA points.
QP_POLE_INTERVAL_SAMPLES = 3

# The pole-aware grid scan of a self-energy known only pointwise
# (`qp_equation.SampledEquation`: contour deformation, analytic continuation).
# f = w - eps - Sigma rises with slope >= 1 between the poles of a
# positive-weight Sigma, so a grid cell whose secant slope is below
# QP_SCAN_SECANT_MIN hides a pole, and with it possibly a root beside it with
# no sign change at the cell's ends; such a cell is halved, at most
# QP_SCAN_SPLIT_DEPTH times, until the sign changes show.
QP_SCAN_SECANT_MIN = 0.5
QP_SCAN_SPLIT_DEPTH = 6
# The scan's points per window are QP_GRAPHICAL_N_OMEGA whether f has a
# closed-form slope or not. A cell must not hold a root, a pole and a second
# root, which show as one sign change: 40 points (25 mHa cells) lost the
# Z 0.54 root of water/cc-pVDZ's orbital 15 1.80 Ha up to its Z 0.37 partner
# 20 mHa away, which 150 points resolve.
# The pole strengths of every root of a positive-weight Sigma sum to one, so
# once the weights a scan window has refined leave no more than the largest of
# them for the brackets not yet refined, those cannot hold a larger weight and
# are passed over (`qp_equation._weighed_roots`). QP_WEIGHT_SUM_SLACK is the
# margin kept on that sum against the quadrature's error in each Z.
QP_WEIGHT_SUM_SLACK = 1e-3

# Width of the energy bins whose median correction is one knot, eV; probes
# within one bin of the window edge are left to the edge's explicit root.
OUTSIDE_CURVE_BIN_EV = 1.0

# Largest excursion, eV, of the curve's spline beyond the values of the two
# knots bounding an interval, read at the members' eigenvalues; a curve past it
# is refused at build time. Measured on the reference curves of LRC-wPBEh
# ISDF-K/cc-pVDZ: formaldehyde 0.046 (gas) and 0.005 eV (toluene), DMABN 0.15
# (gas) and 0.20 eV (toluene), all on the virtual side.
OUTSIDE_CURVE_OVERSHOOT_EV = 0.5

# Ritz pairs the Davidson tracks beyond the requested roots, Tamm-Dancoff and
# Casida alike, as max(floor, nroots // divisor). A root whose guess pair is
# dressed far below its own diagonal by couplings to high pairs starts as a
# Ritz value ABOVE the next root up, and a solve that corrects only the lowest
# nroots then converges on that next root instead: formaldehyde/cc-pVDZ
# triplet in toluene, 4 Tamm-Dancoff roots, returned 0.366009 Ha for a fourth
# root at 0.363695 Ha, every residual below 1e-9. A tracked extra pair is
# corrected until it converges or its residual interval lies above the highest
# requested root: theta - |r| for the Hermitian A, theta - r_S for the Casida
# problem, r_S the residual of J S in the metric S = [[A, B], [B, A]].
DAVIDSON_EXTRA_ROOTS_FLOOR = 2

DAVIDSON_EXTRA_ROOTS_DIVISOR = 2

# How a BSE state is followed between geometries beyond its index: by the
# overlap of its Casida vector with the last accepted one.
ROOT_FOLLOWING = ('overlap',)

# Auxiliary gauge of a static W handed to the ISDF BSE: the separable fit's
# symmetric V^-1/2 metric root. pyscf's cderi is L^-1-whitened, an orthogonal
# rotation of the same index, so a W in that gauge has the right shape and is wrong.
SEPARABLE_AUX_GAUGE = 'separable'


# Relative truncation of the IR singular values (Shinaoka et al., Phys. Rev. B
# 96, 035147 (2017)) for the molecular imaginary-axis grids: the IR transforms
# of `TimeFrequencyGrid.ir`, the default `IRBasis` and the finite-temperature
# Pade cap `ir_continuation_order`. Fits land near this accuracy.
IR_EPS = 1e-10

# Relative to max |a|: the skew part V^T A V - (A V)^T V (and B's) of the
# Casida Davidson's projected blocks that a symmetric block action leaves.
# Measured at most 8.8e-16 on water and benzene/cc-pVDZ (ISDF and DF, BSE,
# TDHF, RPA, one and three ranks); a grid reduction one row off its block
# leaves 1.2e-2 to 0.46 and roots converged 0.19 Ha low.
DAVIDSON_ACTION_SYMMETRY_TOL = 1e-8

# What a pin recorded under one BLAS may miss by under another, per quantity
# class (tests/pins.py; bitwise under the recording BLAS). Measured on a
# cluster's MKL 2025.3 against a workstation's MKL 2021.4/SSE4.2 + Accelerate.
# Energies, Ha: up to 1.4e-12 (water RPABSESurface[singlet]; mean fields 6e-14
# to 4e-13); a pole strength not read off a continuation holds to the same
# number, dimensionless.
PIN_CROSS_BLAS_BAR = 2e-12

# Forces and gradients, Ha/Bohr, the ISDF gradient floor: 1.15e-9 to 2.2e-9
# (surface_comm 1.15e-9 to 1.82e-9, outside_scissor R0 2.0e-9 to 2.2e-9).
PIN_CROSS_BLAS_GRADIENT_BAR = ISDF_GRADIENT_FLOOR

# Quantities of an analytic continuation, which amplifies the fit's last
# bits: the Pade route's quasiparticle energy, Ha, and its pole strength Z,
# dimensionless (Z the continuation's derivative; st-pade HOMO Z missed by
# 5.6e-11 water, 2.1e-11 formaldehyde, 5.6e-10 thioformaldehyde).
PIN_CROSS_BLAS_CONTINUED_BAR = 1e-9

# Vectors and orbital coefficients, after sign alignment, under another BLAS
# than the one a pin was recorded on; dimensionless.
PIN_CROSS_BLAS_VECTOR_BAR = 1e-10

# Roots and matrix elements under another BLAS, Ha; it holds the GWGammaInf
# vertex self-energy's 1.2e-10 miss.
PIN_CROSS_BLAS_ROOT_BAR = 2e-10

# Two computations of one number on the recording machine (two solves, two
# routes, a run and an archived tree's run) are not bitwise: a BLAS reduction
# taken in another order reproduces only to an ulp (1.1e-16 Ha, one ulp of a
# 0.689 Ha operand, measured in one cluster run in five). They are held
# to this many ulp of the larger of their magnitude and one unit (tests/pins
# .same_bar): the unit floor because a
# difference of O(1) operands keeps their ulps, not its own. 8 ulp: a few
# roundings of a reduction, far below every class bar (2e-12 Ha at its least).
PIN_SAME_ULPS = 8

# Two Davidson solves of one problem (two rank splits) each sit within their
# residual |r| <= conv_tol of an eigenvalue (Weyl: some eigenvalue lies within
# |r| of a Ritz value), so their roots differ by at most this many conv_tol.
PIN_DAVIDSON_ROOT_FACTOR = 2


# ---- the gradient and property layer: one Numerics object, the conventions,
# the refreeze outer loop and the rate table

# The three sublevels M = -1, 0, +1 of a triplet. A crossing that STARTS in a
# triplet (RISC, T1 -> S0) averages its rate over them, thermalized when the
# zero-field splitting is far below kT, so it carries |V|^2 / 3 where the sum
# over the final sublevels gives ISC the whole |V|^2.
TRIPLET_SUBLEVELS = 3

# The ISDF fit's choices (`IsdfNumerics`): the angular cutoff of the second
# index of its test co-densities, its pair screening and its Tikhonov shift.
ISDF_FIT_L_MAX_SECOND = 2

# Pair-screening threshold: a test co-density is dropped when its peak
# amplitude anywhere on the interpolation grid falls below this times the
# global maximum. It is what makes the pair count linear in system size.
#
# NOT a Schwarz bound, and it does not read as an accuracy target. A screened
# pair is not set to zero in the result -- the factorization is dense in the
# pair index, so it still returns a value there, just one no equation
# constrained. So the threshold is a statement about which CONSTRAINTS are
# redundant, and the failure is a cliff rather than a slope. Measured on
# benzene and naphthalene at cc-pVTZ, fit error relative to 1e-10:
#
#     1e-06   1.00x        76% / 58% of columns kept
#     1e-05   1.37x/1.97x  65% / 47%
#     1e-04    102x/136x   45% / 29%
#
# 1e-06 is free -- identical to four figures, and the BSE roots move under
# 0.03 meV at cc-pVDZ against the ~5 meV the grid itself contributes. There is
# nothing to harvest past it, so do not tune this looking for more.
ISDF_FIT_PAIR_TOL = 1e-6

# Their eq 9 regularization, "a reasonable parameter for double precision".
# Delesma, Golze and Rinke, reimplementing this in FHI-aims, report that "the
# inversion is numerically stable without L2 regularization" -- so derive 0.0
# (`Numerics.derive`) if the balanced Gram matrix is well conditioned for your
# grid, and compare.
ISDF_FIT_REGULARIZATION = 4e-7

# Geometries whose evaluation a surface retains (`gradients.evaluation`): the
# mean field, the forward pass, the roots and, once asked, the force. A walk
# holds its accepted geometry and the trial step beside it, and every later
# question about either -- the record's energy, the ground state there, a
# root-following match -- reads the same SCF and the same roots. Only the
# newest geometry keeps its forward pass whole; the older keeps its roots.
EVALUATION_CACHE_SIZE = 2

# What a mean field along a GEOMETRY OPTIMIZATION is converged to: the energy
# test at its rounding floor, and an orbital gradient whose bias on the force
# (about ten times |g| through the orbital response) is four decades under
# GEOM_OPT_CONV's residual. A quantity that feeds a finite difference
# (a numerical Hessian, a displaced spin-orbit element) keeps
# SCF_DIFFERENTIABLE_GRAD_TOL.
SCF_OPTIMIZATION_CONV_TOL = SCF_ENERGY_CONV_TOL
SCF_OPTIMIZATION_GRAD_TOL = 1e-8

# Cycle cap of a production mean field (`photophysics.chain`); the PCM
# re-convergence copies it.
SCF_MAX_CYCLE = 200

# pyscf's exchange-correlation integration grid level of every production
# Kohn-Sham mean field (`ScfNumerics.xc_grid`). On pyscf's default level 3, LRC-wPBEh
# steps the Kohn-Sham orbital energies by up to 1.2e-7 Ha between geometries
# 2.5e-4 Bohr apart (not E_KS), which G0W0 carries into every quasiparticle
# energy and excitation energy; level 4 is smooth there (third differences of
# eps_QP 1.7e-11 Ha, as level 5's) and costs DMABN/cc-pVDZ +1% per SCF and +6%
# per force, against 2.1e-5 Ha/Bohr and 3.3e-6 Ha off level 5 on level 3,
# 6.5e-6 and 1.3e-7 on level 4. A record written before the field existed
# was on level 3 (`Numerics.from_record`).
XC_GRID_LEVEL = 4
# The level of a record that names none: pyscf's default, which every mean
# field was built on until the field existed.
XC_GRID_LEVEL_UNRECORDED = 3

# How many of the largest |H_ixjy - H_jyix| a refused finite-difference
# Hessian lists, with their atom/axis pairs, in the asymmetry.json it leaves.
HESSIAN_ASYMMETRY_WORST = 10

# A followed state whose best overlap falls below ROOT_FOLLOW_WEIGHT_MIN is
# searched for again among this many times the roots (at most the whole pair
# space); found nowhere, the step is refused rather than continued on another
# state.
ROOT_FOLLOW_RESOLVE_FACTOR = 2

# Orbital-energy separation (Ha) below which neighbouring orbitals count as one
# degenerate block (one separation for both uses). A quasiparticle
# window is widened so it never cuts one, since inside a degenerate subspace a
# single orbital is no smooth function of the geometry. And, at the geometry the
# explicit set is decided at, neighbouring members of the set closer than this
# form a block carrying ONE correction, the mean of its members'
# eps^QP_p - eps_p, frozen with the set -- for true degeneracies only
# (QP_BLOCK_CORRECTION_TOL): orbitals merely close in energy have corrections of
# their own, and averaging them moves each by half their difference
# (formaldehyde/cc-pVDZ HF orbitals 12/13, 4.2e-3 Ha apart, corrections 370 meV
# apart: SOP - AC -188/+185 meV averaged, -3.0/+0.3 meV each on its own).
QP_DEGENERACY_TOL = 1e-4

# A BAR, not a choice: the spread (Ha) of the
# members' own corrections eps^QP_q - eps_q, at the reference geometry, within
# which a QP_DEGENERACY_TOL block is a true degeneracy and shares one
# correction; an accidental near-degeneracy has corrections of its own
# (PXZ-TRZ/aug-cc-pVDZ virtual pairs 1e-4 Ha apart, corrections 30-170 meV
# apart: averaged, SOP T1 moves by -22 meV).
QP_BLOCK_CORRECTION_TOL = 1e-4

# The frozen conventions keyed by orbital identity (`gradients.gwbse.conventions`):
# the least weight an orbital of a displaced geometry must hold in the span of
# the frozen unit (a near-degenerate group, one explicit orbital, or the holes
# and outside orbitals sharing one frozen shift) it is matched to. Between a
# quarter and three quarters mixed with another unit an orbital belongs to
# neither, and its convention is refused, not guessed; with any bar at one half
# two crossing orbitals would only ever swap labels, a step in the surface.
CONVENTION_SPAN_MIN = 0.75

# Kaltak-Klimes-Kresse residual that buys ~0.1 meV on the quasiparticle
# energy. Calibrated on naphthalene/cc-pVDZ at R_Sigma = 486, against the
# ntau -> infinity limit of the space-time HOMO:
#     eta 1.2e-08 -> 1.96 meV     eta 6.0e-10 -> 0.32 meV
#     eta 4.8e-11 -> 0.08 meV     eta 3.7e-12 -> 0.03 meV
# The mapping is empirical -- eta bounds the quadrature, not the self-energy
# it is used to integrate -- so this is a calibrated default, not a bound.
#
# The default of `QuadratureNumerics.tau_target`, the ONE minimax target every
# route sizes an 'auto' tau grid at (`minimax_points_for_accuracy`
# has no second default of its own); the molecular and periodic space-time
# routes read it through `utils.time_frequency`.
DEFAULT_TAU_TARGET = 1e-10

# The space-time self-energy's Pade continuation: its points on the
# Gauss-Legendre frequency grid of scale GL_FREQUENCY_SCALE (Ha), which the
# frequency axis of chi0 shares where its size is given explicitly.
DEFAULT_NPADE = 16

GL_FREQUENCY_SCALE = 1.0

# How the frozen scissor beyond the explicit window is shaped
# (`GW.qp_states.OutsideCurve`): 'tiers' lends each orbital the shift of the
# explicit orbital nearest in energy, a constant; 'curve' gives it Delta(eps_p)
# of a natural cubic spline (C2) through energy-binned analytic-continuation
# corrections of every orbital, anchored on the window edge's own root. On
# PXZ-TRZ/aug-cc-pVDZ
# the virtual corrections above the window run from -0.5 to +7.6 eV and the deep
# occupied average -4.6 eV, where the tiers lend -0.70 and -0.03 eV.
OUTSIDE_SCISSOR_MODES = ('tiers', 'curve')

# The Casida eigensolvers a surface may ask for; 'auto' is the memory rule
# (`LinearResponse.casida.solver_choice`).
BSE_SOLVERS = ('auto', 'dense', 'davidson')

# Unit-vector guesses per requested root. One each is what a diagonal-dominant
# argument suggests, and it silently returns the WRONG STATES whenever the
# smallest orbital-energy differences are degenerate. A unit vector in the
# occupied-virtual pair basis carries a definite pair symmetry and both A and B
# are block diagonal over it, so the Davidson subspace never leaves the irreps
# its guess started in: it converges, to `nroots` roots that are not the lowest
# ones. benzene/cc-pVDZ RPA at nroots=5 puts a degenerate quartet plus half of
# the next degenerate pair in the guess and returns
#     0.483476 0.519125 0.523487 0.570116 0.570117
# against a true lowest five of
#     0.483476 0.519125 0.519134 0.523487 0.553819
# -- with converged True on all five. So `converged` does not catch this; only
# a better-spanning guess does, and the two fixes are independent.
#
# Over-requesting is a heuristic, not a guarantee -- it buys directions in more
# irreps, and nothing here knows which irreps the true lowest roots live in.
# Completing the degenerate set alone is NOT enough (measured: six guesses on
# that benzene case still miss the fifth root); 2x plus completion is. A result
# worth trusting is worth repeating at a raised `guess_factor`.
DAVIDSON_GUESS_FACTOR = 2

# A Marcus-Levich-Jortner rate splits a crossing's reorganization energy
# lambda, taken from the accepting state's energies at the two minima, into the
# Huang-Rhys sum rule sum_k S_k omega_k of the displaced modes and a classical
# remainder lambda - sum_k S_k omega_k. Where the sum rule reaches or exceeds
# the energy-based lambda the harmonic displacement model has failed (a
# torsion projected on Cartesian normal modes inflates S), and the Marcus and
# MLJ rates are refused with both lambdas recorded, never run on a clamped
# remainder. lambda is a difference of two excited-state energies, each
# carrying up to REFREEZE_TOL_MEV of convention drift, so a remainder within
# twice that is not resolved and is refused as well.
RATE_LAMBDA_REMAINDER_MIN_MEV = 2.0 * REFREEZE_TOL_MEV

# Quanta of the effective mode the rate table's Marcus-Levich-Jortner sum
# keeps; the Poisson weight it drops is refused above sqrt(eps) by the rate.
RATE_MLJ_N_MAX = 400

# Default eigenvalue tolerance (Hartree) of an excitation-energy ADC solve
# (`eeADC.ee_driver.solve_ee_adc` and the ADC(2) state model)
EE_ADC_CONV_TOL = 1e-8

# GB of MP1 amplitudes (one spin channel, this rank's occupied tiles) that the
# folded ADC(2) operator keeps for the second-order singles block instead of
# rebuilding (kc|ld) in every product; tiles past it are rebuilt each time.
EE_RI_AMPLITUDE_CACHE_GB = 32.0

# Auxiliary functions per chunk of the exchange terms of one folded ADC(2)
# product (`eeADC.ee_ri_fold`), the chunks dealt over the MPI ranks and summed
# in chunk order; never set by the rank count, so neither are the bits.
EE_RI_EXCHANGE_CHUNK = 128
# (chunk, nv, nv) arrays alive in one exchange chunk of a folded product
EE_RI_EXCHANGE_ARRAYS = 3
# (kt, nv, no, nv) arrays alive in one occupied tile of a folded product
EE_RI_FOLD_TILE_ARRAYS = 5
# ... in one occupied tile of the RI difference density (`eeADC.ee_ri_density`)
# and in one virtual tile of its vv pass
EE_RI_DENSITY_OCC_TILE_ARRAYS = 9
EE_RI_DENSITY_VIRT_TILE_ARRAYS = 6

# The configuration-dependent part of the folded doubles' diagonal, for the
# block solver's seeds: a configuration (i, a) is pulled down by the
# relaxation of its particle, sum_cld (ac|ld)^2 / (w + e_i + e_l - e_c - e_d),
# and of its hole, sum_kld (ki|ld)^2 / (w - e_a + e_k + e_l - e_d). The weights
# are fitted to the exact diagonal of small molecules, singlets and triplets:
# exchange cancels most of the particle term, the hole term holds its weight.
EE_RI_FOLD_DIAG_PARTICLE = 0.5
EE_RI_FOLD_DIAG_HOLE = 1.5
# 1/x = sum_j W_j exp(-t_j x) by the trapezoid rule in ln t (step below, error
# ~exp(-pi^2 / step)), for x from EE_RI_FOLD_DIAG_GAP Hartree to the largest
# doubles energy: the small-t tail is cut where it holds EE_RI_FOLD_DIAG_TAIL
# of 1/x at the largest x, the large-t end where exp(-t x) at the smallest x
# has fallen to exp(-EE_RI_FOLD_DIAG_EXPONENT)
EE_RI_FOLD_DIAG_STEP = 0.75
EE_RI_FOLD_DIAG_GAP = 1e-2
EE_RI_FOLD_DIAG_TAIL = 1e-4
EE_RI_FOLD_DIAG_EXPONENT = 30.0

# The block solver of the folded ADC(2) roots (`eeADC.ee_fold`, method
# 'block'): each trial vector's folded doubles are made at Chebyshev
# frequencies spanning the roots, and A_eff(w) between them is their Lagrange
# interpolant. For a doubles pole D above the window the interpolant's
# relative error is |node_poly(w) / node_poly(D)| <= 2 (h/2)^p / (d - h)^p
# (h the window's half-width, d the distance of its centre to min D); the
# node count is the smallest that puts this times the fold |y A_fold y| under
# EE_FOLD_BLOCK_NODE_TARGET times tol_omega, the fold taken as
# EE_FOLD_BLOCK_FOLD_BOUND times the largest of the start vectors'. The bound
# puts all doubles weight at min D and lies far above the actual error, so the
# target is tol_omega itself; only an exact product at a root's own frequency
# converges it, and that product sees any shortfall.
EE_FOLD_BLOCK_FOLD_BOUND = 2.0
EE_FOLD_BLOCK_NODE_TARGET = 1.0
EE_FOLD_BLOCK_MAX_NODES = 12
# Hartree added on both sides of the roots' first estimates to make the window
EE_FOLD_BLOCK_MARGIN = 1e-2
# Ritz pairs carried above nroots, so the highest asked root is not the edge
EE_FOLD_BLOCK_BUFFER = 2
# times the window may move to roots that left it, each a restart
EE_FOLD_BLOCK_MAX_MOVES = 3
# block iterations before the unconverged roots go to the per-root Newton
EE_FOLD_BLOCK_MAX_ITER = 60
# fraction of tol_residual the subspace model converges to; the rest is left
# to the interpolation error that the exact product at the root sees
EE_FOLD_BLOCK_MODEL_RESIDUAL = 0.5
# A root's error goes as |r|^2 / delta, delta its distance to the eigenvalues
# outside the subspace; the RI driver asks residual sqrt(tol_omega delta) of
# the roots with this delta (Hartree), so their error stays under tol_omega
EE_FOLD_RESIDUAL_GAP = 0.1
# window half-widths beyond the nodes within which a model root is sought
EE_FOLD_BLOCK_BRACKET = 3.0
# norm below which an orthogonalized correction adds nothing to the subspace
EE_FOLD_BLOCK_LINDEP = 1e-10
# Newton/bisection on a model root: steps, and the relative step that ends it
EE_FOLD_MODEL_ROOT_MAX_ITER = 100
EE_FOLD_MODEL_ROOT_STEP = 1e-15

# The Franck-Condon rows (multimode Condon and Herzberg-Teller) share one
# density built from the S1-T1 displacement, whose harmonic reorganization is
# the sum rule sum_k S_k omega_k. A crossing's FC rows are refused when that
# sum rule and the crossing's energy-based lambda (its accepting state's, the
# one the Marcus rows use) differ by more than this factor either way: the
# displacement model then does not describe the crossing. The bar catches a
# failed model (a Cartesian projection of a torsion is off by 3-15x), not a
# quantitative one: in the inverted region a 30% change of lambda already
# moves the classical rate by orders of magnitude.
RATE_FC_LAMBDA_RATIO_MAX = 2.0

# Thermal reach of the state manifold in the rate table, in k_B T: every
# initial state within this of the lowest one at its minimum is Boltzmann
# populated, and every final state up to this above the higher of the initial
# and the lowest final state is a channel (exp(-10) = 4.5e-5).
RATE_THERMAL_WINDOW_KT = 10.0

# Curvilinear displacements (`properties.curvilinear`): a valence angle
# above this many degrees is linear and neither it nor a dihedral through it is
# a primitive (both are undefined at 180 degrees); the directions they would
# span are carried by the Cartesian displacement instead.
CURVILINEAR_LINEAR_ANGLE_DEG = 175.0
# Relative singular-value cut of the mass-weighted Wilson B matrix's
# pseudo-inverse. Its redundant and rigid-body singular values are zero to
# round-off and the smallest physical one is ~1e-2 of the largest.
CURVILINEAR_PINV_RCOND = 1e-8

# Derivative couplings (`gradients.gwbse.derivative_coupling`): two roots closer
# than this (Hartree) have no well-defined pair of eigenvectors, and the
# amplitude term divides by their gap. Refused, as `DIABAT_GAP_MIN` refuses a
# diabat with a neighbour this close.
COUPLING_GAP_MIN = 1e-4

# Vibronic and rate models (`properties.vibronic`, `rates`, `spin_vibronic`):
# every normal mode they read must be a vibration at least this many cm^-1.
# An imaginary mode is a saddle, which has no Franck-Condon density of
# states; a mode below the floor is a rigid-body remnant or a flat direction,
# whose Huang-Rhys factor g^2 / (2 omega^3) has no bound.
VIBRONIC_MODE_FLOOR_CM = 1.0

# The per-root Newton, its seeds and its count check (`eeADC.ee_fold`)
# Hartree: seeds spread by at most this are one level
EE_FOLD_DEGENERATE = 1e-8
# tight seeds: Ritz error |r|^2/gap far below EE_FOLD_DEGENERATE
EE_FOLD_SEED_RESIDUAL = 1e-9
# seeds of M: they only place the seed frequency
EE_FOLD_M_SEED_RESIDUAL = 1e-3
# seeds of A_eff at the seed frequency
EE_FOLD_AEFF_SEED_RESIDUAL = 1e-4
# Hartree: seeds closer than this are re-solved tight
EE_FOLD_SEED_SPLIT = 1e-5
# loosest residual of a Newton step's eigensolve
EE_FOLD_INNER_CAP = 1e-3
# its residual: this times the previous |lambda - omega|
EE_FOLD_INNER_RATIO = 0.1
# Hartree: the count check's correction r / (d - e + shift)
EE_FOLD_CHECK_SHIFT = 0.1
# the count check decides mu against a window, no tighter
EE_FOLD_CHECK_RESIDUAL = 1e-4
# unit vectors on the lowest lifted diagonal beside the random one
EE_FOLD_CHECK_SEEDS = 8
# relative norm left off a span below which a start vector is in it
EE_FOLD_SPAN = 1e-8
# full-vector overlap above which two roots are one
EE_FOLD_DUPLICATE = 0.5
# rounds of the count check before it warns
EE_FOLD_COUNT_ROUNDS = 3
# seconds: an eigensolve reports at most this often meanwhile
EE_FOLD_HEARTBEAT = 600.0
# Hartree: a root this far from the centre rebuilds the images
EE_FOLD_RECENTRE = 0.01
# roots past nroots the Davidson may follow and seeds whole
EE_FOLD_FOLLOW = 2

# The tiled nuclear force of a density-fitted correlated energy
# (`gradients.df_correlation_force_tiled`): GB of one unit of the
# derivative-integral pass (an AO shell block against an auxiliary chunk, a
# few (n_aux_chunk, nao, n_block) arrays), and the auxiliary functions per
# chunk of that pass and of the JK fit's exchange intermediates. Both fix the
# summation order, so neither is ever set by the rank count.
DF_FORCE_BLOCK_GB = 1.0
DF_FORCE_AUX_CHUNK = 256
# Basis functions above which the ADC(2)/MP2 state models take the tiled force
# instead of the dense reference (every MO factor and doubles tensor whole).
EE_FORCE_DENSE_MAX_NAO = 150
# Residual (Hartree) to which a root is converged before its force: the force
# is first order in the singles vector's error, the energy second.
EE_FORCE_ROOT_RESIDUAL = 1e-11

# Self-consistent solvation loop (src.environments.solvation_loop)
# largest change of an induced charge between cycles (e)
SOLVATION_LOOP_TOL_Q = 1e-7
# change of the free energy G between cycles (Hartree)
SOLVATION_LOOP_TOL_ENERGY = 1e-8
# cycle cap of one loop
SOLVATION_LOOP_MAX_CYCLE = 30
# vectors kept by the DIIS extrapolation of the charges
SOLVATION_LOOP_DIIS_SPACE = 8
# a walk's G_eq at its minimum and the vertical's at the same geometry, each
# on its own frozen conventions, differing by more than this many loop
# tol_energy is flagged in the photophysics record (walk_vs_vertical_flag)
WALK_VERTICAL_G_FLAG_FACTOR = 10
# Loosest orbital gradient (pyscf conv_tol_grad) a state model's SCF may be
# converged to (src.gradients.state_models): the relaxed density and the
# Hellmann-Feynman force assume stationary orbitals, and both are first order
# in the orbital gradient -- the one ceiling SCF_GRAD_TOL. pyscf's default,
# sqrt(conv_tol), is refused.
STATE_MODEL_SCF_GRAD_TOL = SCF_GRAD_TOL
# Largest max |dm - dm^T| / max(1, max |dm|) of an AO density whose PCM force
# is taken (src.environments.pcm_derivatives.potential_integral_gradient, which
# differentiates the first AO index only and doubles it): a symmetric density
# built in floating point stays some 1e-15 off; a transition density that
# was never symmetrized is off at order 1.
PCM_FORCE_DENSITY_SYMMETRY_TOL = 1e-10
# Norm of the difference below which a PCM object's cached density (and its
# charges) is reused for a force (pcm_derivatives.potential_integral_gradient)
PCM_DM_CACHE_TOL = 1e-10

# Folded ADC(2) nuclear force (src.gradients.adc.ee_gradient)
# eigenvalue tolerance (Hartree) of the root whose force is taken
EE_FORCE_ROOT_TOL = 1e-12
# A state model's force re-solves its root at EE_FORCE_ROOT_TOL; at the call's
# own field its omega must equal the call's to this many times the call's
# conv_tol, or the force is another state's (ee_state_model, follow='index').
# Water and formaldehyde 6-31G, roots 0-2: at most 1.9e-10 Ha apart at 1e-8.
EE_FORCE_OMEGA_MATCH = 10.0

# Tiled correlation force (src.gradients.df_correlation_force_tiled)
# (kt, nv, no, nv) arrays alive in one MP2 tile: I, D, tau, lambda, a product
DF_FORCE_MP2_TILE_ARRAYS = 5
# ... and in one tile of ee_relaxed_density.tiled_pieces
DF_FORCE_PIECES_TILE_ARRAYS = 14
# (chunk, nao, block) arrays alive per unit of an AO shell block
DF_FORCE_BLOCK_ARRAYS = 5

# pdaggerq prints term coefficients rounded to seven decimals (1/576 -> 0.0017361);
# the exact rational is recovered within half a unit of the last printed digit
# (src/Base/pdaggerq_interface.exact_coefficient).
PDAGGERQ_COEFFICIENT_PRINT_TOLERANCE = 5.5e-8

# Seed of the random directions solve_symmetric's lowest-k completion adds to
# the start of its solve on the complement of the converged roots.
DAVIDSON_COMPLETION_SEED = 0

# Root following (Solvers.davidson.overlap_pick): two Ritz vectors whose
# overlaps on the followed reference differ by less than this are a tie,
# broken by energy (the roots beyond the reference's rank have overlap ~0).
DAVIDSON_FOLLOW_OVERLAP_TIE = 1e-6

# Generated charged-ADC solves (SingleReference.ADC.sigma_driver): the exact
# k=2 downfold's outer omega iterations, the residual cap of its inner
# Davidson, and the floor on |omega - K2| that keeps a Schur pole off the
# current omega (Ha; the guard keeps the pole's side, warns and flags).
ADC_DOWNFOLD_MAX_OUTER = 50
ADC_DOWNFOLD_DAVIDSON_TOL_CAP = 1e-8
ADC_SCHUR_POLE_GUARD = 1e-10

# Largest raw block of the non-Dyson intermediate-state representation, in
# matrix elements, that `isr_operator.OperatorISR` holds as an array (applied by
# BLAS instead of the generated applies, once per order); classes whose
# diagonal block fits also get their effective states precomputed. 2**27
# elements is 1 GiB of float64: at CO2/aug-cc-pVDZ the 1h and 2h1p classes fit,
# the 3h2p class and its couplings to 2h1p do not.
ISR_DENSE_BLOCK_ELEMENTS = 2 ** 27

# The quasi-boson Z-vector solve (`gradients.gwbse.targets.solve_Z`): its conjugate
# gradient stops at |r| <= QB_Z_SOLVE_TOL max(1, |rhs|) within QB_Z_MAX_ITER
# iterations and is accepted while |mv(z) - rhs| / max(1, |rhs|) stays below
# QB_Z_RESIDUAL_REL; two eigenvalues closer than QB_Z_DEGENERATE_REL times
# their scale are one in its divided differences, and a diagonal below
# QB_Z_PRECONDITIONER_FLOOR preconditions with 1.
QB_Z_SOLVE_TOL = 1e-10
QB_Z_MAX_ITER = 4000
QB_Z_RESIDUAL_REL = 1e-6
QB_Z_DEGENERATE_REL = 1e-12
QB_Z_PRECONDITIONER_FLOOR = 1e-10

# The fragment-diabatic BSE (`fragment_bse`): the cycle cap of its Davidsons,
# the iteration cap of its batched conjugate gradient, the Newton steps of its
# pole guard, and the Feshbach oracle's root tolerance (Ha) and step cap.
FRAGMENT_DAVIDSON_MAX_CYCLE = 500
FRAGMENT_CG_MAX_ITER = 2000
FRAGMENT_POLE_NEWTON_STEPS = 30
FESHBACH_ROOT_TOL = 1e-13
FESHBACH_ROOT_MAX_ITER = 100

# The one floor of a finite-difference step ladder's ratio of successive
# differences (Richardson), below which the ratio is not formed.
FD_LADDER_RATIO_FLOOR = 1e-12

# Every residue route of the contour quasiparticle energy; each consumer takes
# a named subset of this one vocabulary.
RESIDUE_ROUTES_ALL = ('explicit', 'laplace', 'sop', 'none', 'auto')

# The spin-coupling factors of a spin-orbit element: with |S_ia>, |T^M_ia> the
# orthonormal spin-adapted singles and h the coefficient of sigma,
# <S_I|H_SO,z|T_J,0> = VV - OO and <0|H_SO,z|T_J,0> = sqrt(2) sum h_ia X_ia.
# They play the role sqrt(2) plays in `oscillator_strengths`: the spin
# adaptation's own factor, in the same spatial-orbital normalization. Halving
# them is the sigma/s confusion, and it is invisible to every ratio,
# invariance and route-agreement test.
EXCITED_SPIN_FACTOR = 1.0
GROUND_SPIN_FACTOR = math.sqrt(2.0)

# Davidson guess: two orbital-energy differences closer than this (Ha) count
# as one degenerate set, which the guess then takes whole. pyscf's tdscf uses
# the same value and the same window (`deg_eia_thresh`); the splittings it has
# to absorb are the DF and SCF noise on a true degeneracy, ~1e-5 Ha on benzene.
DAVIDSON_GUESS_DEG_TOL = 1e-3
# Ha. The smallest |d - omega| a Davidson correction or Jacobi-Davidson
# preconditioner divides by: a pair on the Ritz value is not divided by zero.
DAVIDSON_DENOMINATOR_FLOOR = 1e-8

# Polarizable sites: Thole's exponential-damping factor (dimensionless), and
# the separation (Bohr) below which two sites are coincident, which no damping
# can rescue -- a site list with a genuine duplicate is an input error.
THOLE_FACTOR = 2.5874
MIN_SITE_SEPARATION = 1e-6

# Bohr. The box of the single-descent ISDF radius search, kept so every cached
# and shipped grid keeps its key (`separable_ri.ELEMENT_R_MAX`).
LEGACY_R_MAX = 5.0

# Laplace quadrature points of the fused rank>=3 amplitudes in the MP3/MP4
# densities (DensityMatrix drivers, ADC static corrections).
MPN_DENSITY_LAPLACE_NTAU = 6

# Excitation-energy ADC (eeADC): channel sizes up to which a solve is a dense
# diagonalization instead of a Davidson. Below EE_ADC_DENSE_LIMIT, Davidson's
# subspace goes linearly dependent long before it converges (a 59-dimensional
# open-shell sector did); every route honours it. With the fold, the singles
# channel up to EE_FOLD_DENSE_LIMIT builds A_eff densely: n products per Newton
# step against a Davidson's dozen, so only where Davidson would run out of space.
EE_ADC_DENSE_LIMIT = 2000
EE_FOLD_DENSE_LIMIT = 64
# (no, nj, nv, nv) slab blocks one slab of the restricted DF ADC(2) coupling
# kernel holds at once (ee_r_sigma_df; the traced peak of
# tests/test_ee_adc_fold.py), what slab_plan budgets the slab width with.
EE_COUPLING_SLAB_BLOCKS = 11
