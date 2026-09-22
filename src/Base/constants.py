"""Numeric defaults and self-energy method registry shared across src/SingleReference/."""

# Physical constants, CODATA 2018. Defined here and nowhere else: import them,
# never re-spell the digits, so every route reports the same number.
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
BOHR_TO_ANGSTROM = 0.52917721092

# Lorentzian broadening for self-energy denominators / spectral functions.
DEFAULT_BROADENING_ETA = 1e-3

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
