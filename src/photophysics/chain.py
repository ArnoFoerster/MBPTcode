"""The production BSE@GW chain of a `Declaration`, built in one place.

Every stage, the vertical arms and the acceptance check build the chain's
`SurfaceSpec`, its Kohn-Sham mean field and its continuum here, so two of
them cannot declare different surfaces for the same declaration.

THE CHAIN: an ISDF-K mean field on the declared mean-field grid (or J and K
density-fitted on <basis>-jkfit), E_0 = E_KS, space-time GW with the
sum-over-poles residues on the declared quasiparticle set and the frozen
scissor outside it, the ISDF Davidson and the grid adjoint of each root, on
the sliced factors and the row-distributed fit in the gas phase and in a
continuum alike. The factors sit on `numerics.isdf.level`.

The chain stands on the declaration's one `Numerics`, which `chain_spec`
hands to `SurfaceSpec.numerics` whole; `potential_energy_surface` refuses a
declared choice the chain does not read.
"""
import warnings

from pyscf import dft
from pyscf import df as pyscf_df

from src.Base.basis.ri_fallback import paired_auxbasis
from src.Base.numerics.constants import SCF_NEWTON_FINISH_FROM, SOLVENT_RI_RESIDUAL_MAX
from src.Base.declaration import GroundState
from src.Base.geometry.geometry import elements_of
from src.Base.factorization.isdf_jk import canonical_span_size, span_confined
from src.gradients.isdf_mean_field import isdf_jk_with_force
from src.Base.pyscf.pyscf_interface import occupied_count
from src.Base.pyscf.scf_convergence import SCFHistory, newton_finished, warm_started
from src.Base.factorization.separable_ri import grid_identity, resolve_isdf_grid
from src.environments.solvent_screening import SolventScreening
from src.Base.utils.parallel.mpi_grid import n_ranks
from src.Base.utils.parallel.threads import blas_single_threaded
from src.properties.excitations import SurfaceSpec

# -------------------------------------------------------------- molecules
def factor_auxbasis(mol):
    """The RI set of the GW and BSE factors (`paired_auxbasis`)."""
    return paired_auxbasis(mol.basis)


def scf_auxbasis(mol, route):
    """The auxiliary set of the mean field's J and K: <basis>-jkfit when it
    is density-fitted (route 'df'), since -ri fits only the
    occupied-virtual products of MP2; the factors' -ri set for ISDF-K and for
    the DF twin of an ISDF-K surface."""
    return f'{mol.basis}-jkfit' if route == 'df' else factor_auxbasis(mol)


def grid_content(level, mol):
    """What the ISDF grid `level` is at `mol`: each element's shell counts
    and the sha256 of the shipped radii rows they select (`grid_identity`
    without the level, which a stamp carries beside it). A level is a name,
    and the table it names has been re-cut under it, so the content is
    stamped too."""
    identity = grid_identity(level, str(mol.basis), elements_of(mol),
                             auxbasis=factor_auxbasis(mol))
    return {'level': level, 'counts': identity['counts'],
            'radii_sha256': identity['radii_sha256']}


# ------------------------------------------------------------- continuum
def continuum(declaration, mol):
    """The continuum of `declaration` around `mol` (a solvent of pyscf's SMD
    table, or 'EPS:EPS_STATIC'), or None in the gas phase.

    The optical constant screens the response, the static one relaxes the
    ground state: the non-equilibrium recipe of Duchemin, Guido, Jacquemin and
    Blase, Chem. Sci. 9, 4430 (2018).
    """
    named = declaration.environment
    if declaration.gas:
        return None
    if ':' in named:
        eps, eps_static = (float(x) for x in named.split(':', 1))
        return SolventScreening(mol, eps=eps, eps_static=eps_static)
    return SolventScreening(mol, solvent=named)


def solvated(env, factory):
    """`factory` in the ground-state reaction field of `env` (PCM at
    eps_static), or `factory` itself in the gas phase."""
    if env is None:
        return factory
    return lambda mol: env.mean_field(mol, factory)


# ------------------------------------------------------------------ span
def span_choice(declaration, mol):
    """The span every SCF of the run is solved in, counted once at the input
    geometry `mol`: {'cut', 'n_kept', 'nao'}, cut and count None for the full
    basis. `numerics.span.cut` and its count of overlap eigenvectors at `mol`
    (`canonical_span_size`; the full basis where nothing lies below it). No
    span is chosen by refusal: a molecule whose orbital densities the factors
    miss in the declared span is refused in the run itself."""
    cut = declaration.numerics.span.cut
    kept = None if cut is None else canonical_span_size(mol, cut)
    return {'cut': None if kept is None else cut, 'n_kept': kept,
            'nao': int(mol.nao_nr())}


# ----------------------------------------------------------------- chain
def chain_spec(declaration, environment):
    """The production chain of `declaration` in `environment` as one
    `SurfaceSpec`: E_0 the declared ground state on the reference functional,
    space-time GW with the declared residues (`numerics.qp.residue_route`) on
    the declared quasiparticle set, the ISDF Davidson and its adjoint, the
    factors on the declared level, all on the declaration's one
    `Numerics`."""
    numerics = declaration.numerics
    return SurfaceSpec(GroundState(declaration.ground, declaration.reference),
                       environment=environment, chi0='space-time',
                       residues=numerics.qp.residue_route, solver='davidson',
                       factorization='isdf', qp_states=declaration.qp_states,
                       numerics=numerics)


# ------------------------------------------------------------ mean field
def distributed_here():
    """Whether this call runs inside the launch's distribution (and not in an
    `mpi_map` item, where each rank converges its own SCF)."""
    return n_ranks() > 1


class SCFNotConverged(RuntimeWarning):
    """A mean field of the run whose SCF stopped short of convergence: the
    stage that built it carries 'scf_not_converged' in its record."""


def mean_field_factory(reference, numerics, max_memory, histories, finish,
                       warm_start, span, surface_route):
    """An SCF factory mol -> mf: `reference` Kohn-Sham on the XC grid
    `numerics.scf.xc_grid`, its exchange ISDF-K
    on `numerics.scf.grid_level` (route 'isdf') or J and K density-fitted
    (route 'df'), converged to `numerics.scf` (conv_tol, conv_tol_grad,
    max_cycle); built and NOT run under ranks, where the surface converges it
    over them.

    conv_tol_grad is not decoration: a gradient Lagrangian assumes the
    occupied-virtual Fock block vanishes, so what is left of it biases every
    force through the orbital response.

    surface_route: the route of the run's own mean field, which names the
    fitting set of a density-fitted build (`scf_auxbasis`): the DF twin of an
    ISDF-K surface fits on the factors' -ri set; omitted is the route.
    histories: a list each mean field's `SCFHistory` is appended to; one
    whose SCF did not converge is warned about (`SCFNotConverged`) and its
    history marked `not_converged`, and the stage counts it. finish
    'newton' ends with second-order steps (`newton_finished`); warm_start a
    `WarmStart` whose density starts each SCF. span: (n_kept, cut), solve
    every SCF in the span of the n_kept largest overlap eigenvectors counted
    at that cut (`span_confined`); n_kept None is the full basis.
    """
    scf = numerics.scf
    route = scf.route

    def build(mol):
        mf = dft.RKS(mol, xc=reference)
        mf.grids.level = scf.xc_grid
        mf.max_memory = max_memory
        if route == 'df':
            mf = mf.density_fit(
                auxbasis=scf_auxbasis(mol, surface_route or route))
        else:
            counts, n_start = resolve_isdf_grid(scf.grid_level, str(mol.basis),
                                                elements_of(mol),
                                                auxbasis=factor_auxbasis(mol))
            mf = isdf_jk_with_force(mf, auxbasis=factor_auxbasis(mol), counts=counts,
                         n_start=n_start)
        mf.conv_tol, mf.conv_tol_grad = scf.conv_tol, scf.conv_tol_grad
        mf.max_cycle = scf.max_cycle
        mf = span_confined(mf, *span)
        mf.max_memory = max_memory
        history = SCFHistory()
        history.attach(mf)
        histories.append(history)
        if finish == 'newton':
            mf = newton_finished(mf, finish_from=SCF_NEWTON_FINISH_FROM)
        if warm_start is not None:
            mf = warm_started(mf, warm_start)
        if distributed_here():
            return mf
        with blas_single_threaded():
            mf.kernel()
        if not mf.converged:
            # the policy: refuse inputs, warn numerics; the
            # stage that built it counts it ('scf_not_converged')
            history.not_converged = True
            warnings.warn(f'SCF did not converge in {mf.max_cycle} cycles; '
                          f'the run carries on and the stage record is '
                          f'flagged', SCFNotConverged, stacklevel=2)
        return mf

    return build


def span_fields(manifold, mol, choice):
    """The span the surface's mean field at `mol` is solved in, how it was
    counted (`span_choice`), the largest residual of its
    factors on an orbital density, and in a continuum the
    residual of the reaction field's auxiliary representation on the cavity
    (`SolventScreening.require_fit`, which refuses above
    SOLVENT_RI_RESIDUAL_MAX)."""
    chain_ = manifold.driven
    mf = chain_.mean_field(mol)[1]
    span = dict(getattr(mf, 'span', None) or
                {'cut': None, 'n_kept': int(mf.mo_coeff.shape[-1]),
                 'n_dropped': int(mol.nao_nr() - mf.mo_coeff.shape[-1]),
                 'nao': int(mol.nao_nr())})
    span['max_orbital_residual'] = chain_.measured_orbital_residual(mf)
    env = chain_.environment_at(mol)
    fit = None
    if hasattr(env, 'require_fit'):
        auxmol = pyscf_df.addons.make_auxmol(mol, factor_auxbasis(mol))
        fit = {'auxbasis': factor_auxbasis(mol),
               'residual': env.require_fit(mol, auxmol, mf.mo_coeff,
                                           occupied_count(mf)),
               'bar': SOLVENT_RI_RESIDUAL_MAX}
    return {'span': span, 'span_choice': choice, 'reaction_field_fit': fit}


def factor_layout(manifold):
    """The layout the evaluation's chain ran on: sliced factors or whole,
    the fit's realization and tile edge, and the environment that dressed the
    kernel's gauge."""
    chain_ = manifold.driven
    env = chain_.environment
    return {'sliced': bool(chain_.sliced), 'fit': chain_.fit,
            'fit_block': chain_.fit_block,
            'environment': None if env is None else type(env).__name__}
