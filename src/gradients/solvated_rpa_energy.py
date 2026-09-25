"""The reverse pass of the solvated dRPA fold, and the diagnostic that sweeps it.

The functional itself -- the block fold of the ACFDT log-determinant over a
non-overlapping solvent, its reaction-field matrix N and its solute-solvent
dispersion term -- is
`SingleReference.LinearResponse.solvated_rpa_energy`, and every name this
module used to define resolves to it. `dispersion_energy` is kept as an alias
of `solute_solvent_dispersion_energy`, the name that tells it apart from the
empirical D3/D4 correction of `Base.dispersion`.

THE FOLD HAS A FORCE. `rpa_energy_and_adjoint` carries the adjoints of the
rescaled interaction and the bare counter-term as `fold_bar`, and
`fold_bar_to_gauge` below turns that into adjoints on the dressed metric root
and on vtilde, which `FactorChain.nuclear_gradient` routes to the nuclei
alongside the D factor's own.

`solvated_correlation_energies` stays here as well: it is a DIAGNOSTIC that
drives `rpa_energy_and_adjoint` four times -- once per member of the family
and once for the traces -- so it is a caller of the reverse-pass module rather
than a piece of the functional. A production surface asks for the fold through
`RPAGroundStateChain(fold=True)`, which sweeps once and differentiates.
"""
import numpy as np

from src.Base.separable_ri import aux_metric_sqrt
from src.Base.solvent_screening import (dynamic_screening_factor,  # noqa: F401
                                        solvent_plasmon_energy)
from src.SingleReference.LinearResponse.solvated_rpa_energy import (  # noqa: F401
    SolvatedCorrelation, fold_terms, screening_matrix,
    solute_solvent_dispersion_energy)
from src.gradients.space_time_adjoint import (rpa_energy_and_adjoint,
                                              rpa_frequency_traces)

#: The solute-solvent term under the name this module bound before the
#: functional moved; `Base.dispersion.dispersion_energy` is the unrelated
#: empirical D3/D4 correction.
dispersion_energy = solute_solvent_dispersion_energy


def fold_bar_to_gauge(auxmol, environment, fold_bar, V=None):
    """(root_bar, kernel_bar): the fold's N adjoint as adjoints on the DRESSED
    metric root R = (V + vtilde)^(1/2) and on vtilde itself.

    `rpa_energy_and_adjoint` returns dE/dN and dE/dC separately because it does
    not know how they were chosen; the fold chooses both out of one N, with
    S_w = I - (1 - g_w) N and C = I - N, so

        N_bar = fold_bar.n_bar - fold_bar.counter_bar

    and the two enter with OPPOSITE SIGNS, the logarithm's screening reduction
    against the counter-term's dispersion. From N = R^-1 vtilde R^-1,

        kernel_bar = R^-1 N_bar R^-1
        root_bar   = -(R^-1 N_bar N + N N_bar R^-1)

    both symmetrized: N, vtilde and the metric are symmetric, so only the
    symmetric part of an adjoint is a derivative of anything.

    TWO DIFFERENT DESTINATIONS, which is why they are returned apart.
    `kernel_bar` is an adjoint on vtilde ALONE and goes only to the
    environment's `aux_kernel_adjoint`; `root_bar` is an adjoint on the root,
    and below it V and vtilde are one matrix, so it goes through
    `sqrtm_adjoint` onto V + vtilde and reaches the two-centre integrals as
    well. It must be SUMMED with the D factor's own root adjoint before that
    square root rather than differentiated on its own -- the same R, one
    Frechet solve, one cavity derivative (`FactorChain.nuclear_gradient`).
    """
    if fold_bar is None:
        return None, None                    # before the kernel is built
    n_mat = screening_matrix(auxmol, environment, V=V)
    if n_mat is None:
        return None, None
    n_bar = np.asarray(fold_bar.n_bar, float)
    if fold_bar.counter_bar is not None:
        n_bar = n_bar - fold_bar.counter_bar
    n_bar = 0.5 * (n_bar + n_bar.T)
    root = aux_metric_sqrt(auxmol, environment, V=V)
    # R^-1 N_bar and N_bar R^-1 by solves, never an explicit inverse
    left = np.linalg.solve(root, n_bar)                     # R^-1 N_bar
    kernel_bar = np.linalg.solve(root, left.T).T            # R^-1 N_bar R^-1
    root_bar = -(left @ n_mat + n_mat @ left.T)
    return 0.5 * (root_bar + root_bar.T), 0.5 * (kernel_bar + kernel_bar.T)


def solvated_correlation_energies(chain, mol=None, mf=None, omega_p=None,
                                  rule='f_sum'):
    """(SolvatedCorrelation) A, B, F and the split of F - A at one geometry.

    chain: any `FactorChain` carrying a frequency grid and an environment that
    dresses the interaction -- `RPAGroundStateChain` is the one this is about.
    omega_p in Hartree; None looks the solvent up in `SOLVENT_PLASMON_EV` under
    `rule`. The polarizability is swept four times, once per energy and once
    for the traces, which is four times the cost of one `correlation_energy`.
    """
    mol, mf = chain.mean_field(mol, mf)
    x_mo, d, eps, auxmol, _, _ = chain.factors_at(mol, mf)
    environment = chain.environment_at(mol)
    n_mat = screening_matrix(auxmol, environment)
    if n_mat is None:
        raise ValueError(
            f'{environment!r} dresses no interaction, so there is no fold to '
            'take: A, B and F are one and the same gas-phase energy')
    if omega_p is None:
        omega_p = solvent_plasmon_energy(getattr(environment, 'solvent', None),
                                         rule)

    grid = chain.grid
    tile = ({} if getattr(chain, 'tile_gb', None) is None
            else {'tile_gb': chain.tile_gb})
    counter = np.eye(n_mat.shape[0]) - n_mat
    g = dynamic_screening_factor(grid.omega_points, omega_p)

    def energy(**kwargs):
        return float(rpa_energy_and_adjoint(x_mo, d, eps, chain.nocc, grid,
                                            want_grad=False, **kwargs, **tile))

    dressed = energy()
    bare = energy(screening=(n_mat, np.zeros(grid.nfreq)), counter_term=counter)
    fold = energy(screening=(n_mat, g), counter_term=counter)
    traces = rpa_frequency_traces(x_mo, d, eps, chain.nocc, grid, n_mat, **tile)
    dispersion = solute_solvent_dispersion_energy(traces, grid, omega_p)
    return SolvatedCorrelation(bare=bare, dressed=dressed, fold=fold,
                               dispersion=dispersion,
                               beyond_first_order=fold - bare - dispersion,
                               omega_p=omega_p, traces=traces,
                               omega=grid.omega_points,
                               weights=grid.omega_weights)
