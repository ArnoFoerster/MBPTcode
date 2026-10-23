"""What the Casida and the Tamm-Dancoff Davidson loops share: guess pairs, trial-space sizing, the preconditioner diagonal and the corrections."""
import warnings

import numpy as np
from pyscf.lib import param
from pyscf.tdscf._lr_eig import MAX_SPACE_INC

from src.Base.numerics.constants import (BYTES_PER_GB, DAVIDSON_DENOMINATOR_FLOOR,
                                         DAVIDSON_EXTRA_ROOTS_DIVISOR, DAVIDSON_GUESS_DEG_TOL,
                                         DAVIDSON_EXTRA_ROOTS_FLOOR,
                                         DAVIDSON_GUESS_FACTOR,
                                         DAVIDSON_PRECONDITIONERS,
                                         DAVIDSON_SPACE_CYCLES, DAVIDSON_SPACE_FRACTION,
                                         DAVIDSON_SPACE_GB)




def guess_indices(diag_d, nroots, guess_factor=DAVIDSON_GUESS_FACTOR,
                   deg_tol=DAVIDSON_GUESS_DEG_TOL, counters=None):
    """Occupied-virtual pairs carrying the unit-vector Davidson guess: more of
    them than `nroots`, and never half a degenerate set -- see
    DAVIDSON_GUESS_FACTOR for
    what one guess per root costs. A guess cut inside a degenerate cluster
    is warned about and named in `counters['guess_truncated']`, which the
    solve's `timings` carry as 'davidson_guess_truncated'
    (refuse inputs, warn numerics).
    """
    d = diag_d.ravel()
    n_pair = d.size
    order = np.argsort(d)
    # The +4 floor is for small nroots, where guess_factor alone leaves too
    # few directions to span anything -- 2 vectors at nroots=1.
    n = min(max(guess_factor * nroots, nroots + 4), n_pair)
    # The members of a degenerate set are the pairs that sit in
    # different irreps, so a cut through one is what loses whole symmetry
    # blocks; extend past the cut for as long as the gaps stay degenerate.
    cut = d[order[n - 1]] + deg_tol
    # real_eig sizes its trial-space holders at max(4 * nroots, 2 * space_inc)
    # in the worst (memory-starved) case and a guess wider than that overruns
    # them, so stop extending there -- space_inc is at least min(20, n_pair//2).
    n_max = min(max(4 * nroots, 20, n), n_pair)
    while n < n_max and d[order[n]] <= cut:
        n += 1
    if n == n_max < n_pair and d[order[n]] <= cut:
        if counters is not None:
            counters['guess_truncated'] = True
        warnings.warn(
            f'Davidson guess truncated at {n} vectors inside a cluster of '
            f'orbital-energy differences degenerate to {deg_tol:g} Ha, so the '
            'cut may split a degenerate set and lose the symmetry blocks its '
            'other members would have opened. Ask for fewer roots, or raise '
            'guess_factor and check the roots stop moving.',
            RuntimeWarning, stacklevel=4)
    return order[:n]


def real_eig_space(max_memory, nroots, n_pair):
    """(space_inc, max_space): the corrections real_eig adds per cycle and the
    trial pairs it holds before collapsing, by its own rule on `max_memory` MB.
    """
    space_inc = (nroots if MAX_SPACE_INC is None
                 else max(nroots, min(MAX_SPACE_INC, n_pair // 2)))
    max_space = int(max_memory * 1e6 / 8 / (4 * n_pair) / 2 - space_inc)
    if max_space < nroots * 4 < n_pair:
        max_space = space_inc * 2
    return space_inc, min(max(max_space, nroots * 4), n_pair)


def trial_space_memory(nroots, n_pair, rows=None, max_memory=None,
                        working_bytes=0):
    """MB handed to real_eig, whose own rule (`real_eig_space`) then gives the
    trial pairs the Davidson holds before it restarts: pyscf's own MAX_MEMORY,
    raised until the space holds DAVIDSON_SPACE_CYCLES increments within this
    rank's budget.

    The budget is per rank. The four holders V, W, U1, U2 take 32 bytes per
    pair row per trial pair, and a rank holds only its `rows` (the whole pair
    space serially). real_eig's rule gives them half of max_memory, less one
    increment for the cycle's Ritz vectors and residuals; here that half,
    DAVIDSON_SPACE_FRACTION of this rank's `max_memory` MB, must also hold the
    block action's `working_bytes` (`action_working_bytes`), which run beside
    the holders. The space is held whole wherever the allocation has room for
    it (on a large system once enough ranks share its pair rows), where
    DAVIDSON_SPACE_CYCLES binds first; a space that collapses every few
    cycles makes the iteration path depend on the rank count.

    max_memory None, no allocation to size against: the whole holders within
    DAVIDSON_SPACE_GB, the bound for a run outside SLURM.
    """
    space_inc = real_eig_space(param.MAX_MEMORY, nroots, n_pair)[0]
    if max_memory is None:
        fits = int(DAVIDSON_SPACE_GB * BYTES_PER_GB / (32 * n_pair))
    else:
        budget = DAVIDSON_SPACE_FRACTION * max_memory * 1e6 - working_bytes
        fits = int(budget / (32 * (n_pair if rows is None else max(rows, 1)))
                   - space_inc)
    target = min(DAVIDSON_SPACE_CYCLES * space_inc, fits)
    # the inverse of real_eig's rule, half a pair clear of its floor
    return max(param.MAX_MEMORY,
               (target + space_inc + 0.5) * 2 * 32 * n_pair / 1e6)


def action_working_bytes(apply_AB, batch, n_pair, stacked):
    """Bytes the block action holds and allocates beside the trial space
    while it applies a `batch` of trial pairs: what it keeps (`held_bytes`),
    one vector's scratch (`scratch_bytes`) and, per pair, one application's
    own buffers (`vector_bytes`, A z and B z alone for an action that does
    not say), the other half's A z and B z held meanwhile, and the (k, 2N)
    batch with the copy it is stacked or gathered into; `stacked` adds the
    serial `vind`'s stacked products U1 and U2 and their halves."""
    per_pair = (getattr(apply_AB, 'vector_bytes', 16 * n_pair)
                + 16 * n_pair + 32 * n_pair + (32 * n_pair if stacked else 0))
    return (getattr(apply_AB, 'held_bytes', 0)
            + getattr(apply_AB, 'scratch_bytes', 0) + batch * per_pair)


def check_preconditioner(preconditioner):
    """A ValueError unless `preconditioner` is one of DAVIDSON_PRECONDITIONERS,
    raised at an entry point before anything is built."""
    if preconditioner not in DAVIDSON_PRECONDITIONERS:
        raise ValueError(f'preconditioner={preconditioner!r}; choose one of '
                         f"{', '.join(map(repr, DAVIDSON_PRECONDITIONERS))}.")


def preconditioner_diagonal(apply_AB, diag_d, preconditioner):
    """(n_occ, n_vir): the diagonal the Davidson's corrections divide by.

    'bare' is the pair energies d, the operator's own d z term; 'screened'
    is d - (ii|W|aa), which the action forms from its own factors
    (`screened_diagonal`). The screened direct term is the part of diag(A)
    the kernel moves most -- on average far more than the Hartree 2(ia|ia)
    -- so it moves the corrections' denominators where the bare d is
    furthest off. `preconditioner` is checked at the entry points
    (`check_preconditioner`).
    """
    if preconditioner == 'bare':
        return diag_d
    screened = getattr(apply_AB, 'screened_diagonal', None)
    if screened is None:
        raise ValueError("preconditioner='screened' needs a block action "
                         'that forms its screened diagonal (isdf_block_action '
                         'or df_block_action); this one does not.')
    return screened()


def tracked_roots(nroots, n_pair, extra=None):
    """Ritz pairs the Hermitian Davidson iterates on for `nroots` roots:
    nroots + max(DAVIDSON_EXTRA_ROOTS_FLOOR, nroots //
    DAVIDSON_EXTRA_ROOTS_DIVISOR), or nroots + `extra` where a caller names
    its own, at most the pair space."""
    if extra is None:
        extra = max(DAVIDSON_EXTRA_ROOTS_FLOOR,
                    nroots // DAVIDSON_EXTRA_ROOTS_DIVISOR)
    return min(nroots + int(extra), n_pair)


def correction(dx, hdiag, e):
    """The raw Davidson corrections r / (d - omega) of the residual rows `dx`
    (len(e), pairs) against the rows `hdiag` of the pair diagonal, a
    vanishing denominator floored at DAVIDSON_DENOMINATOR_FLOOR."""
    d = hdiag[None, :] - e[:, None]
    d[np.abs(d) < DAVIDSON_DENOMINATOR_FLOOR] = DAVIDSON_DENOMINATOR_FLOOR
    return dx.reshape(len(e), -1) / d
