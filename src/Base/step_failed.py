"""The one failure a geometry walk answers with a shorter step.

An evaluation at a trial geometry can fail because the step carried a solve
off the branch it was frozen on: the quasiparticle root finder loses its root, the
Casida problem loses positive-definiteness or its Davidson stops short of the
residual a force needs, or the followed state leaves the solved roots. Each is
a statement about the distance from the last accepted geometry, so the walk
rejects the step and retries it shorter. Every other exception -- a
NotImplementedError, a MemoryError, a ValueError of a setting, a bug -- is
not about the step and leaves the walk.
"""


class StepFailed(RuntimeError):
    """An evaluation that failed because the trial geometry is too far from
    the last accepted one; the only exception a walk catches."""


class DavidsonRefusal(StepFailed):
    """Roots a caller differentiates left above the residual it needs: their
    eigenvectors, and every derivative read off them, are wrong by it."""


class DavidsonNotConverged(DavidsonRefusal):
    """Roots a Davidson left above its residual tolerance: `roots` their
    indices, `residuals` their |r| recomputed from the returned vectors,
    `tol` the tolerance they missed."""

    def __init__(self, message, roots, residuals, tol):
        super().__init__(message)
        self.roots = [int(k) for k in roots]
        self.residuals = [float(r) for r in residuals]
        self.tol = float(tol)
