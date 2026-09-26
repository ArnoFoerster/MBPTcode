"""Periodic (k-point) infrastructure: RPA, GW and the vertex self-energies.

The k/q-point generalization of the molecular SingleReference/{LinearResponse,
GW} pipeline; everything here mirrors a molecular counterpart.

TWO FACTORIZATIONS, with different ceilings:

  * GDF / RI-V -- `pbc_integrals`, `pbc_rpa`, `pbc_self_energy`. Exact-ish and
    well validated, but materializes L for every k-pair, so memory grows as
    nkpts^2 and that is what caps the reachable mesh.
  * ISDF / THC -- `pbc_isdf`, `pbc_isdf_rpa`, `pbc_isdf_gw`. Separable in the
    k-index, so the same object is O(nkpts). This is the production route.

The two THC entry points are re-exported here because the conventions BETWEEN
their steps (which range fits which transform, which frequencies Pi must be
built on) are not guessable and should not be reassembled by callers.
"""
from src.SingleReference.Periodic.pbc_isdf_gw import qp_energy_thc
from src.SingleReference.Periodic.pbc_isdf_rpa import rpa_ecorr_thc

__all__ = ['rpa_ecorr_thc', 'qp_energy_thc']
