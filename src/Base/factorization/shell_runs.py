"""Runs of consecutive basis shells of bounded width.

Every blocked integral pass in this tree cuts a basis (orbital or auxiliary)
into runs of consecutive shells holding at most `width` functions each, one
shell where a single shell holds more. The runs are fixed by the basis and
the width, never by the rank count, so a pass divided over ranks by runs
sums the same blocks at every rank count.

A derivative pass ends per function and is folded onto the atoms the
functions sit on: `functions_to_atoms` (each atom's run of functions summed)
or, for a block of functions anywhere in the basis, `function_atoms` (the
atom of every function).
"""
import numpy as np


def shell_runs(ao_loc, width):
    """(s0, s1) runs of consecutive shells of at most `width` functions each
    (`ao_loc` the shells' offsets), at least one shell per run, so a shell
    wider than `width` is still a run of its own. A byte budget is a width:
    max_bytes // per-function bytes."""
    out, s0 = [], 0
    nsh = len(ao_loc) - 1
    width = int(width)
    while s0 < nsh:
        s1 = s0 + 1
        # int(): ao_loc is int32, and a block's bytes are not
        while s1 < nsh and int(ao_loc[s1 + 1]) - int(ao_loc[s0]) <= width:
            s1 += 1
        out.append((s0, s1))
        s0 = s1
    return out


def shell_ranges(ao_loc, width):
    """(s0, s1, p0, p1): the `shell_runs` with their function ranges."""
    return [(s0, s1, int(ao_loc[s0]), int(ao_loc[s1]))
            for s0, s1 in shell_runs(ao_loc, width)]


def function_atoms(mol):
    """(nao,) the atom each basis function of `mol` sits on."""
    out = np.empty(mol.nao_nr(), dtype=int)
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[p0:p1] = ia
    return out


def functions_to_atoms(t, mol):
    """(natm, 3) of a per-function derivative t (3, nao of `mol`): each atom's
    run of functions summed, in function order."""
    out = np.zeros((mol.natm, 3))
    for ia, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[ia] = t[:, p0:p1].sum(axis=1)
    return out
