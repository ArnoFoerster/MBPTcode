"""Runs of consecutive basis shells of bounded width.

Every blocked integral pass in this tree cuts a basis (orbital or auxiliary)
into runs of consecutive shells holding at most `width` functions each, one
shell where a single shell holds more. The runs are fixed by the basis and
the width, never by the rank count, so a pass divided over ranks by runs
sums the same blocks at every rank count.
"""


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
