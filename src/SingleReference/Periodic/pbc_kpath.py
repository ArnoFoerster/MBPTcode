"""Band paths and Fourier interpolation of k-dependent quantities.

A GW band structure is not computed on a path directly: the self-energy needs
a regular mesh (the q-sum runs over it), while a plot needs a dense line
through high-symmetry points. The standard resolution is to compute the
QUASIPARTICLE CORRECTION on the coarse mesh and interpolate THAT, then add it
to a mean-field band structure evaluated directly on the path -- which is
cheap, because a non-self-consistent diagonalization at arbitrary k needs no
q-sum.

Interpolating the correction rather than the band is what makes this work.
`QP(k) - eps(k)` is a smooth, slowly varying function of k even where the bands
themselves cross, disperse steeply, or are degenerate; the structure lives in
eps(k), which is evaluated exactly.

Why plain Fourier interpolation and not Shankland-Koelling-Wood
---------------------------------------------------------------
SKW augments this with a roughness functional and star functions built from
the SPACE GROUP. That would be a trap here: the operation set for anything
defined on a k-mesh must be the subgroup leaving the MESH invariant, not the
space group -- an n x n x 1 slab mesh keeps far fewer operations than its
lattice, and using the lattice's set is silently wrong rather than an error
(measured elsewhere in this tree: a diamond 2x2x1 ERI error that plateaus at
1.7e-2 instead of converging to 3.8e-7). Plain Fourier interpolation over the
Born-von Karman lattice needs no operations at all, is exact at the mesh
points by construction, and cannot acquire that class of bug. The cost is
possible ringing between mesh points, which `damping` controls.
"""
import numpy as np

# Fractional coordinates of the conventional high-symmetry points, per Bravais
# lattice, with the paths ordinarily plotted. Fractional in the RECIPROCAL
# primitive basis, which is what `cell.get_scaled_kpts` returns.
HIGH_SYMMETRY = {
    'fcc': {'G': (0, 0, 0), 'X': (0.5, 0, 0.5), 'W': (0.5, 0.25, 0.75),
            'K': (0.375, 0.375, 0.75), 'L': (0.5, 0.5, 0.5),
            'U': (0.625, 0.25, 0.625)},
    'bcc': {'G': (0, 0, 0), 'H': (0.5, -0.5, 0.5), 'P': (0.25, 0.25, 0.25),
            'N': (0, 0, 0.5)},
    'sc':  {'G': (0, 0, 0), 'X': (0, 0.5, 0), 'M': (0.5, 0.5, 0),
            'R': (0.5, 0.5, 0.5)},
    'hex2d': {'G': (0, 0, 0), 'M': (0.5, 0, 0), 'K': (1 / 3, 1 / 3, 0)},
    'sq2d': {'G': (0, 0, 0), 'X': (0.5, 0, 0), 'M': (0.5, 0.5, 0)},
}

DEFAULT_PATH = {
    'fcc': 'G X W K G L U W L K',
    'bcc': 'G H N G P H',
    'sc': 'G X M G R X',
    'hex2d': 'G M K G',
    'sq2d': 'G X M G',
}


def band_path(cell, lattice='fcc', path=None, npoints=200):
    """(kpts_cart, x, tick_x, tick_labels) for a high-symmetry path.

    `path` is a space-separated string of labels from HIGH_SYMMETRY[lattice];
    None takes DEFAULT_PATH. `x` is cumulative distance in the RECIPROCAL
    metric, so segment lengths are physical and the plot is not distorted by
    unequal sampling -- points are distributed along each segment in proportion
    to its true length, which a naive equal-points-per-segment split gets wrong
    whenever the segments differ in length.
    """
    pts = HIGH_SYMMETRY[lattice]
    labels = (path or DEFAULT_PATH[lattice]).split()
    unknown = [l for l in labels if l not in pts]
    if unknown:
        raise ValueError(f"unknown high-symmetry labels {unknown} for "
                         f"lattice {lattice!r}; have {sorted(pts)}")
    frac = np.array([pts[l] for l in labels], dtype=float)
    b = cell.reciprocal_vectors()                      # rows are b1, b2, b3
    cart = frac @ b
    seg = np.linalg.norm(np.diff(cart, axis=0), axis=1)
    total = seg.sum()
    if total <= 0:
        raise ValueError('the path has zero length')

    nseg = np.maximum(1, np.round(seg / total * (npoints - 1)).astype(int))
    kpts, xs, ticks = [], [], [0.0]
    x0 = 0.0
    for i, n in enumerate(nseg):
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        kpts.append(cart[i][None, :] + t[:, None] * (cart[i + 1] - cart[i]))
        xs.append(x0 + t * seg[i])
        x0 += seg[i]
        ticks.append(x0)
    kpts.append(cart[-1][None, :])
    xs.append(np.array([x0]))
    return (np.vstack(kpts), np.concatenate(xs), np.array(ticks), labels)


def bvk_vectors(cell, kmesh):
    """Born-von Karman lattice vectors, centred, one per mesh point.

    Centred rather than 0..N-1: the interpolant is a trigonometric polynomial
    in these vectors, and an asymmetric set biases it. For an even mesh the
    split is unavoidably lopsided by one and the Nyquist vector is kept on the
    positive side, matching numpy's fft convention.
    """
    rng = [np.arange(-((n - 1) // 2), n // 2 + 1) for n in kmesh]
    n1, n2, n3 = np.meshgrid(*rng, indexing='ij')
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1)
    return idx @ cell.lattice_vectors(), idx


def fourier_interpolate(cell, kmesh, kpts_mesh, values, kpts_target,
                        damping=0.0):
    """Interpolate a k-dependent quantity from a regular mesh onto any k.

    `values` is (nk,) or (nk, nband) sampled at `kpts_mesh` (Cartesian, in the
    order `cell.make_kpts(kmesh)` returns). Exact at the mesh points when
    `damping` is 0: the transform is square and invertible there, so this is
    interpolation and not a fit.

    `damping` > 0 multiplies each coefficient by exp(-damping (|R|/|R|max)^2),
    trading that exactness for smoothness between the mesh points. Use it only
    if the interpolant rings; the ringing is a genuine property of a
    band-limited reconstruction from few samples, not an artifact to be tuned
    away silently.

    NOTE ON BAND INDEX. Multi-band input is interpolated band by band, which
    presumes the columns of `values` are matched across k. That holds for a QP
    CORRECTION indexed by energy-ordered band, since the correction is smooth
    even where bands cross; it does NOT hold for the bands themselves.
    Interpolate corrections, add them to directly-evaluated eigenvalues.
    """
    kmesh = [int(n) for n in kmesh]
    nk = int(np.prod(kmesh))
    values = np.asarray(values)
    if len(kpts_mesh) != nk:
        raise ValueError(f"{len(kpts_mesh)} k-points for a mesh of {nk}")
    if values.shape[0] != nk:
        raise ValueError(f"values has {values.shape[0]} rows, expected {nk}")
    squeeze = values.ndim == 1
    v = values[:, None] if squeeze else values

    R, _ = bvk_vectors(cell, kmesh)                       # (nk, 3)
    # c_R = (1/nk) sum_k v(k) e^{-i k.R},  v(k') = sum_R c_R e^{+i k'.R}.
    # The two signs must be OPPOSITE. Taking a conjugate transpose here makes
    # both steps e^{+i k.R}, which is exact for a 2x2x2 mesh (phases are +-1,
    # real) and for any even function of k (a cosine band), and wrong for
    # everything else -- so it survives every test built from a smooth model
    # band and fails only on random data at N >= 3.
    phase = np.exp(-1j * (kpts_mesh @ R.T))               # (nk, nR)
    coeff = phase.T @ v / nk                              # (nR, nband)
    if damping > 0:
        rn = np.linalg.norm(R, axis=1)
        scale = np.exp(-damping * (rn / max(rn.max(), 1e-30)) ** 2)
        coeff = coeff * scale[:, None]

    out = np.exp(1j * (np.asarray(kpts_target) @ R.T)) @ coeff
    if not np.iscomplexobj(values):
        # Take the real part ALWAYS for real input, never conditionally.
        #
        # For an EVEN mesh the centred index set is unavoidably lopsided -- it
        # holds the Nyquist component at +N/2 with no -N/2 partner -- so the
        # raw sum is complex between mesh points even for real data. Measured
        # on bulk Al with random values: max|imag| = 0.84 at 2^3 and 1.27 at
        # 4^3, against 1.5e-16 at 3^3 where the set IS symmetric. An earlier
        # version returned real only when the residue fell under 1e-8, so on
        # every even mesh it silently returned a complex array.
        #
        # Re() is not a cast that discards information here, it is the correct
        # interpolant: for real data c_{-R} = conj(c_R), so
        #     Re sum_R c_R e^{ikR} = 1/2 [ sum_R c_R e^{ikR} + c.c. ]
        # which is the sum over the SYMMETRISED index set, splitting the
        # Nyquist component evenly between +-N/2. Exactness at the mesh points
        # survives because the interpolant is already real there.
        out = out.real
    return out[:, 0] if squeeze else out


def qp_bands_on_path(cell, mf, kmesh, kpts_mesh, corrections, bands,
                     lattice='fcc', path=None, npoints=200, damping=0.0):
    """Mean-field bands on a path, plus interpolated QP corrections.

    `corrections` is (nk, len(bands)) of QP(k,n) - eps(k,n) on the mesh.
    Returns a dict with the path, the mean-field bands, the interpolated
    corrections and their sum -- all in Hartree.
    """
    kpts, x, tick_x, labels = band_path(cell, lattice, path, npoints)
    mo_energy = mf.get_bands(kpts)[0]
    eps = np.array([np.asarray(e)[list(bands)].real for e in mo_energy])
    dq = fourier_interpolate(cell, kmesh, kpts_mesh, corrections, kpts,
                             damping=damping)
    dq = dq.reshape(len(kpts), -1)
    return dict(kpts=kpts, x=x, tick_x=tick_x, tick_labels=labels,
                bands=list(bands), eps=eps, correction=dq, qp=eps + dq)
