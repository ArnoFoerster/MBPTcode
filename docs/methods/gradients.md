# Analytic nuclear gradients

**Analytic nuclear gradients** — `src/gradients/` differentiates production's
own forward objects (`src.SingleReference`, `src.Base`) rather than a second
copy of them: the Lagrangian of Toelle,
[arXiv:2412.17085](https://arxiv.org/abs/2412.17085), and Toelle, Kitsaras and
Loos, [arXiv:2507.02160](https://arxiv.org/abs/2507.02160), with the papers'
iterative BCH/truncated-Taylor machinery replaced by exact closed forms.
`RPAGroundStateChain` carries the cubic-scaling dRPA ground-state gradient and
`ExcitedStateChain` the cubic-scaling BSE@GW one, both on a frozen ISDF
factorization; `DenseRPASurface` and `DenseBSESurface` carry the same two
gradients at O(N^6), on the dense quasi-boson layer.
Every continuation ([GW](gw.md): Pade, contour deformation, sum-over-poles)
and every environment ([Environments](environments.md): PCM, dispersion,
polarizable sites) has its own adjoint,
so a quasiparticle or excitation gradient is exact for whichever route and
surroundings computed the energy, not a finite difference of a different one.
One orbital-response (Z-vector) solve is shared by every energy target a
chain carries. See `examples/16_numerical_hessian_from_gradient.py` for a
property built directly from the gradient: a vibrational analysis by central
differences of the analytic force, exact against pyscf's own analytic Hessian
where that exists, and the only route where it does not (an ISDF mean field).
