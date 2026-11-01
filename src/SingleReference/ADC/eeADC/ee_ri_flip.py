"""The Ms = +-1 component of a closed shell's triplets in spin-component
scaled ADC(2), on the density-fitted fold of ee_ri_fold.

Spin-component scaling weights a doubles configuration by the spin class of
its occupied pair (c_os opposite, c_ss same spin), which is not invariant
under spin rotations: the three components of a triplet stop being
degenerate. ee_ri_fold's triplet channel is the Ms = 0 component. The Ms = +1
component has the singles x_ia of i_beta -> a_alpha and two classes of
doubles, (k_beta l_beta -> c_alpha d_beta), a same-spin pair, and
(k_alpha l_beta -> c_alpha d_alpha), an opposite-spin pair; Ms = -1 is its
mirror image. With H the doubles' pair tensor of ee_ri_fold, H' its occupied
swap (H'_kcld = H_lckd = (H^T)~_kcld) and H~ its virtual swap,

    A = H - H'   (same-spin pair)      B = H - H~   (opposite-spin pair)
    R = w [c_ss A + c_os B],   |Y|^2 = sum w^2 [c_ss A^2 + c_os B^2] / 2

and the second-order singles block keeps the triplet's form with the
amplitude T = -c_os tau~ (only the opposite-spin pair (j_beta k_alpha ->
b_alpha c_beta) couples there), x_oo and x_vv the singlet's. At c_os = c_ss
every piece is the Ms = 0 triplet's, and so is every root; the test holds
the roots against the spin-orbital supermatrix's spin-flip roots.

The operator is the triplet channel's with these pieces replaced (CPU only):
products, fold images and doubles overlaps of the same shapes, the same
tiles and ranks, so ee_fold.solve_folded and ee_ri_fold.drive solve it as
they solve the Ms = 0 channel.
"""
import numpy as np

from src.Base.utils.parallel.mpi_grid import ordered_sum
from src.SingleReference.ADC.eeADC.ee_ri_fold import RIFoldADC2, SpinScaling


class FlipScaling(SpinScaling):
    """SpinScaling of the Ms = +-1 channel: the second-order amplitude of the
    singles block is -c_os tau~ (module docstring)."""

    def triplet(self, G, Gx):
        return -self.cos * Gx


class RIFoldSpinFlip(RIFoldADC2):
    """A_eff(w) of the Ms = +-1 triplets of closed-shell SCS-ADC(2) (module
    docstring); the arguments are RIFoldADC2's, the spin the triplet's."""

    def __init__(self, factors, tile_gb=None, threads=None, comm=None, aux_chunk=None,
                 numerics=None, scaling=None):
        kw = {} if tile_gb is None else {'tile_gb': tile_gb}
        super().__init__(factors, 'triplet', level='adc2', threads=threads, comm=comm,
                         aux_chunk=aux_chunk, numerics=numerics, scaling=scaling, **kw)
        if self.gpu is not None:
            raise ValueError('the Ms = +-1 channel runs on the CPU only')
        self.ms = 1

    def _setup(self):
        self.weights = FlipScaling(self.scs)
        super()._setup()

    def _pieces(self, BtT, k0, k1):
        """(A, B) of a tile (module docstring)."""
        H, HT = self.pair(BtT, k0, k1)
        A = H - HT.transpose(0, 3, 2, 1)
        B = H - H.transpose(0, 3, 2, 1)
        return A, B

    def _gt(self, A, B):
        return self.weights.css * A + self.weights.cos * B

    def _doubles_tile(self, k0, k1, BtT, omega):
        nv, L = self.nv, self.f.L
        kt = k1 - k0
        A, B = self._pieces(BtT, k0, k1)
        w = 1.0 / (omega - self._denom_tile(k0, k1))
        R = w * self._gt(A, B)
        nrm = float(np.sum(w * w * (self.weights.css * A * A
                                    + self.weights.cos * B * B))) / 2.0
        del A, B
        Gam = R.reshape(kt * nv, -1) @ L
        del R
        return Gam, self.back_oo(Gam, k0, k1), nrm

    def _images_tile(self, k0, k1, BtT, nodes, derivative):
        nv, L = self.nv, self.f.L
        kt = k1 - k0
        A, B = self._pieces(BtT, k0, k1)
        Gt = self._gt(A, B)
        del A, B
        D = self._denom_tile(k0, k1)
        gams = np.empty((nodes.size, kt * nv, self.f.naux))
        for q, om in enumerate(nodes):
            x = om - D
            R = -Gt / (x * x) if derivative[q] else Gt / x
            gams[q] = R.reshape(kt * nv, -1) @ L
        return gams, self.back_oo(gams, k0, k1)

    def doubles_dot(self, ur, wr, us, ws):
        same = ur is us or np.array_equal(ur, us)
        Br = self.dressed(ur)
        Bs = Br if same else self.dressed(us)
        parts = []
        for t, (k0, k1) in self.my_tiles:
            Ar, Rr = self._pieces(Br, k0, k1)
            As, Rs = (Ar, Rr) if same else self._pieces(Bs, k0, k1)
            D = self._denom_tile(k0, k1)
            w2 = 1.0 / ((wr - D) * (ws - D))
            val = float(np.sum(w2 * (self.weights.css * Ar * As
                                     + self.weights.cos * Rr * Rs))) / 2.0
            parts.append((t, np.array([val])))
        return float(ordered_sum(parts, self.comm, shape=(1,), onto=np.zeros(1))[0])

    def _denom_tile(self, k0, k1):
        """D_kcld = e_c + e_d - e_k - e_l of the rows k0..k1-1."""
        eo, ev = self.f.eo, self.f.ev
        return (ev[None, :, None, None] + ev[None, None, None, :]
                - eo[k0:k1, None, None, None] - eo[None, None, :, None])
