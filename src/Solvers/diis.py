# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""DIIS acceleration for CC codes."""
import numpy as np
from itertools import product


class DIIS:

    def __init__(self, num_diis_vecs: int, start_iter=4):
        """num_diis_vecs: DIIS subspace size. start_iter: iterations before DIIS kicks in."""
        self.nvecs = num_diis_vecs
        self.error_vecs = []
        self.prev_vecs = []
        self.start_iter = start_iter
        self.iter_idx = 0

    def compute_new_vec(self, iterate, error):
        """Compute a DIIS update, once start_iter iterates have accumulated."""
        if self.iter_idx < self.start_iter:
            self.iter_idx += 1
            return iterate

        self.prev_vecs.append(iterate)
        self.error_vecs.append(error)
        self.iter_idx += 1

        if len(self.prev_vecs) > self.nvecs:
            self.prev_vecs.pop(0)
            self.error_vecs.pop(0)

        b_mat, rhs = self.get_bmat()
        c = np.linalg.solve(b_mat, rhs)

        new_iterate = np.zeros_like(self.prev_vecs[0])
        for ii in range(len(self.prev_vecs)):
            new_iterate += c[ii] * self.prev_vecs[ii]
        return new_iterate

    def get_bmat(self):
        """Build the DIIS B-matrix/RHS for the Ax=b problem."""
        dim = len(self.prev_vecs)
        b = np.zeros((dim, dim))
        for i, j in product(range(dim), repeat=2):
            if i <= j:
                b[i, j] = self.edot(self.error_vecs[i], self.error_vecs[j])
                b[j, i] = b[i, j]
        b = np.hstack((b, -1 * np.ones((dim, 1))))
        b = np.vstack((b, -1 * np.ones((1, dim + 1))))
        b[-1, -1] = 0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1, 0] = -1
        return b, rhs

    def edot(self, e1, e2):
        """Dot product of two error vectors/matrices."""
        if len(e1.shape) == 1 and len(e2.shape) == 1:
            return e1.dot(e2)
        elif e1.shape[1] == 1 and e2.shape[1] == 1:
            return e1.T.dot(e2)
        elif len(e1.shape) == 2 and len(e2.shape) == 2 and e1.shape == e2.shape:
            return np.einsum('ij,ij', e1, e2)  # Tr[e1.T @ e2]
        else:
            raise TypeError("Can't take dot of this type of error vec")
