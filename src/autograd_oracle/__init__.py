"""torch.autograd oracles for the ISDF / contour-deformation GW-BSE gradients.

Two independent evaluations of production's surface, differentiated by
autograd and compared with the hand-written adjoints entry by entry:

    gate_factors    the many-body chain from the factors (eps, X_mo, D_dressed,
                    D_bare) to Omega or eps^QP -- `torch_chain`
    gate_nuclear    the whole force from the nuclear coordinates: a torch
                    integral engine (`torch_integrals`), pyscf's PCM cavity, the
                    DF-RHF fixed point, the ISDF fit (`torch_nuclear`), then the
                    same chain
    torch_oracle, torch_nuclear_oracle
                    the torch halves, run as separate processes
    frozen          the discrete choices a surface is differentiated on, and
                    the .npz the two processes exchange
    pin_integrals   the integral engine against pyscf at one geometry

TWO PROCESSES, ONE TORCH THREAD. The torch half must import without pyscf.
On a workstation's x86_64 torch under Rosetta the backward pass
segfaults with two intra-op threads and a process that ran pyscf's integrals
and torch deadlocks at exit, so the production side never imports torch and
the torch side never imports pyscf. This `__init__` imports nothing for the
same reason.
"""
