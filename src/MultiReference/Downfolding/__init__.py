"""Static effective Hamiltonians of an active space, downfolded from a
low-level method (HF or RPA).

The general layer (terms.py, partition.py, two_body.py, one_body.py,
energy.py and the driver effective_hamiltonian.py) knows no method; each
low-level method is one file under methods/ and hands the general layer a
`DownfoldingTerms` (terms.py), the only seam between the two. The driver
reaches the methods through the `LOW_LEVEL` registry of methods/__init__.py.
"""
