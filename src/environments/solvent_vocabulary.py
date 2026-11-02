"""The words every solvation entry point shares: how a state's solvation is
named, in one place.

Three vocabularies are spelled here once, so that their readers cannot drift:
the density a state-specific term is taken of, the scheme an environment is
built in, and the scheme a run declares. A run's scheme names a way of
solvating every state of it; the environment's scheme is the way one
excitation reaches its state-specific solvation.
"""

#: The medium-free setting a run can declare.
GAS = 'gas'

#: Which density difference the state-specific energy reads: the unrelaxed
#: amplitude density, the orbital-relaxed one (the ground state's charges
#: frozen), or the relaxed one with the continuum responding ('relaxed+pcm',
#: the ADC(2) routes only; the classical state-specific energy has no such
#: variant and refuses it).
SS_DENSITIES = ('unrelaxed', 'relaxed', 'relaxed+pcm')

#: The density every state-specific term takes unless a run names another:
#: the orbital-relaxed one, for the classical state-specific energy and the
#: EE-ADC state-specific shift alike.
SS_DENSITY_DEFAULT = 'relaxed'

#: The two ways an excitation reaches its state-specific solvation, by the
#: object that builds it (`environment.solvent_scheme`): the solvated BSE of
#: Duchemin et al. (`SplitEquilibrium`) and the classical state-specific
#: energy of scheme B (`ClassicalStateSpecific`).
CONSTRUCTION_SCHEMES = ('duchemin', 'classical-ss')

#: How the states of a run stand in the environment, each by name. 'gas': no
#: medium. 'scheme-a': the solvated BSE of Duchemin et al. (Chem. Sci. 9, 4430
#: (2018)): the ground state in PCM at eps_static, the response screened by
#: the continuum (non-equilibrium at R0), the relaxed states in the split or
#: non-equilibrium response. 'scheme-b-eq' and 'scheme-b-noneq': every state
#: self-consistent with its own orbital-relaxed density in the continuum
#: (the state-specific loop), its singlets' linear response at eps_inf, the
#: walks on G_eq; absorption and emission are the loop's non-equilibrium
#: energies. The two differ in the energy an excited state has at its
#: minimum: in its own equilibrium ('eq', the solvent relaxed around it) or in
#: the slow charges of S0's equilibrium there ('noneq', the solvent left as S0
#: had it, walked on that non-equilibrium surface). 'scheme-b-eq-bare' and
#: 'scheme-b-noneq-bare': the same without the singlets' linear response (the
#: BSE exchange on the bare interaction, the medium in the loop alone).
SOLVENT_SCHEMES = (GAS, 'scheme-a', 'scheme-b-eq', 'scheme-b-noneq',
                   'scheme-b-eq-bare', 'scheme-b-noneq-bare')

#: The schemes of the state-specific loop, and the relaxed states' solvation
#: each declares.
SCHEME_B = {'scheme-b-eq': 'equilibrium', 'scheme-b-noneq': 'nonequilibrium',
            'scheme-b-eq-bare': 'equilibrium',
            'scheme-b-noneq-bare': 'nonequilibrium'}

#: The scheme-B arms without the singlets' linear-response term.
SCHEME_B_BARE = ('scheme-b-eq-bare', 'scheme-b-noneq-bare')
