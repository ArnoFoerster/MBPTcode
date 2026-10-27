# Diagram-primary ADC generator

Derives ADC the way Schirmer/Cederbaum/Walter 1983 (PRA 28, 1237) did — from the
Goldstone-diagram expansion of the dynamic self-energy M(ω) — instead of the
ISR route. Motivation: the ISR-ADC(4) hierarchy carries an intrinsic
t3-gated λ⁴ representation deficiency whose
missing content requires explicit energy-denominator structure the ISR
cannot produce; the diagram expansion contains it natively.

## Architecture (diagram-primary)

The central data structure is the time-ordered Goldstone diagram = typed
skeleton (graph; h/p-typed internal lines) + a vertex linear order.

- `goldstone.py` — the engine.
  1. pdaggerq fixed-end vertex streams per external o/v block (bare `g`,
     0.25ⁿ; self-loop diagrams dropped = exact W-cancellation for a
     canonical-HF reference). Each merged pdaggerq term is one typed
     skeleton and certifies numerator, sign, and weight (all
     ordering-independent).
  2. Goldstone time orderings = linear extensions of the h/p precedence DAG
     (particle line: creator earlier; hole line: annihilator earlier).
     Per-ordering weight = coeff / n_extensions.
  3. Cut denominators by construction. Complete value rule (zero free
     parameters):
         value = (coeff/n_ext) · σ_ket · ∏_cuts 1/(dir·ω + Σε_h − Σε_p),
     σ_ket = +1/−1 for a virtual/occupied ket external (the hole-line parity
     of the amputated ket line). dir = ±1 for cuts separating the external
     attachments (sign = which side the bra is on), 0 otherwise.
  4. Graph filters: connectivity; static class (both externals on one
     vertex → Σ_∞, handled by the MPn-density machinery elsewhere);
     one-particle-reducible (a single line separating the attachment
     vertices → belongs to the Dyson iteration, not M).
- `exact_oracle.py` — the committed arbiter: determinant-space (Slater–
  Condon) exact Σ_dyn(ω,λ) for H(λ)=F+λ(H−F), pointwise λ-Taylor
  coefficients at complex ω. Self-contained on purpose.
- `emit.py` — emits generated numpy modules (one term per Goldstone
  diagram) into `src/SingleReference/ADC/generatedADC/diagrammatic/`.

## Validation

- n=2: engine == textbook closed form, both sectors, H2+H4, ≤1e-14.
  (H4 is mandatory: H2's vv/sector-I and oo/sector-II entries vanish by
  parity — H2 alone is blind to half the sign table.)
- n=3: census == SCW83 Fig. 3 (6 M^I + 6 M^II per block); engine == exact
  λ³ at the oracle fit floor (1e-10 H2, 3e-8 H4). The engine's C-class ==
  the determinant-basis U1/C1 reference at 5e-18 and its U-class == the
  hand-coded SCW83 (44b)/(45) U⁽²⁾ at 2.6e-18 — i.e. the engine re-derives
  the published third-order ADC objects diagram by diagram. The (45b) sign
  C^II = −C̄⁽¹⁾ is automatic in the value rule above.
- n=4: engine == exact λ⁴ at 1.2e-11 (H2) and ≤3.7e-8 (linear H4 = the
  t3≠0 system where ISR-ADC(4) is off by 8.4e-4 — four orders of
  magnitude). 7-line (4p3h) cuts present and evaluated per ordering; the
  SCW83 Appendix-A cancellations are only needed for the ADC-FORM
  extraction, not for pointwise M.

Gates: `tests/test_diagram_goldstone.py`, `tests/test_diagram_emitted.py`.

## ADC-form extraction state (`adc_form.py`)

- n=3 identities: engine C-class == U1·R·C1·R·U1 and U-class == SCW83
  (44b)/(45) U⁽²⁾, machine precision (committed gates).
- **ADC(4)-form closure on H2: EXACT** (overdetermined ω-sampled fit; U³/C²
  extracted, residual ~1e-17). H2 has no 3p2h space, so this is the complete
  statement there — and H2 is where the PRINTED SCW83 B3/B7 fail the
  λ-oracle; the machine-extracted objects supersede them.
- H4 (3p2h space active): the pure-k1 model does NOT close (out-of-sample
  ~5e-3), and two candidate picture-derivations of the 3p2h couplings were
  falsified the same way (as at n=3: only diagram-extraction
  yields correct couplings, never wave-function-picture shortcuts).

## Quartet class (`quartet_pf.py`)

- **Quartet class resummed in closed form** (`quartet_pf.py`): every
  seven-line-cut skeleton carries exactly the four App-A orderings
  ([3w 7w 3w], [3w 7w 4f], [4f 7w 3w], [4f 7w 4f]) sharing one middle
  bipartition, so Em = Ea + E_f = Eb + E_f′, the 4p3h-pole residues cancel
  PER INDEX TUPLE ((1−1−1+1)/(E_f E_f′) = 0), and the group sums to

      G(u) = (u + Em) / ((u + Ea)(u + Eb) · E_f · E_f′),   u = dir·ω

  — manifestly regular (no dangerous Eb−Ea denominators; degenerate tuples
  force E_f = E_f′ and are included automatically). Verified against the
  engine's filtered ordering sum at 3e-18; η-bounded at 4p3h pole
  positions.

The quartet class is emitted as `q4_blocks.py` (`emit.emit_quartet_order4`),
the δU3 corrections by `emit.emit_delta_u3_order4` and
`emit.emit_delta_u3_laplace_order4`. The δ = diagram − ISR comparison against
the generated ADC(4) supermatrix needs the generated-ADC runtime, which is not
part of this package yet.
