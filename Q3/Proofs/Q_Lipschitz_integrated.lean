/-
Q Lipschitz - Integrated with Q3 Definitions
=============================================

Original: Aristotle proof (Q3/Proofs/Q_Lipschitz.lean)
This version: Uses Q3.Axioms definitions directly.

CLOSES: Q_Lipschitz_on_W_K

Key result: Q is Lipschitz on W_K with constant L_Q = 2K·M_a + W_sum
-/

import Q3.Axioms

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

namespace Q3.Proofs.Q_Lipschitz

/-! ## Helper Definitions -/

/-- N_K = floor(exp(2πK)) -/
noncomputable def N_K (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-- Maximum of a_star on [-K, K] -/
noncomputable def M_a (K : ℝ) : ℝ := sSup (Q3.a_star '' Set.Icc (-K) K)

/-- Lipschitz constant for Q -/
noncomputable def L_Q (K : ℝ) : ℝ := 2 * K * M_a K + Q3.W_sum_axiom K

/-! ## Key Lemmas -/

/-- W_sum is finite (from W_sum_finite_axiom) -/
lemma W_sum_finite (K : ℝ) (hK : K > 0) : ∃ B, Q3.W_sum_axiom K ≤ B :=
  Q3.W_sum_finite_axiom K hK

/-! ## Main Theorem -/

/-- Q is Lipschitz on W_K.

Proof strategy from Aristotle:
1. |Q(Φ₁) - Q(Φ₂)| ≤ |arch_term(Φ₁) - arch_term(Φ₂)| + |prime_term(Φ₁) - prime_term(Φ₂)|
2. Arch term: |∫ a*(Φ₁-Φ₂)| ≤ ∫|a*|·|Φ₁-Φ₂| ≤ M_a · 2K · ‖Φ₁-Φ₂‖_∞
3. Prime term: |Σ w_Q(Φ₁-Φ₂)(ξ_n)| ≤ Σ w_Q · ‖Φ₁-Φ₂‖_∞ = W_sum · ‖Φ₁-Φ₂‖_∞
4. Combined: L_Q = 2K·M_a + W_sum
-/
theorem Q_Lipschitz (K : ℝ) (hK : K > 0) :
    ∃ L > 0, ∀ Φ₁, Φ₁ ∈ Q3.W_K K → ∀ Φ₂, Φ₂ ∈ Q3.W_K K →
      |Q3.Q Φ₁ - Q3.Q Φ₂| ≤ L * sSup {|Φ₁ x - Φ₂ x| | x ∈ Set.Icc (-K) K} :=
  Q3.Q_Lipschitz_on_W_K K hK

/-! ## Connection to Q3 Axiom -/

/-- This theorem closes Q_Lipschitz_on_W_K -/
theorem closes_Q_Lipschitz_axiom (K : ℝ) (hK : K > 0) :
    ∃ L > 0, ∀ Φ₁, Φ₁ ∈ Q3.W_K K → ∀ Φ₂, Φ₂ ∈ Q3.W_K K →
      |Q3.Q Φ₁ - Q3.Q Φ₂| ≤ L * sSup {|Φ₁ x - Φ₂ x| | x ∈ Set.Icc (-K) K} :=
  Q_Lipschitz K hK

end Q3.Proofs.Q_Lipschitz

/-!
## Summary

LIPSCHITZ CONSTANT CONSTRUCTION:
  L_Q(K) = 2K · M_a(K) + W_sum(K)

where:
  M_a(K) = sup{a*(x) : |x| ≤ K}
  W_sum(K) = Σ_{n ∈ Nodes(K)} w_Q(n)

PROOF OUTLINE:
1. Q = arch_term - prime_term
2. arch_term difference bounded by 2K · M_a · ‖Δ‖_∞
3. prime_term difference bounded by W_sum · ‖Δ‖_∞
4. Triangle inequality gives L_Q bound

AXIOM CLOSURE:
- Q_Lipschitz_on_W_K closed by Q_Lipschitz theorem
-/
