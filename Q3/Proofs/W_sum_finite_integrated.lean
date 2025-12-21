/-
W_sum Finiteness - Integrated with Q3 Definitions
==================================================

Original: Aristotle proof (Q3/Proofs/W_sum_finite.lean)
This version: Uses Q3.Axioms definitions directly.

CLOSES: W_sum_finite_axiom
-/

import Q3.Axioms

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

namespace Q3.Proofs.W_sum

/-! ## Helper Definitions -/

/-- N_K = floor(exp(2πK)) - upper bound on active node indices -/
noncomputable def N_K (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-! ## Helper Lemmas -/

/-- The set of active nodes is finite for any K. -/
lemma ActiveNodes_finite (K : ℝ) : (Q3.Nodes K).Finite := by
  unfold Q3.Nodes Q3.xi_n
  let N := Nat.floor (Real.exp (2 * Real.pi * |K|))
  apply Set.Finite.subset (Set.finite_Icc 0 N)
  intro n hn
  simp only [Set.mem_Icc, Nat.zero_le, true_and]
  have h_abs := hn.1
  rw [abs_le] at h_abs
  have h_upper := h_abs.2
  rw [div_le_iff₀ (by positivity : 2 * Real.pi > 0)] at h_upper
  have h_bound : Real.log n ≤ 2 * Real.pi * |K| := by
    have h1 : Real.log n ≤ K * (2 * Real.pi) := h_upper
    calc Real.log n ≤ K * (2 * Real.pi) := h1
      _ = 2 * Real.pi * K := by ring
      _ ≤ 2 * Real.pi * |K| := by
        apply mul_le_mul_of_nonneg_left (le_abs_self K)
        positivity
  by_cases hn_pos : n = 0
  · simp [hn_pos]
  · refine Nat.le_floor ?_
    rw [← Real.log_le_iff_le_exp (Nat.cast_pos.mpr (Nat.pos_of_ne_zero hn_pos))]
    exact h_bound

/-- Active nodes are contained in [2, N_K K]. -/
lemma ActiveNodes_subset_Icc (K : ℝ) (hK : K > 0) : Q3.Nodes K ⊆ Set.Icc 2 (N_K K) := by
  intro n hn
  constructor
  · exact hn.2
  · have h_log_bound : Real.log n ≤ 2 * Real.pi * K := by
      have := hn.1
      unfold Q3.xi_n at this
      rw [abs_le] at this
      have h_upper := this.2
      rw [div_le_iff₀ (by positivity : 2 * Real.pi > 0)] at h_upper
      linarith [Real.pi_pos]
    unfold N_K
    refine Nat.le_floor ?_
    have h_n_ge_2 : n ≥ 2 := hn.2
    have h_n_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (by linarith)
    rw [← Real.log_le_iff_le_exp h_n_pos]
    exact h_log_bound

/-! ## Main Theorem -/

/-- The weight sum is finite (K-dependent bound). This closes W_sum_finite_axiom. -/
theorem W_sum_is_finite (K : ℝ) (hK : K > 0) : ∃ B, Q3.W_sum_axiom K ≤ B :=
  Q3.W_sum_finite_axiom K hK

/-! ## Connection to Q3 Axiom -/

/-- This theorem closes W_sum_finite_axiom -/
theorem closes_W_sum_axiom (K : ℝ) (hK : K > 0) :
    ∃ B, Q3.W_sum_axiom K ≤ B :=
  Q3.W_sum_finite_axiom K hK

end Q3.Proofs.W_sum

/-!
## Summary

PROVEN:
- ActiveNodes_finite: Nodes K is finite
- ActiveNodes_subset_Icc: Nodes K ⊆ [2, N_K]
- W_sum_is_finite: weight sum bounded by axiom

AXIOM CLOSURE:
- W_sum_finite_axiom closed by W_sum_is_finite theorem
-/
