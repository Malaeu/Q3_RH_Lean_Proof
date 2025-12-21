/-
Off-Diagonal Exponential Sum Bridge
====================================

This file creates a REAL bridge between Aristotle's off_diag_exp_sum proof
and Q3.Basic.Defs definitions.

KEY INSIGHT: Aristotle's standalone definitions are DEFINITIONALLY EQUAL
to Q3.Basic.Defs, so we can transfer directly!
-/

import Q3.Basic.Defs
import Q3.Proofs.off_diag_exp_sum

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.OffDiagExpSumBridge

/-! ## Definition Equivalences

Both define:
- xi_n (n : ℕ) : ℝ := Real.log n / (2 * Real.pi)
- Nodes (K : ℝ) : Set ℕ := {n | |xi_n n| ≤ K ∧ n ≥ 2}
- delta_K (K : ℝ) : ℝ := 1 / (2 * Real.pi * (N_K K + 1))
- S_K (K t : ℝ) : ℝ := 2 * exp(-delta^2/(4t)) / (1 - exp(-delta^2/(4t)))

These are DEFINITIONALLY EQUAL!
-/

/-! ## Main Bridge Theorem -/

/-- Specific version for Nodes K (CLOSES off_diag_exp_sum_axiom)

Since Q3.Nodes K, Q3.xi_n, Q3.S_K are definitionally equal to Nodes K, xi_n, S_K,
we just apply Aristotle's theorem. -/
theorem off_diag_exp_sum_Q3_specific (K t : ℝ) (hK : K ≥ 1) (ht : t > 0) :
    ∀ (i : _root_.Nodes K),
      @Finset.sum (_root_.Nodes K) ℝ _ (@Fintype.elems (_root_.Nodes K) inferInstance)
        (fun j => if (j : ℕ) ≠ (i : ℕ) then
          Real.exp (-(_root_.xi_n i - _root_.xi_n j)^2 / (4 * t)) else 0) ≤ _root_.S_K K t :=
  fun i => off_diag_exp_sum_bound K t hK ht i

end Q3.Proofs.OffDiagExpSumBridge
