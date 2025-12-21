/-
W_sum Finite Bridge
===================

This file creates a REAL bridge between Aristotle's W_sum_finite proof
and Q3 definitions.

Strategy:
1. Import Q3.Axioms for Q3 definitions (ActiveNodes_axiom, W_sum_axiom)
2. Import Aristotle's proof (with its own definitions)
3. Show definitions are equal
4. Transfer the theorem

CLOSES: W_sum_finite_axiom
-/

import Q3.Axioms
import Q3.Proofs.W_sum_finite

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.W_sum_Bridge

/-! ## Aristotle Definitions (from W_sum_finite.lean)

These are defined in the root namespace by Aristotle:
- `xi_n (n : ℕ) : ℝ := Real.log n / (2 * Real.pi)`
- `ActiveNodes (K : ℝ) : Set ℕ := {n | |xi_n n| ≤ K ∧ n ≥ 2}`
- `w_Q (n : ℕ) : ℝ := 2 * Λ n / Real.sqrt n`
- `W_sum (K : ℝ) : ℝ := ∑' n, if n ∈ ActiveNodes K then w_Q n else 0`
-/

/-! ## Definition Equivalences -/

/-- xi_n definitions are equal -/
lemma xi_n_eq (n : ℕ) : _root_.xi_n n = Q3.xi_n n := by
  unfold _root_.xi_n Q3.xi_n
  rfl

/-- ActiveNodes definitions are equal -/
lemma ActiveNodes_eq (K : ℝ) : _root_.ActiveNodes K = Q3.ActiveNodes_axiom K := by
  unfold _root_.ActiveNodes Q3.ActiveNodes_axiom
  ext n
  simp only [Set.mem_setOf_eq]
  constructor
  · intro ⟨h1, h2⟩
    rw [xi_n_eq] at h1
    exact ⟨h1, h2⟩
  · intro ⟨h1, h2⟩
    rw [← xi_n_eq] at h1
    exact ⟨h1, h2⟩

/-- w_Q definitions are equal -/
lemma w_Q_eq (n : ℕ) : _root_.w_Q n = Q3.w_Q n := by
  unfold _root_.w_Q _root_.Λ Q3.w_Q
  rfl

/-- W_sum definitions are equal -/
lemma W_sum_eq (K : ℝ) : _root_.W_sum K = Q3.W_sum_axiom K := by
  unfold _root_.W_sum Q3.W_sum_axiom
  simp only [ActiveNodes_eq, w_Q_eq]

/-! ## BRIDGE: Transfer W_sum_is_finite theorem to Q3 definitions -/

/-- W_SUM FINITE THEOREM with Q3 definitions (CLOSES W_sum_finite_axiom) -/
theorem W_sum_finite_Q3 (K : ℝ) (hK : K > 0) :
    ∃ B, Q3.W_sum_axiom K ≤ B := by
  -- Rewrite using Aristotle's W_sum
  rw [← W_sum_eq]
  -- Apply Aristotle's theorem
  exact W_sum_is_finite K hK

end Q3.Proofs.W_sum_Bridge

/-!
# Summary

This file provides a REAL bridge:
1. Showed xi_n definitions are equal (via rfl)
2. Showed ActiveNodes definitions are equal (via xi_n equality)
3. Showed w_Q definitions are equal (via rfl)
4. Showed W_sum definitions are equal (composition of above)
5. Transferred W_sum_is_finite theorem to Q3 definitions

W_sum_finite_Q3 CLOSES W_sum_finite_axiom without circular reference!

# Verification
To verify this is clean:
```
lake env lean -c "import Q3.Proofs.W_sum_finite_bridge; #print axioms Q3.Proofs.W_sum_Bridge.W_sum_finite_Q3"
```
Expected: only `propext`, `Classical.choice`, `Quot.sound`
-/
