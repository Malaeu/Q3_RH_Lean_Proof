/-
W_sum Finite Bridge v3 (CLEAN - no Q3.Axioms)
=============================================

This file creates a CLEAN bridge for W_sum finiteness.
Uses only Q3.Basic.Defs (no Q3.Axioms).

CLOSES: W_sum_finite axiom without importing Q3.Axioms
-/

import Q3.Basic.Defs  -- ONLY Defs, no Axioms!

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.W_sum_BridgeV3

/-! ## Helper Lemmas -/

/-- Nodes is finite for any K -/
lemma Nodes_finite (K : ℝ) : (Q3.Nodes K).Finite := by
  unfold Q3.Nodes Q3.xi_n
  let N := Nat.floor (Real.exp (2 * Real.pi * |K|))
  apply Set.Finite.subset (Set.finite_Icc 0 N)
  intro n hn
  simp only [Set.mem_Icc, Nat.zero_le, true_and]
  cases abs_cases K <;> aesop
  · rw [abs_of_nonneg (div_nonneg (Real.log_nonneg (by norm_cast; linarith)) (by positivity))] at left
    exact Nat.le_floor <| by rw [abs_of_nonneg h]; rw [div_le_iff₀ <| by positivity] at *; rw [← Real.log_le_iff_le_exp <| by positivity] at *; linarith
  · linarith [abs_le.mp left]

/-- Each w_Q n is bounded for n in Nodes -/
lemma w_Q_bound (K : ℝ) (hK : K > 0) (n : ℕ) (hn : n ∈ Q3.Nodes K) :
    Q3.w_Q n ≤ 2 * Real.sqrt 2 * Real.pi * K := by
  unfold Q3.Nodes at hn
  have h_lambda_le_log : ∀ n, ArithmeticFunction.vonMangoldt n ≤ Real.log n :=
    fun n => ArithmeticFunction.vonMangoldt_le_log
  have h_log_le_2piK : Real.log n ≤ 2 * Real.pi * K := by
    unfold Q3.xi_n at hn; aesop
    nlinarith [abs_le.mp left, Real.pi_pos, mul_div_cancel₀ (Real.log n) (by positivity : (2 * Real.pi) ≠ 0)]
  have h_sqrt_ge_sqrt2 : Real.sqrt n ≥ Real.sqrt 2 :=
    Real.sqrt_le_sqrt <| mod_cast hn.2
  have h_w_Q_le : Q3.w_Q n ≤ 2 * (2 * Real.pi * K) / Real.sqrt 2 := by
    unfold Q3.w_Q
    exact div_le_div₀ (by positivity) (by linarith [h_lambda_le_log n]) (by positivity) (by linarith)
  exact h_w_Q_le.trans_eq (by rw [div_eq_iff (by positivity)]; ring_nf; norm_num; ring_nf)

/-! ## Main Theorem -/

/-- W_sum is finite - direct existence proof -/
theorem W_sum_finite_Q3 (K : ℝ) (hK : K > 0) : ∃ B, Q3.W_sum K ≤ B := by
  -- The sum is over a finite set and each term is bounded
  have h_finite : (Q3.Nodes K).Finite := Nodes_finite K
  -- Convert tsum to finite sum
  have h_eq : Q3.W_sum K = ∑ n ∈ h_finite.toFinset, Q3.w_Q n := by
    unfold Q3.W_sum
    have h_summable : Summable (fun n => if n ∈ Q3.Nodes K then Q3.w_Q n else 0) := by
      apply summable_of_ne_finset_zero (s := h_finite.toFinset)
      intro n hn
      have : n ∉ Q3.Nodes K := fun h_in => hn (h_finite.mem_toFinset.mpr h_in)
      simp only [this, ite_false]
    rw [tsum_eq_sum (f := fun n => if n ∈ Q3.Nodes K then Q3.w_Q n else 0) (s := h_finite.toFinset)]
    · apply Finset.sum_congr rfl
      intro n hn
      have hn' : n ∈ Q3.Nodes K := h_finite.mem_toFinset.mp hn
      simp only [hn', ite_true]
    · intro n hn
      have : n ∉ Q3.Nodes K := fun h_in => hn (h_finite.mem_toFinset.mpr h_in)
      simp only [this, ite_false]
  -- Use explicit bound
  use h_finite.toFinset.card * (2 * Real.sqrt 2 * Real.pi * K)
  rw [h_eq]
  calc ∑ n ∈ h_finite.toFinset, Q3.w_Q n
      ≤ ∑ _n ∈ h_finite.toFinset, (2 * Real.sqrt 2 * Real.pi * K) := by
        apply Finset.sum_le_sum
        intro n hn
        exact w_Q_bound K hK n (h_finite.mem_toFinset.1 hn)
    _ = h_finite.toFinset.card * (2 * Real.sqrt 2 * Real.pi * K) := by
        rw [Finset.sum_const, nsmul_eq_mul]

end Q3.Proofs.W_sum_BridgeV3

/-!
# Summary

CLEAN bridge for W_sum_finite:
- Imports only Q3.Basic.Defs (no Q3.Axioms!)
- Uses Q3.Nodes, Q3.w_Q, Q3.W_sum directly
- Proves finiteness using finite set + bounded terms

# Verification
```
lake build Q3.Proofs.W_sum_finite_bridge_v3
#print axioms Q3.Proofs.W_sum_BridgeV3.W_sum_finite_Q3
```
Expected: only [propext, Classical.choice, Quot.sound]
-/
