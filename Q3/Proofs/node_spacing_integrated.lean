/-
Node Spacing - Integrated with Q3 Definitions
==============================================

Original: Aristotle proof (Q3/Proofs/node_spacing.lean)
This version: Uses Q3.Basic.Defs definitions.

CLOSES: node_spacing_axiom
-/

import Q3.Axioms

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

namespace Q3.Proofs.NodeSpacing

/-! ## Helper Definition -/

/-- N_K = floor(exp(2πK)) -/
noncomputable def N_K (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-! ## Helper Lemmas (from Aristotle) -/

lemma log_diff_eq_log_div (n₁ n₂ : ℕ) (h₁ : n₁ > 0) (h₂ : n₂ > 0) :
    Real.log n₂ / (2 * Real.pi) - Real.log n₁ / (2 * Real.pi) =
    Real.log (n₂ / n₁) / (2 * Real.pi) := by
  rw [Real.log_div (by positivity) (by positivity)]
  ring

lemma log_one_plus_inv_ge_inv_add_one (x : ℝ) (hx : x > 0) :
    Real.log (1 + 1 / x) ≥ 1 / (x + 1) := by
  have h_log_ineq : ∀ y : ℝ, 0 < y → Real.log (1 + y) ≥ y / (1 + y) := by
    exact fun y hy => by
      rw [ge_iff_le, div_le_iff₀ (by positivity)]
      nlinarith [Real.log_inv (1 + y), Real.log_le_sub_one_of_pos (inv_pos.mpr (by positivity : 0 < (1 + y))),
                 mul_inv_cancel₀ (by positivity : (1 + y) ≠ 0)]
  specialize h_log_ineq (1 / x) (by positivity)
  field_simp [hx] at h_log_ineq ⊢
  exact h_log_ineq

lemma N_K_pos (K : ℝ) (hK : K ≥ 1) : 0 < N_K K := by
  have h_exp_pos : 1 < Real.exp (2 * Real.pi * K) := by
    have h_exp_pos : 0 < 2 * Real.pi * K := by
      apply mul_pos
      exact mul_pos (by norm_num) (Real.pi_pos)
      exact lt_of_lt_of_le zero_lt_one hK
    apply Real.exp_lt_exp.mpr h_exp_pos |> lt_of_le_of_lt (by norm_num)
  apply Nat.floor_pos.mpr h_exp_pos.le

lemma n_le_N_K (K : ℝ) (n : ℕ) (hn : n ∈ Q3.Nodes K) : n ≤ N_K K := by
  have h_ln_le : Real.log n ≤ 2 * Real.pi * K := by
    have h_abs : |Q3.xi_n n| ≤ K := hn.1
    unfold Q3.xi_n at h_abs
    nlinarith [abs_le.mp h_abs, Real.pi_pos, mul_div_cancel₀ (Real.log n) (by positivity : (2 * Real.pi) ≠ 0)]
  refine Nat.le_floor ?_
  rwa [Real.log_le_iff_le_exp (Nat.cast_pos.mpr <| Nat.pos_of_ne_zero <| by rintro rfl; exact absurd hn.2 <| by norm_num)] at h_ln_le

/-- delta_K equivalence: Q3.delta_K = 1 / (2π(N_K + 1)) -/
lemma delta_K_eq (K : ℝ) : Q3.delta_K K = 1 / (2 * Real.pi * (N_K K + 1)) := by
  unfold Q3.delta_K N_K
  rfl

/-! ## Main Theorem -/

/-- Node spacing theorem. CLOSES node_spacing_axiom. -/
theorem node_spacing (K : ℝ) (hK : K ≥ 1) :
    ∀ (n₁ n₂ : ℕ), n₁ ∈ Q3.Nodes K → n₂ ∈ Q3.Nodes K → n₁ < n₂ →
      Q3.xi_n n₂ - Q3.xi_n n₁ ≥ Q3.delta_K K := by
  intro n₁ n₂ hn₁ hn₂ hn₁₂
  rw [delta_K_eq]
  set x₁ := n₁
  set x₂ := n₂
  have h_log_div : Real.log (x₂ / x₁) ≥ Real.log (1 + 1 / x₁) := by
    bound
    · positivity
    · rw [le_div_iff₀] <;> aesop
      · nlinarith [inv_mul_cancel₀ (show (n₁ : ℝ) ≠ 0 by norm_cast; linarith [hn₁.2]),
                   show (n₁ : ℝ) + 1 ≤ n₂ by norm_cast]
      · linarith [hn₁.2]
  have h_final_order : Real.log (1 + 1 / (n₁ : ℝ)) ≥ 1 / (N_K K + 1 : ℝ) := by
    have h_inv_le_inv_NK : 1 / (n₁ : ℝ) ≥ 1 / (N_K K : ℝ) := by
      exact one_div_le_one_div_of_le (Nat.cast_pos.mpr <| Nat.pos_of_ne_zero <| by
        rintro rfl; exact absurd hn₁.2 <| by norm_num) <| Nat.cast_le.mpr <| n_le_N_K K n₁ hn₁
    refine le_trans ?_ (Real.log_le_log (by positivity) (add_le_add_left h_inv_le_inv_NK _))
    convert log_one_plus_inv_ge_inv_add_one (N_K K : ℝ) _ using 1
    norm_num
    exact N_K_pos K hK
  unfold Q3.xi_n
  aesop
  rw [Real.log_div] at h_log_div <;> ring_nf at * <;> aesop
  · nlinarith [inv_pos.mpr Real.pi_pos]
  · exact not_le_of_gt (by positivity) h_final_order

end Q3.Proofs.NodeSpacing
