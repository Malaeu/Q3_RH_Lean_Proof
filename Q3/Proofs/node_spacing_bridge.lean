/-
Node Spacing Bridge
===================

This file creates a REAL bridge between Aristotle's node_spacing proof
and Q3.Basic.Defs definitions.

Strategy:
1. Import Q3.Basic.Defs for Q3 definitions
2. Import Aristotle's proof (with its own definitions)
3. Show definitions are equal (via rfl where possible)
4. Transfer the theorem

NOTE: We create local aliases to avoid name collision.
-/

import Q3.Basic.Defs

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.NodeSpacingBridge

/-! ## Aristotle Definitions (local copies) -/

/-- Aristotle's xi_n -/
def xi_n_A (n : ℕ) : ℝ := Real.log n / (2 * Real.pi)

/-- Aristotle's Nodes -/
def Nodes_A (K : ℝ) : Set ℕ := {n | |xi_n_A n| ≤ K ∧ n ≥ 2}

/-- Aristotle's N_K -/
def N_K_A (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-- Aristotle's delta_K -/
def delta_K_A (K : ℝ) : ℝ := 1 / (2 * Real.pi * (N_K_A K + 1))

/-! ## Definition Equivalences -/

/-- xi_n is identical -/
lemma xi_n_eq : ∀ n, xi_n_A n = Q3.xi_n n := fun _ => rfl

/-- Nodes is identical (uses xi_n_eq) -/
lemma Nodes_eq : ∀ K, Nodes_A K = Q3.Nodes K := by
  intro K
  unfold Nodes_A Q3.Nodes
  simp only [xi_n_eq]

/-- N_K helper -/
def N_K_Q3 (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

lemma N_K_eq : ∀ K, N_K_A K = N_K_Q3 K := fun _ => rfl

/-- delta_K is identical -/
lemma delta_K_eq : ∀ K, delta_K_A K = Q3.delta_K K := by
  intro K
  unfold delta_K_A Q3.delta_K N_K_A
  rfl

/-! ## Helper Lemmas (from Aristotle, adapted) -/

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

lemma N_K_pos (K : ℝ) (hK : K ≥ 1) : 0 < N_K_A K := by
  have h_exp_pos : 1 < Real.exp (2 * Real.pi * K) := by
    have h_exp_pos : 0 < 2 * Real.pi * K := by
      apply mul_pos; exact mul_pos (by norm_num) (Real.pi_pos)
      exact lt_of_lt_of_le zero_lt_one hK
    apply Real.exp_lt_exp.mpr h_exp_pos |> lt_of_le_of_lt (by norm_num)
  apply Nat.floor_pos.mpr h_exp_pos.le

lemma n_le_N_K (K : ℝ) (n : ℕ) (hn : n ∈ Nodes_A K) : n ≤ N_K_A K := by
  have h_ln_le : Real.log n ≤ 2 * Real.pi * K := by
    have h_abs : |Real.log n / (2 * Real.pi)| ≤ K := hn.1
    nlinarith [abs_le.mp h_abs, Real.pi_pos,
      mul_div_cancel₀ (Real.log n) (by positivity : (2 * Real.pi) ≠ 0)]
  refine Nat.le_floor ?_
  rwa [Real.log_le_iff_le_exp (Nat.cast_pos.mpr <| Nat.pos_of_ne_zero <|
    by rintro rfl; exact absurd hn.2 <| by norm_num)] at h_ln_le

/-! ## Main Theorem (Aristotle proof, local definitions) -/

theorem node_spacing_A (K : ℝ) (hK : K ≥ 1) :
    ∀ (n₁ n₂ : ℕ), n₁ ∈ Nodes_A K → n₂ ∈ Nodes_A K → n₁ < n₂ →
      xi_n_A n₂ - xi_n_A n₁ ≥ delta_K_A K := by
  intro n₁ n₂ hn₁ hn₂ hn₁₂
  have h_log_div : Real.log (n₂ / n₁) ≥ Real.log (1 + 1 / n₁) := by
    bound
    · positivity
    · rw [le_div_iff₀] <;> aesop
      · nlinarith [inv_mul_cancel₀ (show (n₁ : ℝ) ≠ 0 by norm_cast; linarith [hn₁.2]),
          show (n₁ : ℝ) + 1 ≤ n₂ by norm_cast]
      · linarith [hn₁.2]
  have h_final_order : Real.log (1 + 1 / (n₁ : ℝ)) ≥ 1 / (N_K_A K + 1 : ℝ) := by
    have h_inv_le_inv_NK : 1 / (n₁ : ℝ) ≥ 1 / (N_K_A K : ℝ) := by
      exact one_div_le_one_div_of_le (Nat.cast_pos.mpr <| Nat.pos_of_ne_zero <|
        by rintro rfl; exact absurd hn₁.2 <| by norm_num) <|
        Nat.cast_le.mpr <| n_le_N_K K n₁ hn₁
    refine le_trans ?_ (Real.log_le_log (by positivity) (add_le_add_left h_inv_le_inv_NK _))
    exact log_one_plus_inv_ge_inv_add_one (N_K_A K) (Nat.cast_pos.mpr (N_K_pos K hK))
  unfold xi_n_A delta_K_A
  calc Real.log n₂ / (2 * Real.pi) - Real.log n₁ / (2 * Real.pi)
      = Real.log (n₂ / n₁) / (2 * Real.pi) := log_diff_eq_log_div n₁ n₂
          (Nat.pos_of_ne_zero <| by rintro rfl; exact absurd hn₁.2 <| by norm_num)
          (Nat.pos_of_ne_zero <| by rintro rfl; exact absurd hn₂.2 <| by norm_num)
    _ ≥ Real.log (1 + 1 / n₁) / (2 * Real.pi) := by
        apply div_le_div_of_nonneg_right h_log_div (by positivity)
    _ ≥ (1 / (N_K_A K + 1)) / (2 * Real.pi) := by
        apply div_le_div_of_nonneg_right h_final_order (by positivity)
    _ = 1 / (2 * Real.pi * (N_K_A K + 1)) := by field_simp

/-! ## BRIDGE: Transfer to Q3 definitions -/

/-- NODE SPACING THEOREM with Q3 definitions (CLOSES node_spacing_axiom) -/
theorem node_spacing_Q3 (K : ℝ) (hK : K ≥ 1) :
    ∀ (n₁ n₂ : ℕ), n₁ ∈ Q3.Nodes K → n₂ ∈ Q3.Nodes K → n₁ < n₂ →
      Q3.xi_n n₂ - Q3.xi_n n₁ ≥ Q3.delta_K K := by
  -- Rewrite using equivalences
  simp only [← xi_n_eq, ← Nodes_eq, ← delta_K_eq]
  -- Apply Aristotle's theorem
  exact node_spacing_A K hK

end Q3.Proofs.NodeSpacingBridge

/-!
# Summary

This file provides a REAL bridge:
1. Defined local copies of Aristotle's definitions
2. Proved they equal Q3 definitions (via rfl)
3. Proved node_spacing_A using Aristotle's proof
4. Transferred to node_spacing_Q3 with Q3 definitions

node_spacing_Q3 CLOSES node_spacing_axiom without circular reference!
-/
