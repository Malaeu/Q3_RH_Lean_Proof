/-
Q3 Formalization: S_K Geometric Series Bound
=============================================

This file proves that S_K K t ≤ η when t ≤ t_min K η.

Mathematical argument:
- Let δ = delta_K K, x = exp(-δ²/(4t))
- S_K = 2x/(1-x) (geometric series sum)
- t_min = δ²/(4·log((2+η)/η))

At t = t_min:
- δ²/(4t) = log((2+η)/η)
- x = exp(-log((2+η)/η)) = η/(2+η)
- S_K = 2·(η/(2+η))/(1 - η/(2+η)) = 2η/2 = η ✓

For t < t_min:
- δ²/(4t) > log((2+η)/η)
- x = exp(-δ²/(4t)) < exp(-log((2+η)/η)) = η/(2+η)
- Since 2x/(1-x) is increasing in x, we get S_K < η

IMPORTANT: S_K(t) INCREASES with t (not decreases!):
- As t → ∞: -δ²/(4t) → 0, x → 1, S_K → ∞
- As t → 0+: -δ²/(4t) → -∞, x → 0, S_K → 0
-/

import Q3.Basic.Defs

set_option linter.mathlibStandardSet false

open scoped Real
open scoped Classical

set_option maxHeartbeats 400000

noncomputable section

namespace Q3.S_K_Bound

/-!
## Helper Lemmas

These lemmas establish the mathematical components of the proof.
-/

/-- The function f(x) = 2x/(1-x) is monotone increasing for 0 < x < 1

Proof: f(x) = 2x/(1-x) = -2 + 2/(1-x)
Derivative: f'(x) = 2/(1-x)² > 0 for x < 1
-/
lemma two_x_div_one_minus_x_mono {x y : ℝ} (hx : 0 < x) (hx1 : x < 1)
    (hy : 0 < y) (hy1 : y < 1) (hxy : x ≤ y) :
    2 * x / (1 - x) ≤ 2 * y / (1 - y) := by
  -- Proof: 2x/(1-x) ≤ 2y/(1-y) ⟺ 2x(1-y) ≤ 2y(1-x) ⟺ x - xy ≤ y - xy ⟺ x ≤ y
  have h1x_pos : 1 - x > 0 := by linarith
  have h1y_pos : 1 - y > 0 := by linarith
  have h1x_ne : 1 - x ≠ 0 := ne_of_gt h1x_pos
  have h1y_ne : 1 - y ≠ 0 := ne_of_gt h1y_pos
  -- Cross multiply: a/b ≤ c/d ⟺ a·d ≤ c·b for positive b,d
  have h_cross : 2 * x * (1 - y) ≤ 2 * y * (1 - x) := by nlinarith
  calc 2 * x / (1 - x)
      = (2 * x) * (1 - y) / ((1 - x) * (1 - y)) := by field_simp
    _ ≤ (2 * y) * (1 - x) / ((1 - x) * (1 - y)) := by
        apply div_le_div_of_nonneg_right h_cross
        exact le_of_lt (mul_pos h1x_pos h1y_pos)
    _ = 2 * y / (1 - y) := by field_simp

/-- delta_K is positive for K ≥ 1 -/
lemma delta_K_pos' (K : ℝ) (hK : K ≥ 1) : delta_K K > 0 := by
  unfold delta_K
  -- delta_K K = 1 / (2 * π * (⌊exp(2πK)⌋ + 1))
  positivity

/-- t_min is positive -/
lemma t_min_pos (K η : ℝ) (hK : K ≥ 1) (hη : η > 0) : t_min K η > 0 := by
  unfold t_min
  -- t_min = δ² / (4 * log((2+η)/η))
  apply div_pos
  · -- δ² > 0
    apply sq_pos_of_pos (delta_K_pos' K hK)
  · -- 4 * log((2+η)/η) > 0
    apply mul_pos (by norm_num : (4:ℝ) > 0)
    -- log((2+η)/η) > 0 iff (2+η)/η > 1 iff 2+η > η iff 2 > 0 ✓
    apply Real.log_pos
    have h2η_pos : (2 : ℝ) + η > 0 := by linarith
    rw [one_lt_div hη]
    linarith

/-- At t = t_min, the exponent equals log((2+η)/η) -/
lemma exponent_at_t_min (K η : ℝ) (hK : K ≥ 1) (hη : η > 0) :
    (delta_K K)^2 / (4 * t_min K η) = Real.log ((2 + η) / η) := by
  unfold t_min
  -- t_min = δ² / (4 * log((2+η)/η))
  -- So δ² / (4 * (δ² / (4 * log))) = log
  have hδ_pos := delta_K_pos' K hK
  have hδ_sq_pos : (delta_K K)^2 > 0 := sq_pos_of_pos hδ_pos
  have hδ_sq_ne : (delta_K K)^2 ≠ 0 := ne_of_gt hδ_sq_pos
  have hlog_pos : Real.log ((2 + η) / η) > 0 := by
    apply Real.log_pos
    rw [one_lt_div hη]
    linarith
  have h4_ne : (4:ℝ) ≠ 0 := by norm_num
  have hlog_ne : Real.log ((2 + η) / η) ≠ 0 := ne_of_gt hlog_pos
  field_simp

/-- At t = t_min, x = η/(2+η) -/
lemma x_at_t_min (K η : ℝ) (hK : K ≥ 1) (hη : η > 0) :
    Real.exp (-(delta_K K)^2 / (4 * t_min K η)) = η / (2 + η) := by
  -- Using exponent_at_t_min: δ²/(4t_min) = log((2+η)/η)
  -- So exp(-log((2+η)/η)) = η/(2+η) (inverse)
  have h_exp := exponent_at_t_min K η hK hη
  -- -(δ²/(4t_min)) = -log((2+η)/η)
  have h_neg : -(delta_K K)^2 / (4 * t_min K η) = -Real.log ((2 + η) / η) := by
    simp only [neg_div, h_exp]
  rw [h_neg]
  -- exp(-log((2+η)/η)) = 1/((2+η)/η) = η/(2+η)
  have h2η_pos : (2 : ℝ) + η > 0 := by linarith
  have h_frac_pos : (2 + η) / η > 0 := div_pos h2η_pos hη
  rw [Real.exp_neg, Real.exp_log h_frac_pos]
  field_simp

/-- S_K equals η exactly at t = t_min -/
lemma S_K_at_t_min (K η : ℝ) (hK : K ≥ 1) (hη : η > 0) :
    S_K K (t_min K η) = η := by
  unfold S_K
  rw [x_at_t_min K η hK hη]
  -- 2 · (η/(2+η)) / (1 - η/(2+η)) = 2η/(2+η) / (2/(2+η)) = η
  have h2η_pos : (2 : ℝ) + η > 0 := by linarith
  have h2η_ne : (2 : ℝ) + η ≠ 0 := ne_of_gt h2η_pos
  have h_denom : 1 - η / (2 + η) = 2 / (2 + η) := by field_simp; ring
  rw [h_denom]
  field_simp

/-- For t ≤ t_min, x ≤ η/(2+η) -/
lemma x_bound_for_small_t (K η t : ℝ) (hK : K ≥ 1) (hη : η > 0)
    (ht_pos : t > 0) (ht : t ≤ t_min K η) :
    Real.exp (-(delta_K K)^2 / (4 * t)) ≤ η / (2 + η) := by
  -- When t ≤ t_min:
  -- δ²/(4t) ≥ δ²/(4·t_min) = log((2+η)/η)
  -- So -δ²/(4t) ≤ -log((2+η)/η)
  -- Thus exp(-δ²/(4t)) ≤ exp(-log((2+η)/η)) = η/(2+η)
  have ht_min_pos : t_min K η > 0 := t_min_pos K η hK hη
  have hδ_pos : delta_K K > 0 := delta_K_pos' K hK
  have hδ_sq_pos : (delta_K K)^2 > 0 := sq_pos_of_pos hδ_pos
  -- Key: t ≤ t_min ⟹ 4t ≤ 4t_min ⟹ δ²/(4t) ≥ δ²/(4t_min) ⟹ -δ²/(4t) ≤ -δ²/(4t_min)
  have h_denom_ineq : 4 * t ≤ 4 * t_min K η := by linarith
  have h_div_ineq : (delta_K K)^2 / (4 * t) ≥ (delta_K K)^2 / (4 * t_min K η) := by
    apply div_le_div_of_nonneg_left (le_of_lt hδ_sq_pos)
    · exact mul_pos (by norm_num : (4:ℝ) > 0) ht_pos
    · exact h_denom_ineq
  have h_neg_ineq : -(delta_K K)^2 / (4 * t) ≤ -(delta_K K)^2 / (4 * t_min K η) := by
    simp only [neg_div]
    exact neg_le_neg h_div_ineq
  -- Use x_at_t_min to get the RHS
  rw [← x_at_t_min K η hK hη]
  exact Real.exp_le_exp.mpr h_neg_ineq

/-!
## Main Theorem
-/

/-- **Theorem**: S_K K t ≤ η when t ≤ t_min K η

This is the content of S_K_small_axiom.

Proof structure:
1. S_K = 2x/(1-x) where x = exp(-δ²/(4t))
2. By x_bound_for_small_t, x ≤ η/(2+η) when t ≤ t_min
3. By monotonicity of 2x/(1-x), we get S_K ≤ 2·(η/(2+η))/(1 - η/(2+η)) = η
-/
theorem S_K_small_proof (K t η : ℝ) (hK : K ≥ 1) (hη : η > 0)
    (ht_pos : t > 0) (ht : t ≤ t_min K η) :
    S_K K t ≤ η := by
  unfold S_K
  set x := Real.exp (-(delta_K K)^2 / (4 * t)) with hx_def
  -- Get bound on x
  have hx_bound := x_bound_for_small_t K η t hK hη ht_pos ht
  have hx_pos : x > 0 := Real.exp_pos _
  -- x < 1 follows from bound and η > 0 implies η/(2+η) < 1
  have hη_frac_lt_1 : η / (2 + η) < 1 := by
    have h2_pos : (2 : ℝ) + η > 0 := by linarith
    rw [div_lt_one h2_pos]
    linarith
  have hx_lt_1 : x < 1 := lt_of_le_of_lt hx_bound hη_frac_lt_1
  -- Apply monotonicity: since x ≤ η/(2+η) and 2x/(1-x) is increasing
  have hη_pos' : (0 : ℝ) < η / (2 + η) := by
    apply div_pos hη
    linarith
  have h_mono := two_x_div_one_minus_x_mono hx_pos hx_lt_1 hη_pos' hη_frac_lt_1 hx_bound
  -- Final calculation: 2·(η/(2+η))/(1 - η/(2+η)) = η
  have h_final : 2 * (η / (2 + η)) / (1 - η / (2 + η)) = η := by
    have h2η_pos : (2 : ℝ) + η > 0 := by linarith
    have h2η_ne : (2 : ℝ) + η ≠ 0 := ne_of_gt h2η_pos
    have h2_ne : (2 : ℝ) ≠ 0 := two_ne_zero
    -- 1 - η/(2+η) = 2/(2+η)
    have h_denom : 1 - η / (2 + η) = 2 / (2 + η) := by field_simp; ring
    rw [h_denom]
    -- 2 * (η/(2+η)) / (2/(2+η)) = η
    field_simp
  calc 2 * x / (1 - x)
      ≤ 2 * (η / (2 + η)) / (1 - η / (2 + η)) := h_mono
    _ = η := h_final

end Q3.S_K_Bound

end
