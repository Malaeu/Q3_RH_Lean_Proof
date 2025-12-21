/-
RKHS Coordinate Rescaling Bridge
=================================

This file provides coordinate rescaling lemmas between:
- Standalone α-coordinates: α_n = log(n)
- Q3 ξ-coordinates: ξ_n = log(n)/(2π)

Key rescaling invariant:
  α = 2π · ξ
  t_α = (2π)² · t_ξ
  δ_α = 2π · δ_ξ
  δ²/(4t) is INVARIANT under rescaling!

This allows transferring RKHS_contraction from standalone to Q3.
-/

import Q3.Basic.Defs

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real
open Real

namespace RKHS_Rescaling

/-! ## Coordinate Definitions -/

/-- Standalone α-coordinate: α_n = log(n) -/
noncomputable def alpha (n : ℕ) : ℝ := Real.log n

/-- Q3 ξ-coordinate: ξ_n = log(n)/(2π) -/
noncomputable def xi (n : ℕ) : ℝ := Real.log n / (2 * Real.pi)

/-! ## Core Rescaling Lemmas -/

/-- Key rescaling: α_n = (2π) · ξ_n -/
lemma alpha_eq_two_pi_mul_xi (n : ℕ) : alpha n = 2 * Real.pi * xi n := by
  unfold alpha xi
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp

/-- Q3.xi_n matches our xi -/
lemma xi_eq_Q3_xi_n (n : ℕ) : xi n = Q3.xi_n n := by
  unfold xi Q3.xi_n
  ring

/-- xi expressed in terms of alpha -/
lemma xi_eq_alpha_div (n : ℕ) : xi n = alpha n / (2 * Real.pi) := by
  have h := alpha_eq_two_pi_mul_xi n
  have hpi : 2 * Real.pi ≠ 0 := by positivity
  field_simp [hpi] at h ⊢
  linarith

/-- Coordinate difference rescaling -/
lemma alpha_diff_eq_two_pi_mul_xi_diff (m n : ℕ) :
    alpha m - alpha n = 2 * Real.pi * (xi m - xi n) := by
  simp only [alpha_eq_two_pi_mul_xi]
  ring

/-- Squared difference rescaling -/
lemma alpha_diff_sq_eq_four_pi_sq_mul_xi_diff_sq (m n : ℕ) :
    (alpha m - alpha n)^2 = (2 * Real.pi)^2 * (xi m - xi n)^2 := by
  rw [alpha_diff_eq_two_pi_mul_xi_diff]
  ring

/-! ## Heat Kernel Invariance -/

/-- Heat kernel exponent is invariant under coordinate rescaling:
    (α_m - α_n)² / (4 t_α) = (ξ_m - ξ_n)² / (4 t_ξ)
    when t_α = (2π)² · t_ξ -/
lemma heat_exponent_rescale_invariant (m n : ℕ) (t_xi : ℝ) :
    let t_alpha := (2 * Real.pi)^2 * t_xi
    (alpha m - alpha n)^2 / (4 * t_alpha) = (xi m - xi n)^2 / (4 * t_xi) := by
  simp only
  rw [alpha_diff_sq_eq_four_pi_sq_mul_xi_diff_sq]
  have hpi : (2 * Real.pi)^2 ≠ 0 := by positivity
  have h4 : (4 : ℝ) ≠ 0 := by norm_num
  by_cases ht : t_xi = 0
  · simp [ht]
  · field_simp [hpi, h4, ht]

/-- Negated heat kernel exponent is invariant -/
lemma neg_heat_exponent_rescale_invariant (m n : ℕ) (t_xi : ℝ) :
    -(alpha m - alpha n)^2 / (4 * ((2 * Real.pi)^2 * t_xi)) =
    -(xi m - xi n)^2 / (4 * t_xi) := by
  have h := heat_exponent_rescale_invariant m n t_xi
  simp only at h
  simp only [neg_div]
  rw [h]

/-- Heat kernel exponential is invariant:
    exp(-(α_m - α_n)² / (4 t_α)) = exp(-(ξ_m - ξ_n)² / (4 t_ξ)) -/
lemma heat_exp_rescale_invariant (m n : ℕ) (t_xi : ℝ) (ht : t_xi > 0) :
    Real.exp (-(alpha m - alpha n)^2 / (4 * ((2 * Real.pi)^2 * t_xi))) =
    Real.exp (-(xi m - xi n)^2 / (4 * t_xi)) := by
  congr 1
  exact neg_heat_exponent_rescale_invariant m n t_xi

/-! ## Node Window Rescaling -/

/-- Node count in α-window K_α corresponds to ξ-window K_ξ when K_α = 2π · K_ξ -/
lemma nodes_window_correspondence (K_xi : ℝ) :
    let K_alpha := 2 * Real.pi * K_xi
    ∀ n : ℕ, (alpha n ≤ K_alpha ∧ n ≥ 1) ↔ (xi n ≤ K_xi ∧ n ≥ 1) := by
  intro K_alpha n
  have hpi : 0 < 2 * Real.pi := by positivity
  have hpi_ne : 2 * Real.pi ≠ 0 := ne_of_gt hpi
  constructor
  · intro ⟨hα, hn⟩
    constructor
    · rw [xi_eq_alpha_div]
      calc alpha n / (2 * Real.pi) ≤ K_alpha / (2 * Real.pi) := by
            exact div_le_div_of_nonneg_right hα (le_of_lt hpi)
        _ = K_xi := by simp only [K_alpha]; field_simp
    · exact hn
  · intro ⟨hξ, hn⟩
    constructor
    · calc alpha n = 2 * Real.pi * xi n := alpha_eq_two_pi_mul_xi n
        _ ≤ 2 * Real.pi * K_xi := mul_le_mul_of_nonneg_left hξ (le_of_lt hpi)
    · exact hn

/-! ## Spacing Rescaling -/

/-- Minimal spacing rescales: δ_α = 2π · δ_ξ

For consecutive nodes n₁ < n₂:
  α_{n₂} - α_{n₁} = log(n₂/n₁) ≥ log(1 + 1/N_K)
  ξ_{n₂} - ξ_{n₁} = log(n₂/n₁)/(2π)

So δ_α = 2π · δ_ξ -/
lemma delta_rescale (delta_xi : ℝ) (m n : ℕ) (hmn : m < n) :
    let delta_alpha := 2 * Real.pi * delta_xi
    alpha n - alpha m ≥ delta_alpha ↔ xi n - xi m ≥ delta_xi := by
  intro delta_alpha
  have hpi : 0 < 2 * Real.pi := by positivity
  have hpi_ne : 2 * Real.pi ≠ 0 := ne_of_gt hpi
  have hdiff : xi n - xi m = (alpha n - alpha m) / (2 * Real.pi) := by
    rw [xi_eq_alpha_div, xi_eq_alpha_div]
    field_simp
  constructor
  · intro hα
    rw [hdiff]
    calc (alpha n - alpha m) / (2 * Real.pi) ≥ delta_alpha / (2 * Real.pi) := by
          exact div_le_div_of_nonneg_right hα (le_of_lt hpi)
      _ = delta_xi := by simp only [delta_alpha]; field_simp
  · intro hξ
    calc alpha n - alpha m = 2 * Real.pi * (xi n - xi m) := alpha_diff_eq_two_pi_mul_xi_diff n m
      _ ≥ 2 * Real.pi * delta_xi := mul_le_mul_of_nonneg_left hξ (le_of_lt hpi)

/-! ## S_K Geometric Series Rescaling -/

/-- S_K is defined by δ²/(4t), which is invariant under rescaling.
    So S_K(K_α, t_α) = S_K(K_ξ, t_ξ) when properly rescaled.

    This is a documentation lemma - the actual invariance follows from
    heat_exponent_rescale_invariant. -/
lemma S_K_rescale_doc : True := trivial

/-! ## t_min Rescaling -/

/-- t_min rescales: t_min_α = (2π)² · t_min_ξ

Since t_min = δ² / (4 · log((2+η)/η)):
  t_min_α = δ_α² / (4 · log((2+η)/η))
         = (2π)² · δ_ξ² / (4 · log((2+η)/η))
         = (2π)² · t_min_ξ -/
lemma t_min_rescale (K_xi eta : ℝ) (hK : K_xi > 0) (heta : eta > 0) :
    let delta_xi := Q3.delta_K K_xi
    let delta_alpha := 2 * Real.pi * delta_xi
    let t_min_xi := delta_xi^2 / (4 * Real.log ((2 + eta) / eta))
    let t_min_alpha := delta_alpha^2 / (4 * Real.log ((2 + eta) / eta))
    t_min_alpha = (2 * Real.pi)^2 * t_min_xi := by
  simp only
  ring

/-! ## Weight Preservation -/

/-- w_RKHS is independent of coordinate system (defined in terms of n, not coordinates) -/
lemma w_RKHS_coordinate_independent (n : ℕ) :
    Q3.w_RKHS n = ArithmeticFunction.vonMangoldt n / Real.sqrt n := by
  rfl

/-- w_max is independent of coordinate system -/
lemma w_max_coordinate_independent :
    Q3.w_max = 2 / Real.exp 1 := by
  rfl

end RKHS_Rescaling
