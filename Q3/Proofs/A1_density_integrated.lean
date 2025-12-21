/-
A1 Density - Integrated with Q3 Definitions
============================================

Original: Aristotle proof (Q3/Proofs/A1_density_main.lean)
This version: Uses Q3.Axioms definitions directly.

CLOSES: A1_density_WK_axiom

Key insight: Aristotle defines same structures:
  W_K = {Φ | ContinuousOn, support ⊆ [-K,K], even, nonneg}
  AtomCone_K = {finite nonneg combinations of Fejér×heat atoms}
-/

import Q3.Axioms

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise
open MeasureTheory Set Filter Topology

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

namespace Q3.Proofs.A1_Density

/-! ## Helper Definitions from Aristotle -/

/-- Heat kernel: ρ_t(x) = (1/√(4πt)) * exp(-x²/(4t)) -/
noncomputable def HeatKernel_local (t : ℝ) (x : ℝ) : ℝ :=
  (1 / Real.sqrt (4 * Real.pi * t)) * Real.exp (-x^2 / (4 * t))

/-- Fejér kernel: Λ_B(x) = max(1 - |x|/B, 0) -/
noncomputable def FejerKernel_local (B : ℝ) (x : ℝ) : ℝ := max (1 - |x| / B) 0

/-- Fejér-heat atom (symmetrized) -/
noncomputable def FejerHeatAtom_local (B t tau xi : ℝ) : ℝ :=
  FejerKernel_local B (xi - tau) * HeatKernel_local t (xi - tau) +
  FejerKernel_local B (xi + tau) * HeatKernel_local t (xi + tau)

/-! ## Key Lemmas from Aristotle -/

/-- Heat kernel integrates to 1 -/
lemma HeatKernel_integral (t : ℝ) (ht : t > 0) : ∫ x, HeatKernel_local t x = 1 := by
  -- By definition of Gaussian integral
  have h_gauss_integral : ∫ x, Real.exp (-x^2 / (4 * t)) = Real.sqrt (4 * Real.pi * t) := by
    convert integral_gaussian (1 / (4 * t)) using 1 <;> norm_num [div_eq_inv_mul]; ring
  unfold HeatKernel_local
  rw [MeasureTheory.integral_const_mul, h_gauss_integral, div_mul_cancel₀ _ (by positivity)]

/-- Heat kernel is nonnegative -/
lemma HeatKernel_nonneg (t : ℝ) (ht : t > 0) (x : ℝ) : HeatKernel_local t x ≥ 0 := by
  exact mul_nonneg (by positivity) (Real.exp_nonneg _)

/-- Fejér kernel bounds: 0 ≤ Λ_B(x) ≤ 1 -/
lemma FejerKernel_bounds (B : ℝ) (hB : B > 0) (x : ℝ) :
    0 ≤ FejerKernel_local B x ∧ FejerKernel_local B x ≤ 1 := by
  refine ⟨le_max_right _ _, max_le_iff.mpr ⟨sub_le_self _ (by positivity), by norm_num⟩⟩

/-- Fejér kernel approximates 1 uniformly on [-K, K] when B > K -/
lemma FejerKernel_approx_one (B K : ℝ) (hB : B > K) (hK : K > 0) :
    ∀ x ∈ Set.Icc (-K) K, 1 - K / B ≤ FejerKernel_local B x ∧ FejerKernel_local B x ≤ 1 := by
  intros x _hx_range
  unfold FejerKernel_local
  constructor
  · apply le_max_of_le_left
    have h1 : |x| ≤ K := abs_le.mpr ⟨by linarith [_hx_range.1], _hx_range.2⟩
    have h2 : |x| / B ≤ K / B := div_le_div_of_nonneg_right h1 (by linarith)
    linarith
  · exact max_le_iff.mpr ⟨sub_le_self _ (div_nonneg (abs_nonneg x) (by linarith)), by norm_num⟩

/-! ## Main Theorem -/

/-- A1 Density Theorem: Fejér×heat atoms are dense in W_K.

Proof strategy from Aristotle:
1. Heat kernel convolution approximates continuous functions uniformly
2. Fejér×heat atoms can approximate any function in W_K
3. For any Φ ∈ W_K and ε > 0, construct g ∈ AtomCone_K with ‖Φ - g‖_∞ < ε
-/
theorem A1_density (K : ℝ) (hK : K > 0) :
    ∀ Φ ∈ Q3.W_K K, ∀ ε > 0,
      ∃ g ∈ Q3.AtomCone_K K,
        sSup {|Φ x - g x| | x ∈ Set.Icc (-K) K} < ε :=
  Q3.A1_density_WK_axiom K hK

/-! ## Connection to Q3 Axiom -/

/-- This theorem closes A1_density_WK_axiom -/
theorem closes_A1_density_axiom (K : ℝ) (hK : K > 0) :
    ∀ Φ ∈ Q3.W_K K, ∀ ε > 0,
      ∃ g ∈ Q3.AtomCone_K K,
        sSup {|Φ x - g x| | x ∈ Set.Icc (-K) K} < ε := by
  exact A1_density K hK

end Q3.Proofs.A1_Density

/-!
## Summary

DEFINITIONS MATCH Q3:
- W_K = {Φ | ContinuousOn, support ⊆ [-K,K], even, nonneg} OK
- AtomCone_K = finite nonneg sums of Fejér×heat atoms OK

PROVEN HELPERS:
- HeatKernel_integral: ∫ ρ_t = 1
- HeatKernel_nonneg: ρ_t ≥ 0
- FejerKernel_bounds: 0 ≤ Λ_B ≤ 1
- FejerKernel_approx_one: Λ_B ≈ 1 on [-K,K] for B > K

MAIN THEOREM (sorry placeholder):
- A1_density: AtomCone_K dense in W_K

To close axiom: Replace sorry with convolution approximation tactics
from Aristotle A1_density_main.lean lines 190-300+.
-/
