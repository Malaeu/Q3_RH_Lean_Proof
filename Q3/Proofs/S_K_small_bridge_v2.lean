/-
S_K Small Bridge v2 (SELF-CONTAINED)
====================================

This file creates a REAL bridge between Aristotle's S_K_small proof
and Q3.Basic.Defs definitions.

SELF-CONTAINED: Does NOT import standalone proof.
Instead, defines local copies and re-proves the theorem.

Strategy (same as node_spacing_bridge):
1. Define local copies of Aristotle's definitions in namespace
2. Prove the theorem using Aristotle's proof
3. Show definitions equal Q3 definitions (via rfl)
4. Transfer the theorem

CLOSES: S_K_small_axiom
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

namespace Q3.Proofs.S_K_SmallBridgeV2

/-! ## Aristotle Definitions (local copies) -/

/-- Aristotle's N_K -/
def N_K_A (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-- Aristotle's delta_K -/
def delta_K_A (K : ℝ) : ℝ := 1 / (2 * Real.pi * (N_K_A K + 1))

/-- Aristotle's t_min -/
def t_min_A (K η : ℝ) : ℝ := (delta_K_A K)^2 / (4 * Real.log (2/η + 1))

/-- Aristotle's S_K -/
def S_K_A (K t : ℝ) : ℝ :=
  2 * Real.exp (-(delta_K_A K)^2 / (4 * t)) / (1 - Real.exp (-(delta_K_A K)^2 / (4 * t)))

/-! ## Definition Equivalences -/

/-- N_K helper for Q3 (matching Aristotle's definition) -/
def N_K_Q3 (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

lemma N_K_eq : ∀ K, N_K_A K = N_K_Q3 K := fun _ => rfl

/-- delta_K definitions are equal -/
lemma delta_K_eq (K : ℝ) : delta_K_A K = Q3.delta_K K := by
  unfold delta_K_A N_K_A Q3.delta_K
  rfl

/-- S_K definitions are equal (given delta_K equality) -/
lemma S_K_eq (K t : ℝ) : S_K_A K t = Q3.S_K K t := by
  unfold S_K_A Q3.S_K
  rw [delta_K_eq]

/-- t_min definitions are algebraically equal:
    Aristotle: log(2/η + 1) = log((2 + η)/η)
    Q3: log((2 + η)/η) -/
lemma t_min_eq (K η : ℝ) (hη : η > 0) : t_min_A K η = Q3.t_min K η := by
  unfold t_min_A Q3.t_min
  rw [delta_K_eq]
  congr 1
  -- Show log(2/η + 1) = log((2 + η)/η)
  congr 1
  field_simp

/-! ## Main Theorem (Aristotle proof, local definitions) -/

/-- S_K_small with Aristotle definitions -/
theorem S_K_small_A (K t η : ℝ) (hK : K ≥ 1) (hη : η > 0) (ht : t ≤ t_min_A K η) :
    S_K_A K t ≤ η := by
  unfold t_min_A S_K_A at *
  by_cases h_cases : 0 < t
  · -- From t ≤ t_min, we have exp(-(delta_K)²/(4t)) ≤ exp(-log(2/η + 1))
    have h_exp : Real.exp (-(delta_K_A K)^2 / (4 * t)) ≤ Real.exp (-Real.log (2/η + 1)) := by
      rw [le_div_iff₀] at ht
      · exact Real.exp_le_exp.mpr (by rw [div_le_iff₀] <;> linarith)
      · exact mul_pos zero_lt_four (Real.log_pos <| by norm_num; positivity)
    norm_num [Real.exp_neg, Real.exp_log (show 0 < 2/η + 1 by positivity)] at *
    rw [div_le_iff₀] <;> nlinarith [Real.exp_pos (-(delta_K_A K)^2 / (4 * t)),
      mul_inv_cancel₀ (show (2/η + 1) ≠ 0 by positivity), div_mul_cancel₀ 2 hη.ne']
  · rcases eq_or_lt_of_le (le_of_not_gt h_cases) with rfl | h_cases'
    · -- t = 0 case: 4*0 = 0, so -delta²/(4*0) = -delta²/0 = 0, exp(0) = 1, denom = 0
      simp only [mul_zero, div_zero, neg_zero, Real.exp_zero, sub_self]
      exact hη.le
    · -- t < 0 case
      have h_neg_t : t < 0 := h_cases'
      have h_denom_neg : 1 - Real.exp (-(delta_K_A K)^2 / (4 * t)) ≤ 0 := by
        apply sub_nonpos.mpr
        apply Real.one_le_exp
        apply div_nonneg_of_nonpos
        · nlinarith
        · exact mul_nonpos_of_nonneg_of_nonpos (by norm_num : (4 : ℝ) ≥ 0) h_neg_t.le
      have h_numer_pos : 2 * Real.exp (-(delta_K_A K)^2 / (4 * t)) ≥ 0 := by positivity
      exact le_trans (div_nonpos_of_nonneg_of_nonpos h_numer_pos h_denom_neg) hη.le

/-! ## BRIDGE: Transfer to Q3 definitions -/

/-- S_K SMALL THEOREM with Q3 definitions (CLOSES S_K_small_axiom) -/
theorem S_K_small_Q3 (K t η : ℝ) (hK : K ≥ 1) (hη : η > 0) (ht : t ≤ Q3.t_min K η) :
    Q3.S_K K t ≤ η := by
  -- Rewrite t constraint using Aristotle's t_min
  rw [← t_min_eq K η hη] at ht
  -- Rewrite goal using Aristotle's S_K
  rw [← S_K_eq]
  -- Apply Aristotle's theorem
  exact S_K_small_A K t η hK hη ht

end Q3.Proofs.S_K_SmallBridgeV2

/-!
# Summary

This file provides a SELF-CONTAINED bridge (no import of standalone proof):
1. Defined local copies of Aristotle's definitions in namespace
2. Proved definition equivalences (via rfl)
3. Re-proved S_K_small_A using Aristotle's proof structure
4. Transferred to S_K_small_Q3 with Q3 definitions

S_K_small_Q3 CLOSES S_K_small_axiom without:
- Circular reference
- Namespace conflicts (all definitions are namespaced)

# Verification
```
lake env lean -c "import Q3.Proofs.S_K_small_bridge_v2; #print axioms Q3.Proofs.S_K_SmallBridgeV2.S_K_small_Q3"
```
Expected: only `propext`, `Classical.choice`, `Quot.sound`
-/
