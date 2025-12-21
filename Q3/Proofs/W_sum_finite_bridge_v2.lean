/-
W_sum Finite Bridge v2 (SELF-CONTAINED)
=======================================

This file creates a REAL bridge between Aristotle's W_sum_finite proof
and Q3 definitions.

SELF-CONTAINED: Does NOT import standalone proof.
Instead, defines local copies and re-proves the theorem.

Strategy:
1. Define local copies of Aristotle's definitions in namespace
2. Prove finiteness using Aristotle's approach
3. Show definitions equal Q3 definitions
4. Transfer the theorem

CLOSES: W_sum_finite_axiom
-/

import Q3.Axioms  -- Need W_sum_axiom definition

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.W_sum_BridgeV2

/-! ## Aristotle Definitions (local copies) -/

/-- Aristotle's xi_n -/
def xi_n_A (n : ℕ) : ℝ := Real.log n / (2 * Real.pi)

/-- Aristotle's Λ (von Mangoldt) -/
def Λ_A (n : ℕ) : ℝ := ArithmeticFunction.vonMangoldt n

/-- Aristotle's w_Q -/
def w_Q_A (n : ℕ) : ℝ := 2 * Λ_A n / Real.sqrt n

/-- Aristotle's ActiveNodes -/
def ActiveNodes_A (K : ℝ) : Set ℕ := {n | |xi_n_A n| ≤ K ∧ n ≥ 2}

/-- Aristotle's N_K -/
def N_K_A (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-- Aristotle's W_sum -/
def W_sum_A (K : ℝ) : ℝ := ∑' n, if n ∈ ActiveNodes_A K then w_Q_A n else 0

/-! ## Definition Equivalences -/

/-- xi_n definitions are equal -/
lemma xi_n_eq (n : ℕ) : xi_n_A n = Q3.xi_n n := by
  unfold xi_n_A Q3.xi_n
  rfl

/-- ActiveNodes_A equals Q3.ActiveNodes_axiom -/
lemma ActiveNodes_eq (K : ℝ) : ActiveNodes_A K = Q3.ActiveNodes_axiom K := by
  unfold ActiveNodes_A Q3.ActiveNodes_axiom
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
lemma w_Q_eq (n : ℕ) : w_Q_A n = Q3.w_Q n := by
  unfold w_Q_A Λ_A Q3.w_Q
  rfl

/-- W_sum definitions are equal -/
lemma W_sum_eq (K : ℝ) : W_sum_A K = Q3.W_sum_axiom K := by
  unfold W_sum_A Q3.W_sum_axiom
  simp only [ActiveNodes_eq, w_Q_eq]

/-! ## Helper Lemmas -/

/-- ActiveNodes is finite for any K -/
lemma ActiveNodes_finite_A (K : ℝ) : (ActiveNodes_A K).Finite := by
  unfold ActiveNodes_A xi_n_A
  let N := Nat.floor (Real.exp (2 * Real.pi * |K|))
  apply Set.Finite.subset (Set.finite_Icc 0 N)
  intro n hn
  simp only [Set.mem_Icc, Nat.zero_le, true_and]
  cases abs_cases K <;> aesop
  · rw [abs_of_nonneg (div_nonneg (Real.log_nonneg (by norm_cast; linarith)) (by positivity))] at left
    exact Nat.le_floor <| by rw [abs_of_nonneg h]; rw [div_le_iff₀ <| by positivity] at *; rw [← Real.log_le_iff_le_exp <| by positivity] at *; linarith
  · linarith [abs_le.mp left]

/-- Each w_Q n is bounded for n ∈ ActiveNodes -/
lemma w_Q_bound_A (K : ℝ) (hK : K > 0) (n : ℕ) (hn : n ∈ ActiveNodes_A K) :
    w_Q_A n ≤ 2 * Real.sqrt 2 * Real.pi * K := by
  unfold ActiveNodes_A at hn
  have h_lambda_le_log : ∀ n, Λ_A n ≤ Real.log n := by
    unfold Λ_A
    exact fun n => ArithmeticFunction.vonMangoldt_le_log
  have h_log_le_2piK : Real.log n ≤ 2 * Real.pi * K := by
    unfold xi_n_A at hn; aesop
    nlinarith [abs_le.mp left, Real.pi_pos, mul_div_cancel₀ (Real.log n) (by positivity : (2 * Real.pi) ≠ 0)]
  have h_sqrt_ge_sqrt2 : Real.sqrt n ≥ Real.sqrt 2 := by
    exact Real.sqrt_le_sqrt <| mod_cast hn.2
  have h_w_Q_le : w_Q_A n ≤ 2 * (2 * Real.pi * K) / Real.sqrt 2 := by
    exact div_le_div₀ (by positivity) (by linarith [h_lambda_le_log n]) (by positivity) (by linarith)
  exact h_w_Q_le.trans_eq (by rw [div_eq_iff (by positivity)]; ring_nf; norm_num; ring_nf)

/-! ## Main Theorem (Direct approach) -/

/-- W_sum is finite - direct existence proof -/
lemma W_sum_is_finite_A (K : ℝ) (hK : K > 0) : ∃ B, W_sum_A K ≤ B := by
  -- The sum is over a finite set (ActiveNodes is finite)
  -- and each term is bounded, so the sum is bounded
  have h_finite : (ActiveNodes_A K).Finite := ActiveNodes_finite_A K
  -- Convert tsum to finite sum
  have h_eq : W_sum_A K = ∑ n ∈ h_finite.toFinset, w_Q_A n := by
    unfold W_sum_A
    have h_summable : Summable (fun n => if n ∈ ActiveNodes_A K then w_Q_A n else 0) := by
      apply summable_of_ne_finset_zero (s := h_finite.toFinset)
      intro n hn
      have : n ∉ ActiveNodes_A K := fun h_in => hn (h_finite.mem_toFinset.mpr h_in)
      simp only [this, ite_false]
    rw [tsum_eq_sum (f := fun n => if n ∈ ActiveNodes_A K then w_Q_A n else 0) (s := h_finite.toFinset)]
    · apply Finset.sum_congr rfl
      intro n hn
      have hn' : n ∈ ActiveNodes_A K := h_finite.mem_toFinset.mp hn
      simp only [hn', ite_true]
    · intro n hn
      have : n ∉ ActiveNodes_A K := fun h_in => hn (h_finite.mem_toFinset.mpr h_in)
      simp only [this, ite_false]
  -- Each term is bounded
  have h_bound : ∀ n ∈ ActiveNodes_A K, w_Q_A n ≤ 2 * Real.sqrt 2 * Real.pi * K :=
    fun n hn => w_Q_bound_A K hK n hn
  -- Use explicit bound
  use h_finite.toFinset.card * (2 * Real.sqrt 2 * Real.pi * K)
  rw [h_eq]
  calc ∑ n ∈ h_finite.toFinset, w_Q_A n
      ≤ ∑ _n ∈ h_finite.toFinset, (2 * Real.sqrt 2 * Real.pi * K) := by
        apply Finset.sum_le_sum
        intro n hn
        exact h_bound n (h_finite.mem_toFinset.1 hn)
    _ = h_finite.toFinset.card * (2 * Real.sqrt 2 * Real.pi * K) := by
        rw [Finset.sum_const, nsmul_eq_mul]

/-! ## BRIDGE: Transfer to Q3 definitions -/

/-- W_SUM FINITE THEOREM with Q3 definitions (CLOSES W_sum_finite_axiom) -/
theorem W_sum_finite_Q3 (K : ℝ) (hK : K > 0) : ∃ B, Q3.W_sum_axiom K ≤ B := by
  rw [← W_sum_eq]
  exact W_sum_is_finite_A K hK

end Q3.Proofs.W_sum_BridgeV2

/-!
# Summary

This file provides a SELF-CONTAINED bridge (no import of standalone proof):
1. Defined local copies of Aristotle's definitions in namespace
2. Proved ActiveNodes_finite, w_Q_bound, W_sum_bound_explicit
3. Showed definition equivalences
4. Transferred W_sum_is_finite to Q3 definitions

W_sum_finite_Q3 CLOSES W_sum_finite_axiom without:
- Circular reference
- Namespace conflicts (all definitions are namespaced)

# Verification
```
lake env lean -c "import Q3.Proofs.W_sum_finite_bridge_v2; #print axioms Q3.Proofs.W_sum_BridgeV2.W_sum_finite_Q3"
```
Expected: only `propext`, `Classical.choice`, `Quot.sound`
-/
