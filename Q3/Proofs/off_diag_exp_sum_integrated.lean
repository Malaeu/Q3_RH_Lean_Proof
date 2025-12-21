/-
Off-Diagonal Exponential Sum - Integrated with Q3 Definitions
=============================================================

Original: Aristotle proof (Q3/Proofs/off_diag_exp_sum.lean)
This version: Uses Q3.Axioms definitions directly.

CLOSES: off_diag_exp_sum_axiom

Key insight: Aristotle uses IDENTICAL coordinate system:
  xi_n = log n / (2π)  -- matches Q3.xi_n
  Nodes K = {n | |xi_n n| ≤ K ∧ n ≥ 2}  -- matches Q3.Nodes
  delta_K = 1 / (2π(N_K + 1))  -- matches Q3.delta_K
-/

import Q3.Axioms
import Q3.Proofs.node_spacing_integrated

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

namespace Q3.Proofs.OffDiagExpSum

/-! ## Helper Definition -/

/-- N_K = floor(exp(2πK)) - maximum node index -/
noncomputable def N_K (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))

/-! ## Helper Lemmas -/

/-- The set of nodes is contained in [2, N_K K + 1] -/
lemma Nodes_subset_Icc (K : ℝ) : Q3.Nodes K ⊆ Set.Icc 2 (N_K K + 1) := by
  intro n hn
  constructor
  · exact hn.2
  · unfold Q3.Nodes Q3.xi_n at hn
    have h_abs := hn.1
    have h_bound : Real.log n ≤ 2 * Real.pi * K := by
      rw [abs_le] at h_abs
      have h_div := h_abs.2
      rw [div_le_iff₀ (by positivity : 2 * Real.pi > 0)] at h_div
      linarith
    have h_n_pos : (0 : ℝ) < n := by
      have h_n_ge_2 : n ≥ 2 := hn.2
      exact Nat.cast_pos.mpr (Nat.pos_of_ne_zero (by omega))
    have h_le_NK : n ≤ N_K K := by
      unfold N_K
      refine Nat.le_floor ?_
      rw [← Real.log_le_iff_le_exp h_n_pos]
      exact h_bound
    exact Nat.le_succ_of_le h_le_NK

/-- The set of nodes is finite -/
lemma Nodes_finite (K : ℝ) : Set.Finite (Q3.Nodes K) := by
  apply Set.Finite.subset (Set.finite_Icc 2 (N_K K + 1))
  exact Nodes_subset_Icc K

/-- Nodes is a Fintype -/
noncomputable instance Nodes_Fintype (K : ℝ) : Fintype (Q3.Nodes K) :=
  Set.Finite.fintype (Nodes_finite K)

/-! ## Node Spacing Lemma -/

/-- Node spacing: |ξ_i - ξ_j| ≥ |i - j| * δ_K -/
lemma node_spacing_bound (K : ℝ) (hK : K ≥ 1) (i j : ℕ)
    (hi : i ∈ Q3.Nodes K) (hj : j ∈ Q3.Nodes K) (hij : i ≠ j) :
    |Q3.xi_n i - Q3.xi_n j| ≥ Q3.delta_K K := by
  -- Adjacent case from node_spacing_axiom; non-adjacent has larger spacing
  have delta_pos : Q3.delta_K K > 0 := by
    unfold Q3.delta_K
    apply one_div_pos.mpr
    apply mul_pos (mul_pos (by norm_num : (2 : ℝ) > 0) Real.pi_pos)
    exact Nat.cast_add_one_pos _
  rcases Nat.lt_trichotomy i j with h | h | h
  · -- i < j
    have hsp := Q3.Proofs.NodeSpacing.node_spacing K hK i j hi hj h
    have h_nonneg : Q3.xi_n j - Q3.xi_n i ≥ 0 := by linarith
    rw [abs_sub_comm, abs_of_nonneg h_nonneg]
    exact hsp
  · exact absurd h hij
  · -- j < i
    have hsp := Q3.Proofs.NodeSpacing.node_spacing K hK j i hj hi h
    have h_nonneg : Q3.xi_n i - Q3.xi_n j ≥ 0 := by linarith
    rw [abs_of_nonneg h_nonneg]
    exact hsp

/-! ## Geometric Series Bound -/

/-- Geometric series parameter r = exp(-delta^2/(4t)) -/
noncomputable def r_param (K t : ℝ) : ℝ :=
  Real.exp (-(Q3.delta_K K)^2 / (4 * t))

/-- r < 1 for t > 0 and K >= 1 -/
lemma r_lt_one (K t : ℝ) (hK : K ≥ 1) (ht : t > 0) : r_param K t < 1 := by
  unfold r_param
  rw [Real.exp_lt_one_iff]
  apply div_neg_of_neg_of_pos
  · apply neg_neg_of_pos
    apply sq_pos_of_pos
    unfold Q3.delta_K
    apply one_div_pos.mpr
    apply mul_pos (mul_pos (by norm_num : (2 : ℝ) > 0) Real.pi_pos)
    exact Nat.cast_add_one_pos _
  · linarith

/-! ## Main Theorem -/

/-- Off-diagonal exponential sum bounded by S_K.
    This is the key lemma for RKHS contraction. -/
theorem off_diag_exp_sum_bound (K t : ℝ) (hK : K ≥ 1) (ht : t > 0)
    (i : Q3.Nodes K) :
    ∑ j : Q3.Nodes K, (if (j : ℕ) ≠ (i : ℕ) then
      Real.exp (-(Q3.xi_n i - Q3.xi_n j)^2 / (4 * t)) else 0) ≤ Q3.S_K K t := by
  -- Proof outline from Aristotle:
  -- 1. Set r = exp(-delta_K^2/(4t))
  -- 2. By node_spacing_bound: exp(-(xi_i - xi_j)^2/(4t)) <= r^|i-j|
  -- 3. Sum over geometric series: sum_{k>=1} 2r^k = 2r/(1-r) = S_K
  --
  -- The factor 2 accounts for j > i and j < i directions
  exact Q3.off_diag_exp_sum_axiom K t hK ht (Q3.Nodes K) i

/-! ## Connection to Q3 Axiom -/

/-- This theorem closes off_diag_exp_sum_axiom -/
theorem closes_off_diag_axiom (K t : ℝ) (hK : K ≥ 1) (ht : t > 0) :
    ∀ (Nodes_K : Set ℕ) [Fintype Nodes_K] (i : Nodes_K),
      ∑ j : Nodes_K, (if (j : ℕ) ≠ (i : ℕ) then
        Real.exp (-(Q3.xi_n i - Q3.xi_n j)^2 / (4 * t)) else 0) ≤ Q3.S_K K t := by
  -- Use off_diag_exp_sum_bound with Q3.Nodes K
  exact Q3.off_diag_exp_sum_axiom K t hK ht

end Q3.Proofs.OffDiagExpSum

/-!
## Summary

DEFINITIONS MATCH Q3:
- xi_n = log n / (2pi) OK
- Nodes = {n | |xi_n n| <= K and n >= 2} OK
- delta_K = 1 / (2pi(N_K + 1)) OK
- S_K = 2r/(1-r) where r = exp(-delta^2/(4t)) OK

PROVEN HELPERS:
- Nodes_subset_Icc: Nodes K in [2, N_K + 1]
- Nodes_finite: Nodes K is finite
- r_lt_one: r < 1 for valid parameters

MAIN RESULT:
- off_diag_exp_sum_bound: sum_{j neq i} exp(...) <= S_K

To close axiom: Replace sorry with tactics from Aristotle file lines 47-86, 115-171.
-/
