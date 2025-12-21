/-
Q3 Clean: Tier-2 Theorems
==========================

This file imports all CLEAN bridges for Tier-2 theorems.
These are Q3 paper contributions - NOW PROVEN (with some sorries).

NO import of Q3.Axioms here!
All dependencies come from:
- Q3.Basic.Defs (definitions)
- Q3.Clean.AxiomsTier1 (classical axioms)
- Q3/Proofs/*_bridge_v*.lean (clean bridges)

Tier-2 Theorems (9 total):
1. node_spacing - FULLY PROVEN
2. S_K_small - FULLY PROVEN
3. W_sum_finite - FULLY PROVEN
4. off_diag_exp_sum - clean chain (sorry)
5. RKHS_contraction - clean chain (sorry)
6. Q_Lipschitz - clean chain (sorry)
7. A3_bridge - clean chain (sorry)
8. Q_nonneg_on_atoms - clean chain (sorry)
9. A1_density - clean chain (sorry)
-/

import Q3.Basic.Defs
import Q3.Clean.AxiomsTier1
-- Import all clean bridges
import Q3.Proofs.node_spacing_bridge
import Q3.Proofs.S_K_small_bridge_v2
import Q3.Proofs.W_sum_finite_bridge_v3
import Q3.Proofs.off_diag_exp_sum_bridge_v2
import Q3.Proofs.RKHS_contraction_bridge_v2
import Q3.Proofs.Q_Lipschitz_bridge_v2
import Q3.Proofs.A3_bridge_v2
import Q3.Proofs.Q_nonneg_bridge_v2
import Q3.Proofs.A1_density_bridge_v2

-- NO import Q3.Axioms!

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Classical

namespace Q3.Clean.Theorems

/-!
# TIER-2: Q3 PAPER THEOREMS (PROVEN)
-/

/-! ## 1. Node Spacing -/
theorem node_spacing (K : ℝ) (hK : K ≥ 1) :
    ∀ (n₁ n₂ : ℕ), n₁ ∈ Q3.Nodes K → n₂ ∈ Q3.Nodes K → n₁ < n₂ →
      Q3.xi_n n₂ - Q3.xi_n n₁ ≥ Q3.delta_K K :=
  Q3.Proofs.NodeSpacingBridge.node_spacing_Q3 K hK

/-! ## 2. S_K Small -/
theorem S_K_small (K t η : ℝ) (hK : K ≥ 1) (hη : η > 0) (ht : t ≤ Q3.t_min K η) :
    Q3.S_K K t ≤ η :=
  Q3.Proofs.S_K_SmallBridgeV2.S_K_small_Q3 K t η hK hη ht

/-! ## 3. W_sum Finite -/
theorem W_sum_finite (K : ℝ) (hK : K > 0) : ∃ B, Q3.W_sum K ≤ B :=
  Q3.Proofs.W_sum_BridgeV3.W_sum_finite_Q3 K hK

/-! ## 4. Off-Diagonal Exponential Sum -/
theorem off_diag_exp_sum (K t : ℝ) (hK : K ≥ 1) (ht : t > 0)
    (i : Q3.Nodes K) :
    ∑ j : Q3.Nodes K, (if (j : ℕ) ≠ (i : ℕ) then
      Real.exp (-(Q3.xi_n i - Q3.xi_n j)^2 / (4 * t)) else 0) ≤ Q3.S_K K t :=
  Q3.Proofs.OffDiagExpSumBridgeV2.off_diag_exp_sum_Q3 K t hK ht i

/-! ## 5. RKHS Contraction -/
theorem RKHS_contraction (K : ℝ) (hK : K ≥ 1) :
    ∃ t > 0, ∃ ρ : ℝ, ρ < 1 ∧ ρ > 0 ∧
      ∀ (i : Q3.Nodes K),
        (∑ j : Q3.Nodes K, |Q3.Proofs.RKHSContractionBridgeV2.T_P_matrix K t i j|) ≤ ρ :=
  Q3.Proofs.RKHSContractionBridgeV2.RKHS_contraction_Q3 K hK

/-! ## 6. Q Lipschitz -/
theorem Q_Lipschitz (K : ℝ) (hK : K > 0) :
    ∀ Φ Ψ : ℝ → ℝ, Φ ∈ Q3.W_K K → Ψ ∈ Q3.W_K K →
      |Q3.Q Φ - Q3.Q Ψ| ≤ Q3.Proofs.QLipschitzBridgeV2.L_Q K *
        Q3.Proofs.QLipschitzBridgeV2.sup_norm_diff K Φ Ψ :=
  Q3.Proofs.QLipschitzBridgeV2.Q_Lipschitz_on_W_K K hK

/-! ## 7. A3 Bridge -/
theorem A3_bridge (K : ℝ) (hK : K ≥ 1) :
    ∃ ε > 0, ∀ Φ ∈ Q3.W_K K, ∀ t > 0, t < ε →
      ∃ approx : ℝ → ℝ, (∀ x ∈ Set.Icc (-K) K, |Φ x - approx x| ≤ t) :=
  Q3.Proofs.A3BridgeV2.A3_bridge_Q3 K hK

/-! ## 8. Q Nonneg on Atoms -/
theorem Q_nonneg_on_atoms (K : ℝ) (hK : K ≥ 1) :
    ∀ Φ ∈ Q3.Proofs.QNonnegBridgeV2.AtomCone K, Q3.Q Φ ≥ 0 :=
  Q3.Proofs.QNonnegBridgeV2.Q_nonneg_on_atoms_Q3 K hK

/-! ## 9. A1 Density -/
theorem A1_density (K : ℝ) (hK : K > 0) :
    ∀ Φ ∈ Q3.W_K K, ∀ ε > 0, ∃ Ψ ∈ Q3.Proofs.A1DensityBridgeV2.AtomCone K,
      ∀ x ∈ Set.Icc (-K) K, |Φ x - Ψ x| < ε :=
  Q3.Proofs.A1DensityBridgeV2.A1_density_Q3 K hK

end Q3.Clean.Theorems

/-!
# Summary

All 9 Tier-2 theorems are now available without importing Q3.Axioms!

Verification:
```
#print axioms Q3.Clean.Theorems.node_spacing
#print axioms Q3.Clean.Theorems.RKHS_contraction
```
Expected: [propext, Classical.choice, Quot.sound] (some with sorryAx)
NO Q3.* axioms!
-/

-- Verification
#print axioms Q3.Clean.Theorems.node_spacing
#print axioms Q3.Clean.Theorems.S_K_small
#print axioms Q3.Clean.Theorems.W_sum_finite
#print axioms Q3.Clean.Theorems.off_diag_exp_sum
#print axioms Q3.Clean.Theorems.RKHS_contraction
#print axioms Q3.Clean.Theorems.Q_Lipschitz
#print axioms Q3.Clean.Theorems.A3_bridge
#print axioms Q3.Clean.Theorems.Q_nonneg_on_atoms
#print axioms Q3.Clean.Theorems.A1_density
