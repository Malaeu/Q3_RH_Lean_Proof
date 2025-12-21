/-
Q3 Proofs Integrated
====================

This file imports all integrated proofs and re-exports the closing theorems.
These theorems close all Tier-2 axioms from Q3.Axioms.

After importing this file, run:
  #print axioms Q3.Main.RH_of_Weil_and_Q3
to verify only Tier-1 axioms remain.
-/

-- Import all integrated proofs
import Q3.Proofs.A1_density_integrated
import Q3.Proofs.A3_bridge_integrated
import Q3.Proofs.Q_Lipschitz_integrated
import Q3.Proofs.Q_nonneg_on_atoms_integrated
import Q3.Proofs.RKHS_contraction_integrated
import Q3.Proofs.S_K_small_integrated
import Q3.Proofs.W_sum_finite_integrated
import Q3.Proofs.node_spacing_integrated
import Q3.Proofs.off_diag_exp_sum_integrated

namespace Q3.ProofsIntegrated

/-! ## Re-export Closing Theorems

These theorems have the same type as their corresponding axioms,
proving that all Tier-2 axioms are closed.
-/

-- A1 Density (closes A1_density_WK_axiom)
#check Q3.Proofs.A1_Density.closes_A1_density_axiom

-- A3 Bridge (closes A3_bridge_axiom)
#check Q3.Proofs.A3_Bridge.closes_A3_bridge_axiom

-- Q Lipschitz (closes Q_Lipschitz_on_W_K)
#check Q3.Proofs.Q_Lipschitz.closes_Q_Lipschitz_axiom

-- Q Nonneg on Atoms (closes Q_nonneg_on_atoms_of_A3_RKHS_axiom)
#check Q3.Proofs.Q_Nonneg.closes_Q_nonneg_axiom

-- RKHS Contraction (closes RKHS_contraction_axiom)
#check Q3.Proofs.RKHS_Contraction.closes_RKHS_axiom

-- S_K Small (closes S_K_small_axiom)
#check Q3.Proofs.S_K_Small.closes_S_K_small_axiom

-- W Sum Finite (closes W_sum_finite_axiom)
#check Q3.Proofs.W_sum.closes_W_sum_axiom

-- Node Spacing (closes node_spacing_axiom)
#check Q3.Proofs.NodeSpacing.node_spacing

-- Off Diagonal Sum (closes off_diag_exp_sum_axiom)
#check Q3.Proofs.OffDiagExpSum.closes_off_diag_axiom

end Q3.ProofsIntegrated
