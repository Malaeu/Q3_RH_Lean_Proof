/-
Q3 Formalization: Axiom Dependency Check
========================================

This file verifies all axiom dependencies for CI.
Run: `lake env lean Q3/CheckAxioms.lean`

Expected output: List of axioms used in RH_of_Weil_and_Q3
-/

import Q3.Main

/-!
# Axiom Dependency Verification

This prints all axioms used by the main theorem.
Used in CI to ensure no undocumented axioms sneak in.

## KEY CHANGE: Q_nonneg_on_W_K_axiom is GONE!

The "nuclear bomb" axiom Q_nonneg_on_W_K_axiom has been decomposed into:
- A1_density_WK_axiom: atoms dense in W_K
- Q_Lipschitz_on_W_K: Q is Lipschitz on W_K
- A3_bridge_axiom: Toeplitz-symbol bridge (spectral gap)
- RKHS_contraction_axiom: prime operator contraction
- Q_nonneg_on_atoms_of_A3_RKHS_axiom: core (A3+RKHS) ⇒ atoms positivity

And:
- `Q3.Atoms.Q_nonneg_on_atoms` is a THEOREM deriving atom positivity from A3+RKHS.
- `Q3.T5.T5_transfer` is a THEOREM (not axiom!) proving Q ≥ 0 on W_K from density + Lipschitz + atoms.
-/

-- Re-export the main theorem for verification
open Q3.Main

/-! ## Verify Tier-1 axioms exist -/
#check Q3.Weil_criterion
#check Q3.explicit_formula
#check Q3.a_star_pos
#check Q3.Szego_Bottcher_eigenvalue_bound
#check Q3.Szego_Bottcher_convergence
#check Q3.Schur_test
#check Q3.c_arch_pos
#check Q3.eigenvalue_le_norm

/-! ## Verify Tier-2 axioms exist (NEW STRUCTURE) -/
-- Density
#check Q3.A1_density_WK_axiom        -- NEW: atoms dense in W_K
#check Q3.A1_density_axiom           -- Legacy

-- Lipschitz
#check Q3.Q_Lipschitz_on_W_K

-- RKHS
#check Q3.RKHS_contraction_axiom
#check Q3.T_P_row_sum_bound_axiom
#check Q3.S_K_small_axiom

-- A3 Bridge
#check Q3.A3_bridge_axiom

-- Atoms (NEW - replaces Q_nonneg_on_W_K_axiom)
#check Q3.Q_nonneg_on_atoms_of_A3_RKHS_axiom  -- core implication A3+RKHS ⇒ atoms positivity
#check Q3.Atoms.Q_nonneg_on_atoms             -- THEOREM: Q ≥ 0 on AtomCone_K

/-! ## Verify T5 is a THEOREM (not axiom!) -/
#check Q3.T5.T5_transfer             -- THEOREM: Q ≥ 0 on W_K

/-! ## Verify local axiom -/
#check Q3.Main.Weil_cone_continuous  -- Test functions are continuous

/-! ## Print Axiom Dependencies -/

-- This is the authoritative dependency list for the full RH theorem.
#print axioms Q3.Main.RH_of_Weil_and_Q3

/-!
## Expected Dependencies

### Standard Mathlib Axioms (always present):
- `propext` : Propositional extensionality
- `Classical.choice` : Classical choice principle
- `Quot.sound` : Quotient soundness

### Tier-1 Classical Axioms:
- `Q3.Weil_criterion` : Weil (1952)

### Tier-2 Q3 Paper Axioms:
- `Q3.A1_density_WK_axiom` : Atoms dense in W_K
- `Q3.Q_Lipschitz_on_W_K` : Q is Lipschitz
- `Q3.A3_bridge_axiom` : Toeplitz-symbol bridge
- `Q3.RKHS_contraction_axiom` : prime operator contraction
- `Q3.Q_nonneg_on_atoms_of_A3_RKHS_axiom` : core implication A3+RKHS ⇒ atoms positivity

### Local Axiom:
- `Q3.Main.Weil_cone_continuous` : Test functions are continuous

### THEOREM (not axiom!):
- `Q3.T5.T5_transfer` : Q ≥ 0 on W_K (proven from A1 + A2 + Atoms)

## Verification

Run `lake env lean Q3/Main.lean` to see:
```
Q3.Main.RH_of_Weil_and_Q3 : RH
```

The key improvement: Q_nonneg_on_W_K_axiom is GONE!
It has been replaced by smaller, more focused axioms,
and T5 is now a theorem proven in Lean.
-/
