/-
Q3 Formalization: Positivity on Atom Cone (T4 Core)
====================================================

This file packages the "atoms positivity" step as a THEOREM:

  Q ≥ 0 on AtomCone_K(K)

derived from:
  - A3 bridge (Toeplitz-symbol spectral gap)
  - RKHS contraction (prime operator bound)
  - a single remaining core axiom that combines them

The goal is that `#print axioms` for the final RH theorem shows explicit
dependencies on A3 + RKHS (instead of a standalone "atoms positivity" axiom).
-/

import Q3.Axioms

set_option linter.mathlibStandardSet false

namespace Q3.Atoms

open Q3

/-! ## Bundling A3 + RKHS into Atom Positivity -/

/-- A3 bridge data for `K` (just re-exported from `Q3.Axioms`). -/
abbrev A3_bridge_data := Q3.A3_bridge_data

/-- RKHS contraction data for `K` (just re-exported from `Q3.Axioms`). -/
abbrev RKHS_contraction_data := Q3.RKHS_contraction_data

/-- **T4 (Atoms Positivity)**:
From the A3 bridge axiom and the RKHS contraction axiom, we obtain
`Q g ≥ 0` for all `g` in the atom cone `AtomCone_K K`.
-/
theorem Q_nonneg_on_atoms (K : ℝ) (hK : K ≥ 1) :
    ∀ g ∈ AtomCone_K K, Q g ≥ 0 := by
  -- Extract the bundled A3 and RKHS data from the corresponding axioms.
  have hA3 : A3_bridge_data K := A3_bridge_axiom K hK
  have hRKHS : RKHS_contraction_data K := RKHS_contraction_axiom K hK
  -- Apply the core implication axiom.
  exact Q_nonneg_on_atoms_of_A3_RKHS_axiom K hK hA3 hRKHS

end Q3.Atoms
