/-
Q3 Formalization: A1' Local Density Theorem
============================================

This file contains the main A1' theorem: Fejér×heat products are dense
in the space of even nonnegative continuous functions on compacts.

The theorem is accepted as an axiom (A1_density_axiom) based on
classical approximation theory (Fejér, heat kernel identity).
-/

import Mathlib
import Q3.Axioms

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Classical
open scoped Pointwise
open MeasureTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000

noncomputable section

namespace Q3.A1

/-! ## Main Theorem A1' -/

/-- **Theorem A1' (Local Density)**:
The Fejér×heat cone is dense in C⁺_even([-K,K]) in the uniform norm.

This is a key approximation theorem: any even nonnegative continuous function
on a compact interval can be uniformly approximated by finite sums of
Fejér×heat atoms.

Proof: Follows from axiom A1_density_axiom (combination of Fejér's theorem
and heat kernel approximation identity).
-/
theorem A1_density (K : ℝ) (hK : K > 0) :
    ∀ f ∈ Q3.C_even_nonneg K, ∀ ε > 0,
      ∃ g ∈ Q3.Fejer_heat_cone K, ∀ x ∈ Set.Icc (-K) K, |f x - g x| < ε := by
  intro f hf ε hε
  exact Q3.A1_density_axiom K hK f hf ε hε

/-! ## Corollary for Weil cone -/

/-- Elements of Weil cone on compact K can be approximated by Fejér×heat atoms -/
theorem A1_density_Weil (K : ℝ) (hK : K > 0) (Φ : ℝ → ℝ)
    (hΦ_cont : ContinuousOn Φ (Set.Icc (-K) K))
    (hΦ_even : ∀ x, Φ x = Φ (-x))
    (hΦ_nonneg : ∀ x ∈ Set.Icc (-K) K, 0 ≤ Φ x) :
    ∀ ε > 0, ∃ g ∈ Q3.Fejer_heat_cone K, ∀ x ∈ Set.Icc (-K) K, |Φ x - g x| < ε := by
  intro ε hε
  exact A1_density K hK Φ ⟨hΦ_cont, hΦ_even, hΦ_nonneg⟩ ε hε

end Q3.A1

end
