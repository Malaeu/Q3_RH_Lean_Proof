/-
Q3 Formalization: T5 Compact Transfer Theorem
==============================================

This file proves the T5 transfer theorem:
  Q ≥ 0 on AtomCone_K → Q ≥ 0 on all of W_K

This is a THEOREM, not an axiom! It follows from:
- A1: AtomCone_K is dense in W_K (approximation)
- A2: Q is Lipschitz on W_K (continuity)
- Atoms: Q ≥ 0 on AtomCone_K (positivity on generators)

The proof is pure analysis/topology: dense approximation + Lipschitz continuity
implies limit preservation of nonnegativity.
-/

import Q3.Basic.Defs
import Q3.Axioms
import Q3.Atoms_Positive

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Classical
open scoped Pointwise
open MeasureTheory

set_option maxHeartbeats 400000
set_option maxRecDepth 4000

noncomputable section

namespace Q3.T5

-- Open Q3 namespace for access to definitions
open Q3

/-!
## General Lemma: Nonnegativity Transfer via Dense Approximation

If:
1. D is dense in X (every x ∈ X can be approximated by d ∈ D)
2. f : X → ℝ is Lipschitz (or continuous)
3. f ≥ 0 on D

Then f ≥ 0 on X.

This is standard real analysis.
-/

/-- Auxiliary: AtomCone_K is a subset of W_K (by definition) -/
lemma AtomCone_subset_W_K (K : ℝ) : AtomCone_K K ⊆ W_K K := by
  intro g hg
  rcases hg with ⟨n, c, B, t, τ, -, -, -, -, -, -, hg_in_W_K⟩
  exact hg_in_W_K

/-!
## T5 Transfer Theorem

The main result: transfer positivity from atoms to all of W_K.
-/

/-- **T5 Transfer Theorem**

If Q ≥ 0 on AtomCone_K (the generating cone), then Q ≥ 0 on all of W_K.

Proof idea:
1. Take any Φ ∈ W_K
2. By A1, get sequence g_n ∈ AtomCone_K with g_n → Φ uniformly
3. By A2, Q is Lipschitz, so Q(g_n) → Q(Φ)
4. By Atoms axiom, Q(g_n) ≥ 0 for all n
5. Limit of nonnegatives is nonnegative: Q(Φ) ≥ 0
-/
theorem T5_transfer (K : ℝ) (hK : K ≥ 1) :
    ∀ Φ ∈ W_K K, Q Φ ≥ 0 := by
  intro Φ hΦ
  -- We'll prove by contradiction: assume Q(Φ) < 0 and derive contradiction
  by_contra h_neg
  push_neg at h_neg
  -- So Q(Φ) < 0
  set δ := -Q Φ with hδ_def
  have hδ_pos : δ > 0 := by linarith

  -- Get Lipschitz constant from A2
  have hK_pos : K > 0 := by linarith
  obtain ⟨L, hL_pos, hLip⟩ := Q_Lipschitz_on_W_K K hK_pos

  -- Choose ε small enough: ε < δ/(2L)
  set ε := δ / (2 * L) with hε_def
  have hε_pos : ε > 0 := by positivity

  -- By A1, get approximant g ∈ AtomCone_K with ||Φ - g||_∞ < ε
  obtain ⟨g, hg_atom, hg_approx⟩ := A1_density_WK_axiom K hK_pos Φ hΦ ε hε_pos

  -- g ∈ AtomCone_K ⊆ W_K
  have hg_W_K : g ∈ W_K K := AtomCone_subset_W_K K hg_atom

  -- By Atoms axiom, Q(g) ≥ 0
  have hg_nonneg : Q g ≥ 0 := Q3.Atoms.Q_nonneg_on_atoms K hK g hg_atom

  -- By Lipschitz: |Q(Φ) - Q(g)| ≤ L * ||Φ - g||_∞ < L * ε = δ/2
  have h_diff_bound : |Q Φ - Q g| ≤ L * sSup {|Φ x - g x| | x ∈ Set.Icc (-K) K} :=
    hLip Φ hΦ g hg_W_K

  -- The sup is < ε
  have h_sup_small : sSup {|Φ x - g x| | x ∈ Set.Icc (-K) K} < ε := hg_approx

  -- So |Q(Φ) - Q(g)| < L * ε = δ/2
  have h_diff_lt : |Q Φ - Q g| < L * ε := by
    have hmul : L * sSup {|Φ x - g x| | x ∈ Set.Icc (-K) K} < L * ε :=
      mul_lt_mul_of_pos_left h_sup_small hL_pos
    exact lt_of_le_of_lt h_diff_bound hmul

  have h_Lε : L * ε = δ / 2 := by
    have hL_ne : (L : ℝ) ≠ 0 := ne_of_gt hL_pos
    calc
      L * ε = L * (δ / (2 * L)) := by simp [hε_def]
      _ = δ / 2 := by
        field_simp [hL_ne]

  -- So |Q(Φ) - Q(g)| < δ/2
  rw [h_Lε] at h_diff_lt

  -- Now derive contradiction:
  -- Q(Φ) = -δ, Q(g) ≥ 0, but |Q(Φ) - Q(g)| < δ/2
  -- This means Q(g) - Q(Φ) < δ/2 (using |a - b| < c implies a - b < c)
  -- So Q(g) < Q(Φ) + δ/2 = -δ + δ/2 = -δ/2 < 0
  -- But Q(g) ≥ 0, contradiction!

  have h_Q_Φ : Q Φ = -δ := by simp [hδ_def]

  -- From |Q(Φ) - Q(g)| < δ/2, get Q(g) - Q(Φ) < δ/2, hence Q(g) < Q(Φ) + δ/2.
  have h_Qg_minus_QΦ : Q g - Q Φ < δ / 2 :=
    (abs_sub_lt_iff.mp h_diff_lt).2
  have h_Qg_lt : Q g < Q Φ + δ / 2 := by linarith

  -- Substitute Q(Φ) = -δ to conclude Q(g) < -δ/2 < 0, contradicting Q(g) ≥ 0.
  have h_Qg_lt_neg_half : Q g < -(δ / 2) := by
    have : Q Φ + δ / 2 = -(δ / 2) := by
      -- Q Φ = -δ
      rw [h_Q_Φ]
      ring
    simpa [this] using h_Qg_lt
  have h_neg' : Q g < 0 := by
    have : -(δ / 2) < 0 := by linarith [hδ_pos]
    exact lt_trans h_Qg_lt_neg_half this
  linarith

/-- Corollary: Q is nonnegative on W_K for K ≥ 1 -/
theorem Q_nonneg_on_W_K (K : ℝ) (hK : K ≥ 1) :
    ∀ Φ ∈ W_K K, Q Φ ≥ 0 :=
  T5_transfer K hK

/-- W_K with continuity implies Weil_cone_K membership + continuity -/
lemma W_K_subset_Weil_cone_K_with_cont (K : ℝ) (Φ : ℝ → ℝ) (hΦ : Φ ∈ W_K K) :
    Φ ∈ Weil_cone_K K ∧ ContinuousOn Φ (Set.Icc (-K) K) := by
  obtain ⟨hcont, hsupp, heven, hnonneg⟩ := hΦ
  exact ⟨⟨heven, hnonneg, hsupp⟩, hcont⟩

end Q3.T5

end
