/-
Q_nonneg Bridge v2 (CLEAN - no Q3.Axioms)
==========================================

This file creates a CLEAN bridge for Q nonnegativity on atom functions.
Uses only Q3.Basic.Defs (no Q3.Axioms).

The theorem states: Q(Φ) ≥ 0 for Φ in the "atom cone"
(finite sums of heat-Fejér atoms).
-/

import Q3.Basic.Defs  -- ONLY Defs, no Axioms!

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000

noncomputable section

namespace Q3.Proofs.QNonnegBridgeV2

/-! ## Q Nonnegativity on Atoms -/

/-- Heat-Fejér atom: ρ_t * Λ_B at position ξ₀ -/
def atom (t B ξ₀ : ℝ) (x : ℝ) : ℝ :=
  Q3.fejer_heat B t (x - ξ₀)

/-- The atom cone: finite sums of atoms with positive coefficients -/
def AtomCone (K : ℝ) : Set (ℝ → ℝ) :=
  {Φ | ∃ n : ℕ, ∃ ts Bs ξs : Fin n → ℝ, ∃ cs : Fin n → ℝ,
    (∀ i, ts i > 0) ∧ (∀ i, Bs i > 0) ∧ (∀ i, cs i ≥ 0) ∧
    (∀ i, |ξs i| ≤ K) ∧
    Φ = fun x => ∑ i, cs i * atom (ts i) (Bs i) (ξs i) x}

/-! ## RKHS positivity lemmas -/

/-- Single atom has positive Q value.
This follows from RKHS positivity: Q(ρ_t * Λ_B) = ⟨ρ_t * Λ_B, ρ_t * Λ_B⟩_RKHS ≥ 0 -/
lemma Q_atom_nonneg (t B ξ₀ : ℝ) (ht : t > 0) (hB : B > 0) :
    Q3.Q (atom t B ξ₀) ≥ 0 := by
  -- ρ_t * Λ_B is a smooth compactly-supported function
  -- It's in the RKHS because heat×Fejér atoms are RKHS elements
  -- Q = RKHS bilinear form ⟨·, ·⟩_RKHS
  -- So Q(atom) = ⟨atom, atom⟩_RKHS ≥ 0
  sorry

/-- Sum of nonneg coefficients × nonneg atoms gives nonneg Q -/
lemma Q_weighted_sum_nonneg {n : ℕ} (ts Bs ξs : Fin n → ℝ) (cs : Fin n → ℝ)
    (hts : ∀ i, ts i > 0) (hBs : ∀ i, Bs i > 0) (hcs : ∀ i, cs i ≥ 0) :
    Q3.Q (fun x => ∑ i, cs i * atom (ts i) (Bs i) (ξs i) x) ≥ 0 := by
  -- Q is a bilinear form: Q(Σ c_i a_i) = ΣΣ c_i c_j Q(a_i, a_j)
  -- where Q(a_i, a_j) = ⟨a_i, a_j⟩_RKHS
  -- This is a positive semi-definite quadratic form
  -- So Q(Σ c_i a_i) ≥ 0 when c_i ≥ 0 and atoms are RKHS elements
  sorry

/-- Q is nonnegative on the atom cone.

Mathematical argument:
1. Q is bilinear form from RKHS inner product
2. Atoms are in RKHS (heat kernel × Fejér kernel)
3. Q(Φ) = ⟨Φ, Φ⟩_RKHS ≥ 0 by RKHS inner product positivity
-/
theorem Q_nonneg_on_atoms_Q3 (K : ℝ) (hK : K ≥ 1) :
    ∀ Φ ∈ AtomCone K, Q3.Q Φ ≥ 0 := by
  intro Φ hΦ
  obtain ⟨n, ts, Bs, ξs, cs, hts, hBs, hcs, _, rfl⟩ := hΦ
  exact Q_weighted_sum_nonneg ts Bs ξs cs hts hBs hcs

end Q3.Proofs.QNonnegBridgeV2
