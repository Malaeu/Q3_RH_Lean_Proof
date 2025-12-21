/-
S_K Small Bridge
================

This file creates a REAL bridge between Aristotle's S_K_small proof
and Q3.Basic.Defs definitions.

Strategy:
1. Import Q3.Basic.Defs for Q3 definitions
2. Import Aristotle's proof (with its own definitions)
3. Show definitions are equal (delta_K, S_K, t_min)
4. Transfer the theorem

CLOSES: S_K_small_axiom
-/

import Q3.Basic.Defs
import Q3.Proofs.S_K_small

set_option linter.mathlibStandardSet false
set_option linter.unusedVariables false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option synthInstance.maxHeartbeats 20000
set_option synthInstance.maxSize 128

noncomputable section

namespace Q3.Proofs.S_K_SmallBridge

/-! ## Aristotle Definitions (from S_K_small.lean)

These are defined in the root namespace by Aristotle:
- `N_K (K : ℝ) : ℕ := Nat.floor (Real.exp (2 * Real.pi * K))`
- `delta_K (K : ℝ) : ℝ := 1 / (2 * Real.pi * (N_K K + 1))`
- `t_min (K η : ℝ) : ℝ := (delta_K K)^2 / (4 * Real.log (2/η + 1))`
- `S_K (K t : ℝ) : ℝ := 2 * exp(-delta_K²/(4t)) / (1 - exp(-delta_K²/(4t)))`
-/

/-! ## Definition Equivalences -/

/-- delta_K definitions are equal -/
lemma delta_K_eq (K : ℝ) : _root_.delta_K K = Q3.delta_K K := by
  unfold _root_.delta_K _root_.N_K Q3.delta_K
  rfl

/-- S_K definitions are equal (given delta_K equality) -/
lemma S_K_eq (K t : ℝ) : _root_.S_K K t = Q3.S_K K t := by
  unfold _root_.S_K Q3.S_K
  rw [delta_K_eq]

/-- t_min definitions are algebraically equal:
    Aristotle: log(2/η + 1) = log((2 + η)/η)
    Q3: log((2 + η)/η)
-/
lemma t_min_eq (K η : ℝ) (hη : η > 0) : _root_.t_min K η = Q3.t_min K η := by
  unfold _root_.t_min Q3.t_min
  rw [delta_K_eq]
  congr 1
  -- Show log(2/η + 1) = log((2 + η)/η)
  congr 1
  field_simp

/-! ## BRIDGE: Transfer S_K_small theorem to Q3 definitions -/

/-- S_K SMALL THEOREM with Q3 definitions (CLOSES S_K_small_axiom) -/
theorem S_K_small_Q3 (K t η : ℝ) (hK : K ≥ 1) (hη : η > 0) (ht : t ≤ Q3.t_min K η) :
    Q3.S_K K t ≤ η := by
  -- Rewrite t constraint using Aristotle's t_min
  rw [← t_min_eq K η hη] at ht
  -- Rewrite goal using Aristotle's S_K
  rw [← S_K_eq]
  -- Apply Aristotle's theorem
  exact S_K_small K t η hK hη ht

end Q3.Proofs.S_K_SmallBridge

/-!
# Summary

This file provides a REAL bridge:
1. Showed delta_K definitions are equal (via rfl)
2. Showed S_K definitions are equal (using delta_K equality)
3. Showed t_min definitions are algebraically equal (2/η + 1 = (2+η)/η)
4. Transferred S_K_small theorem to Q3 definitions

S_K_small_Q3 CLOSES S_K_small_axiom without circular reference!

# Verification
To verify this is clean:
```
lake env lean -c "import Q3.Proofs.S_K_small_bridge; #print axioms Q3.Proofs.S_K_SmallBridge.S_K_small_Q3"
```
Expected: only `propext`, `Classical.choice`, `Quot.sound`
-/
