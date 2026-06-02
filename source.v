From Coq Require Import Reals Lra Psatz.
From Coquelicot Require Import Coquelicot.
From Interval Require Import Tactic.

Definition sech (u : R) : R :=
  2 * exp (u) / (exp (2 * u) + 1).

Definition f (x : R) : R :=
    (sech (10 * x - 2))^2
  + (sech (100 * x - 40))^4
  + (sech (1000 * x - 600))^6.

Definition I : R := RInt f 0 1.

Theorem I_first_4_decimal_digits : Rabs (I - 0.2108) <= 1e-4.
Proof.
  unfold I, f, sech.
  integral with (i_prec 25, i_degree 3, i_fuel 300).
Qed.

(************************************************************)
(* Analytic computation by antiderivatives                   *)
(************************************************************)

Definition tanhE (u : R) : R :=
  (exp (2 * u) - 1) / (exp (2 * u) + 1).

Definition A2 (u : R) : R :=
  tanhE u.

Definition A4 (u : R) : R :=
  tanhE u - (/ 3) * (tanhE u)^3.

Definition A6 (u : R) : R :=
  tanhE u - (2 / 3) * (tanhE u)^3 + (/ 5) * (tanhE u)^5.

Definition F2 (x : R) : R :=
  A2 (10 * x - 2) / 10.

Definition F4 (x : R) : R :=
  A4 (100 * x - 40) / 100.

Definition F6 (x : R) : R :=
  A6 (1000 * x - 600) / 1000.

Definition F (x : R) : R :=
  F2 x + F4 x + F6 x.

Definition I_closed_form : R :=
  F 1 - F 0.

Lemma sech_denominator_nonzero (u : R) :
  exp (2 * u) + 1 <> 0.
Proof.
  apply Rgt_not_eq.
  apply Rplus_lt_0_compat.
  - apply exp_pos.
  - lra.
Qed.

Lemma tanhE_derivative (u : R) :
  is_derive tanhE u (sech u ^ 2).
Proof.
  unfold tanhE, sech.
  auto_derive.
  - apply sech_denominator_nonzero.
  - field_simplify.
    + replace (2 * u) with (u + u) by ring.
      rewrite exp_plus.
      replace (exp u * exp u) with (exp u ^ 2) by ring.
      reflexivity.
    + apply sech_denominator_nonzero.
    + apply sech_denominator_nonzero.
Qed.

Lemma sech_tanhE_identity (u : R) :
  sech u ^ 2 = 1 - tanhE u ^ 2.
Proof.
  unfold sech, tanhE.
  field_simplify.
  - replace (2 * u) with (u + u) by ring.
    rewrite exp_plus.
    replace (exp u * exp u) with (exp u ^ 2) by ring.
    ring.
  - apply sech_denominator_nonzero.
  - apply sech_denominator_nonzero.
Qed.

Lemma A2_derivative (u : R) :
  is_derive A2 u (sech u ^ 2).
Proof.
  unfold A2.
  apply tanhE_derivative.
Qed.

Lemma A4_derivative (u : R) :
  is_derive A4 u (sech u ^ 4).
Proof.
  unfold A4.
  auto_derive.
  - repeat split.
    + exists (sech u ^ 2). apply tanhE_derivative.
    + exists (sech u ^ 2). apply tanhE_derivative.
  - rewrite (is_derive_unique (fun x : R => tanhE x) u (sech u ^ 2)).
    + replace (sech u ^ 4) with ((sech u ^ 2) ^ 2) by ring.
      repeat rewrite sech_tanhE_identity.
      field.
    + apply tanhE_derivative.
Qed.

Lemma A6_derivative (u : R) :
  is_derive A6 u (sech u ^ 6).
Proof.
  unfold A6.
  auto_derive.
  - repeat split.
    + exists (sech u ^ 2); apply tanhE_derivative.
    + exists (sech u ^ 2); apply tanhE_derivative.
    + exists (sech u ^ 2); apply tanhE_derivative.
  - rewrite (is_derive_unique (fun x : R => tanhE x) u (sech u ^ 2)).
    + replace (sech u ^ 6) with ((sech u ^ 2) ^ 3) by ring.
      repeat rewrite sech_tanhE_identity.
      field.
    + apply tanhE_derivative.
Qed.

Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2).
Proof.
  unfold F2.
  auto_derive.
  - exists (sech (10 * x + - (2)) ^ 2).
    apply A2_derivative.
  - replace (Derive (fun y : R => A2 y) (10 * x + - (2)))
      with (sech (10 * x + - (2)) ^ 2).
    + replace (10 * x + - (2)) with (10 * x - 2) by ring.
      field.
    + symmetry.
      apply is_derive_unique.
      apply A2_derivative.
Qed.

Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4).
Proof.
  unfold F4.
  auto_derive.
  - exists (sech (100 * x + - (40)) ^ 4).
    apply A4_derivative.
  - replace (Derive (fun y : R => A4 y) (100 * x + - (40)))
      with (sech (100 * x + - (40)) ^ 4).
    + replace (100 * x + - (40)) with (100 * x - 40) by ring.
      field.
    + symmetry.
      apply is_derive_unique.
      apply A4_derivative.
Qed.

Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6).
Proof.
  unfold F6.
  auto_derive.
  - exists (sech (1000 * x + - (600)) ^ 6).
    apply A6_derivative.
  - replace (Derive (fun y : R => A6 y) (1000 * x + - (600)))
      with (sech (1000 * x + - (600)) ^ 6).
    + replace (1000 * x + - (600)) with (1000 * x - 600) by ring.
      field.
    + symmetry.
      apply is_derive_unique.
      apply A6_derivative.
Qed.

Lemma F_derivative (x : R) :
  is_derive F x (f x).
Proof.
  unfold F, f.
  apply (is_derive_plus 
    (fun y : R => F2 y + F4 y) F6 x
    ((sech (10 * x - 2)) ^ 2 + (sech (100 * x - 40)) ^ 4)
    ((sech (1000 * x - 600)) ^ 6)).
  - apply (is_derive_plus F2 F4 x
      ((sech (10 * x - 2)) ^ 2)
      ((sech (100 * x - 40)) ^ 4));
      [apply F2_derivative | apply F4_derivative].
  - apply F6_derivative.
Qed.

Lemma f_ex_derive (x : R) :
  ex_derive f x.
Proof.
  unfold f, sech.
  auto_derive.
  repeat split; try exact Logic.I.
  all: apply sech_denominator_nonzero.
Qed.

Lemma f_continuous (x : R) :
  continuous f x.
Proof.
  apply (ex_derive_continuous f x).
  apply f_ex_derive.
Qed.

Theorem I_closed_form_correct : I = I_closed_form.
Proof.
  unfold I, I_closed_form.
  apply is_RInt_unique.
  apply (is_RInt_derive F f 0 1).
  - intros x _. apply F_derivative.
  - intros x _. apply f_continuous.
Qed.