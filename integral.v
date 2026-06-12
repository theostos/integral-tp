From Coq Require Import Reals Lra Psatz.
From Coquelicot Require Import Coquelicot.
From Interval Require Import Tactic Plot.

Definition sech (u : R) : R :=
  2 * exp (u) / (exp (2 * u) + 1).

Definition f (x : R) : R :=
    (sech (10 * x - 2))^2
  + (sech (100 * x - 40))^4
  + (sech (1000 * x - 600))^6.

Definition I : R := RInt f 0 1.

Do integral
  ltac:(let J := eval cbv [I f sech] in I in exact J)
  with (i_prec 25, i_degree 3, i_fuel 300,
        i_width (-15), i_decimal).

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
  exp u + 1 <> 0.
Proof.
apply Rgt_not_eq with (r1 := ((exp (u)) + 1)) (r2 := 0).
apply Rplus_lt_0_compat with (r1 := (exp (u))) (r2 := 1).
apply exp_pos.
lra.
Qed.

Lemma F2_derivative (x : R) :
  is_derive F2 x ((sech (10 * x - 2)) ^ 2).
Proof.
  unfold F2, A2, sech, tanhE.
  auto_derive.
simpl.
apply sech_denominator_nonzero.
simpl.
replace (10 * x + - (2)) with (10 * x - 2) by ring.
replace (2 * (10 * x - 2)) with ((10 * x - 2) + (10 * x - 2)) by ring.
repeat rewrite exp_plus.
field; nra.
Qed.

Lemma F4_derivative (x : R) :
  is_derive F4 x ((sech (100 * x - 40)) ^ 4).
Proof.
  unfold F4, A4, sech, tanhE.
  auto_derive.
  - repeat split.
    - apply sech_denominator_nonzero.
    - apply sech_denominator_nonzero.
  simpl.
  replace (100*x + - (40)) with (100* x - 40) by ring.
  replace (2 * (100 * x - 40)) with ((100 * x - 40) + (100 * x - 40)) by ring.
  rewrite exp_plus.
  field.
  nra.
Qed.

Lemma F6_derivative (x : R) :
  is_derive F6 x ((sech (1000 * x - 600)) ^ 6).
Proof.
  unfold F6, A6, sech, tanhE.
  auto_derive.
  repeat split.
  apply sech_denominator_nonzero.
  apply sech_denominator_nonzero.
  apply sech_denominator_nonzero.
  trivial.
  simpl.
  replace (1000 * x + - (600)) with (1000 * x - 600) by ring.
  replace (2 * (1000 * x - 600)) with ((1000 * x - 600) + (1000 * x - 600)) by ring.
  rewrite exp_plus.
  field.
  nra.
Qed.


Lemma F_derivative (x : R) :
  is_derive F x (f x).
Proof.
  unfold F, f.
  apply is_derive_plus with (f := fun x0 => ((F2 x0) + (F4 x0))) (g := F6).
- apply is_derive_plus with (f := F2) (g := F4).
  + apply F2_derivative.
  + apply F4_derivative.
- apply F6_derivative.
Qed.

Lemma f_ex_derive (x : R) :
  ex_derive f x.
Proof.
  unfold f, sech.
  auto_derive.
  repeat split.
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
