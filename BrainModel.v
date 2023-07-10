Inductive Neuron : Type :=
| neuron : Neuron.

Definition Connection := Neuron * Neuron.

Definition network := list Connection.

Definition isConnected (n1 n2 : Neuron) (net : network) :=
  In (n1, n2) net \/ In (n2, n1) net.

Lemma reflexivity_of_connection : forall (n : Neuron) (net : network), 
  isConnected n n net.
Proof.
  intros. unfold isConnected. left. simpl. reflexivity.
Qed.
