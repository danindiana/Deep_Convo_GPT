# File: BrainModel.awk
BEGIN {
    # Initialize neurons and synapses
    FS=" ";
    while (getline < "neurons.txt") {
        neuron[$1] = $2;
    }
    while (getline < "synapses.txt") {
        synapse[$1, $2] = $3;
    }
}

{
    # Update synapse strengths based on neuron activity
    for (i in neuron) {
        for (j in neuron) {
            if ((i, j) in synapse) {
                synapse[i, j] += neuron[i] * neuron[j];
            }
        }
    }
}

END {
    # Print final synapse strengths
    for (i in neuron) {
        for (j in neuron) {
            if ((i, j) in synapse) {
                print i, j, synapse[i, j];
            }
        }
    }
}
