// File: BrainModel.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

class Neuron {
    vector<pair<Neuron*, double>> connections;

public:
    void connect(Neuron* neuron, double weight) {
        connections.push_back(make_pair(neuron, weight));
    }

    double process(double signal) {
        double result = 0.0;
        for (auto& conn : connections) {
            result += conn.second * conn.first->process(signal);
        }
        return result;
    }
};

class Brain {
    vector<Neuron> neurons;

public:
    Brain(int size) {
        srand(time(0));
        neurons = vector<Neuron>(size);
        for (int i = 0; i < size; ++i) {
            int connections = rand() % (size / 2);
            for (int j = 0; j < connections; ++j) {
                int target = rand() % size;
                double weight = (double)rand() / RAND_MAX;
                neurons[i].connect(&neurons[target], weight);
            }
        }
    }

    double encode(double input_data) {
        int target = rand() % neurons.size();
        return neurons[target].process(input_data);
    }

    double retrieve(double encoded_data) {
        double result = 0.0;
        for (auto& neuron : neurons) {
            result += neuron.process(encoded_data);
        }
        return result;
    }
};

int main() {
    Brain brain(1000);

    double sensory_input = (double)rand() / RAND_MAX;

    double encoded = brain.encode(sensory_input);

    double retrieved = brain.retrieve(encoded);

    cout << "Sensory input: " << sensory_input << '\n';
    cout << "Encoded: " << encoded << '\n';
    cout << "Retrieved: " << retrieved << '\n';

    return 0;
}
