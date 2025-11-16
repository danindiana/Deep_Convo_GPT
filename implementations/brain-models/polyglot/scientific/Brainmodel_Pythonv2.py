# File: BrainModel.py
import random

class Neuron:
    def __init__(self):
        self.connections = []

    def connect(self, neuron, weight):
        self.connections.append((neuron, weight))

    def process(self, signal):
        return sum(neuron.process(signal) * weight
                   for neuron, weight in self.connections)

class Brain:
    def __init__(self, size):
        self.neurons = [Neuron() for _ in range(size)]
        for neuron in self.neurons:
            for _ in range(random.randint(1, size // 2)):
                target = random.choice(self.neurons)
                weight = random.random()
                neuron.connect(target, weight)

    def encode(self, input_data):
        # The sensory input is simply directed to a random neuron
        target = random.choice(self.neurons)
        return target.process(input_data)

    def retrieve(self, encoded_data):
        # The retrieval process is a matter of reverse-processing the signal
        return sum(neuron.process(encoded_data) for neuron in self.neurons)

brain = Brain(1000)

# Sensory input is represented as a list of floats
sensory_input = [random.random() for _ in range(100)]

# The encoding process
encoded = brain.encode(sensory_input)

# The retrieval process
retrieved = brain.retrieve(encoded)

print('Sensory input:', sensory_input)
print('Encoded:', encoded)
print('Retrieved:', retrieved)
