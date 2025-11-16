struct Neuron
    id::Int
    encodedData::Float64
end

function encode!(neuron::Neuron, sensoryInput::Float64)
    neuron.encodedData = sensoryInput
end

function retrieve(neuron::Neuron)
    return neuron.encodedData
end

function apply_plasticity!(neurons::Vector{Neuron})
    for i in 2:length(neurons)
        neurons[i].encodedData += neurons[i-1].encodedData
    end
end

# Creating neurons
neurons = [Neuron(i, 0.0) for i in 1:10]

# Encoding process
for i in 1:10
    println("Enter sensory input data (1 to 10) for neuron ", i, ": ")
    inputData = parse(Float64, readline())
    encode!(neurons[i], inputData)
end

# Applying Neural Plasticity: Each neuron's encoded data is influenced by its neighbours
apply_plasticity!(neurons)

# Retrieval process
for neuron in neurons
    println("Retrieved data from neuron ", neuron.id, " is ", retrieve(neuron))
end
