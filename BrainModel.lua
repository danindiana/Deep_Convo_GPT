Neuron = {}
Neuron.__index = Neuron

function Neuron.create(id, input)
  local n = {}
  setmetatable(n, Neuron)
  n.id = id
  n.sensoryInput = input
  n.encodedData = input
  return n
end

neurons = {}
for i = 1, 10 do
  io.write("Enter sensory input data (1 to 10) for neuron " .. i .. ": ")
  local inputData = io.read("*n")
  neurons[i] = Neuron.create(i, inputData)
end

-- Neural Plasticity: Each neuron's encoded data is influenced by its neighbours
for i = 2, #neurons do
  neurons[i].encodedData = neurons[i].encodedData + neurons[i - 1].encodedData
end

-- Retrieval process
for _, neuron in ipairs(neurons) do
  print("Retrieved data from neuron " .. neuron.id .. " is " .. neuron.encodedData)
end
