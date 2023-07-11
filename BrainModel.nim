type
  Neuron = object
    id: int
    sensoryInput: int
    encodedData: int

var neurons: array[10, Neuron]

# Encoding process
for i in 0..<neurons.len:
  echo "Enter sensory input data (1 to 10) for neuron ", i+1, " : "
  var inputData: int = readLine(stdin).parseInt
  neurons[i].id = i
  neurons[i].sensoryInput = inputData
  neurons[i].encodedData = inputData

# Neural Plasticity: Each neuron's encoded data is influenced by its neighbours
for i in 0..<neurons.len:
  if i != 0:
    neurons[i].encodedData += neurons[i-1].encodedData
  if i != neurons.high:
    neurons[i].encodedData += neurons[i+1].encodedData

# Retrieval process
for neuron in neurons:
  echo "Retrieved data from neuron ", neuron.id+1, " is ", neuron.encodedData
