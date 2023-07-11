Neuron <- setRefClass(
  "Neuron",
  fields = list(
    id = "integer",
    encodedData = "numeric"
  ),
  methods = list(
    encode = function(sensoryInput) {
      encodedData <<- sensoryInput
    },
    retrieve = function() {
      return(encodedData)
    }
  )
)

neurons <- list()
for (i in 1:10) {
  neurons[[i]] <- Neuron$new(id=i, encodedData=0.0)
}

cat("Encoding process\n")
for (i in 1:10) {
  cat("Enter sensory input data (1 to 10) for neuron ", i, ": ")
  inputData <- as.numeric(readline())
  neurons[[i]]$encode(inputData)
}

cat("Retrieval process\n")
for (i in 1:10) {
  cat("Retrieved data from neuron ", i, " is ", neurons[[i]]$retrieve(), "\n")
}
