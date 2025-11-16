' Filename: BrainModel.bas

DIM SensoryInput(10)
DIM EncodedData(10)
DIM i

' Sensory input and encoding
FOR i = 1 TO 10
    PRINT "Enter sensory input data (1 to 10) for neuron"; i;
    INPUT SensoryInput(i)
    EncodedData(i) = SensoryInput(i) ' Here we're just storing the data, but in a real model this could be a complex process.
NEXT i

' Retrieval
FOR i = 1 TO 10
    PRINT "Retrieved data from neuron"; i; "is"; EncodedData(i)
NEXT i
