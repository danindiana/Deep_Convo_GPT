\ Define some constants for our simulation
1 CONSTANT firing
0 CONSTANT resting

\ Define a word to simulate a neuron
: neuron ( n -- n' )
  DUP 2 < IF drop resting ELSE drop firing THEN ;

\ Define a word to simulate a layer of neurons
: layer ( n1 n2 ... -- n1' n2' ... )
  BEGIN DUP 0 > WHILE 
    OVER neuron SWAP 1- SWAP 
  REPEAT drop ;

\ Simulate a small network of neurons
3 layer . \ Prints: 0 (resting)
4 layer . \ Prints: 1 (firing)
