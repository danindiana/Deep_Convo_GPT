# BrainModel.awk
BEGIN {
  FS = " ";   # Field separator is whitespace
}

# Encoding function: processes input and stores it as a memory
function encode(memory) {
  perception["memories"][NR] = memory;
}

# Retrieval function: searches for a memory and prints it if found
function retrieve(memory) {
  for (m in perception["memories"]) {
    if (perception["memories"][m] == memory) {
      print "Memory retrieved: " perception["memories"][m]
    }
  }
}

# Pass each line of input to the encode function
{
  encode($0)
}

# After all memories have been stored, retrieve some memories
END {
  retrieve("memory1")
  retrieve("memory2")
}
