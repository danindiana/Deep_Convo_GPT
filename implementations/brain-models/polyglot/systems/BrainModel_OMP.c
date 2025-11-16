// File: BrainModel_OMP.c
#include <stdio.h>
#include <omp.h>

#define NUM_MEMORIES 10000

// The "brain" is represented as an array of string memories
char brain[NUM_MEMORIES][256];

// The "encoding" process simply copies a memory into the brain
void encode(int memory_index, char* memory) {
    strcpy(brain[memory_index], memory);
}

// The "retrieval" process prints a memory from the brain
void retrieve(int memory_index) {
    printf("Memory %d: %s\n", memory_index, brain[memory_index]);
}

int main() {
    // Encoding memories in parallel
    #pragma omp parallel for
    for (int i = 0; i < NUM_MEMORIES; i++) {
        char memory[256];
        sprintf(memory, "This is memory %d", i);
        encode(i, memory);
    }

    // Retrieving memories in parallel
    #pragma omp parallel for
    for (int i = 0; i < NUM_MEMORIES; i++) {
        retrieve(i);
    }

    return 0;
}
