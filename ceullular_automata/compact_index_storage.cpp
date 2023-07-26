#include <iostream>
#include <vector>

const int INDEX_SIZE = 100; // Number of cells in the index

class CellularAutomaton {
private:
    std::vector<bool> index;

public:
    CellularAutomaton() {
        index = std::vector<bool>(INDEX_SIZE, false);
    }

    // Update the index based on cellular automaton rules
    void updateIndex() {
        // Implement your cellular automaton rules here
        // For demonstration purposes, we'll set all cells to true if their index is divisible by 3.
        for (int i = 0; i < INDEX_SIZE; i++) {
            index[i] = (i % 3 == 0);
        }
    }

    // Convert the index to a compact representation using RLE
    std::vector<std::pair<bool, int>> compressIndex() {
        std::vector<std::pair<bool, int>> compressed_index;

        bool current_value = index[0];
        int count = 1;

        for (int i = 1; i < INDEX_SIZE; i++) {
            if (index[i] == current_value) {
                count++;
            } else {
                compressed_index.push_back(std::make_pair(current_value, count));
                current_value = index[i];
                count = 1;
            }
        }

        // Add the last run
        compressed_index.push_back(std::make_pair(current_value, count));

        return compressed_index;
    }

    // Print the compact representation of the index
    void printCompressedIndex(const std::vector<std::pair<bool, int>>& compressed_index) {
        for (const auto& run : compressed_index) {
            std::cout << "[" << (run.first ? "1" : "0") << ", " << run.second << "] ";
        }
        std::cout << std::endl;
    }
};

int main() {
    CellularAutomaton automaton;

    // Update the index
    automaton.updateIndex();

    // Convert and print the compact representation of the index
    std::vector<std::pair<bool, int>> compressed_index = automaton.compressIndex();
    automaton.printCompressedIndex(compressed_index);

    return 0;
}
