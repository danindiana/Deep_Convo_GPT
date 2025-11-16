#include <iostream>
#include <vector>

class CellularAutomaton {
private:
    std::vector<bool> index;
    std::vector<bool> compressed_index;

public:
    CellularAutomaton() {
        index = std::vector<bool>();
        compressed_index = std::vector<bool>();
    }

    // Set the index with the data to be compressed
    void setData(const std::vector<bool>& data) {
        index = data;
    }

    // Compress the data using run-length encoding (RLE)
    void compressData() {
        compressed_index.clear();

        if (index.empty()) {
            return;
        }

        int count = 1;
        for (size_t i = 1; i < index.size(); i++) {
            if (index[i] == index[i - 1]) {
                count++;
            } else {
                compressed_index.push_back(index[i - 1]);
                compressed_index.push_back(count);
                count = 1;
            }
        }

        // Add the last run
        compressed_index.push_back(index.back());
        compressed_index.push_back(count);
    }

    // Decompress the data using run-length encoding (RLE)
    void decompressData() {
        index.clear();

        if (compressed_index.empty()) {
            return;
        }

        for (size_t i = 0; i < compressed_index.size(); i += 2) {
            bool value = compressed_index[i];
            int count = compressed_index[i + 1];

            for (int j = 0; j < count; j++) {
                index.push_back(value);
            }
        }
    }

    // Print the compressed or decompressed data
    void printData() {
        for (bool value : compressed_index) {
            std::cout << (value ? "1" : "0");
        }
        std::cout << std::endl;
    }
};

int main() {
    // Sample data to be compressed
    std::vector<bool> data = {true, true, true, false, false, true, true, false, true};

    CellularAutomaton automaton;

    // Set the data to be compressed
    automaton.setData(data);

    // Compress the data using cellular automata and RLE
    automaton.compressData();
    std::cout << "Compressed Data: ";
    automaton.printData();

    // Decompress the data using cellular automata and RLE
    automaton.decompressData();
    std::cout << "Decompressed Data: ";
    automaton.printData();

    return 0;
}
