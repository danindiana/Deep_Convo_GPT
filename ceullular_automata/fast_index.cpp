#include <iostream>
#include <vector>

const int INDEX_SIZE = 100; // Number of cells in the index
const int RULES = 8; // Number of rules for cellular automaton (2^3 for 3-cell neighborhood)

class CellularAutomaton {
private:
    std::vector<bool> index;
    std::vector<int> rules;

public:
    CellularAutomaton() {
        index = std::vector<bool>(INDEX_SIZE, false);
        rules = std::vector<int>(RULES, 0);
        initializeRules();
    }

    // Initialize the rules for cellular automaton
    void initializeRules() {
        // Example rule: update cell based on its current state and its neighbors' states
        // You can customize this array to implement your specific rules for index updating.
        // Here, we just assign random rules for demonstration purposes.
        rules[0] = 0;
        rules[1] = 1;
        rules[2] = 0;
        rules[3] = 1;
        rules[4] = 1;
        rules[5] = 0;
        rules[6] = 1;
        rules[7] = 0;
    }

    // Update the index based on cellular automaton rules
    void updateIndex() {
        std::vector<bool> new_index(INDEX_SIZE, false);

        for (int i = 1; i < INDEX_SIZE - 1; i++) {
            int neighborhood = (index[i - 1] << 2) | (index[i] << 1) | index[i + 1];
            new_index[i] = rules[neighborhood];
        }

        index = new_index;
    }

    // Add data to the index
    void addData(int position) {
        if (position >= 0 && position < INDEX_SIZE) {
            index[position] = true;
        }
    }

    // Remove data from the index
    void removeData(int position) {
        if (position >= 0 && position < INDEX_SIZE) {
            index[position] = false;
        }
    }

    // Print the current state of the index
    void printIndex() {
        for (int i = 0; i < INDEX_SIZE; i++) {
            std::cout << (index[i] ? "1" : "0");
        }
        std::cout << std::endl;
    }
};

int main() {
    CellularAutomaton automaton;

    // Initially empty index
    automaton.printIndex();

    // Add data at position 10 and 20
    automaton.addData(10);
    automaton.addData(20);
    automaton.updateIndex();
    automaton.printIndex();

    // Modify data at position 10 (remove) and 30 (add)
    automaton.removeData(10);
    automaton.addData(30);
    automaton.updateIndex();
    automaton.printIndex();

    // Remove data from position 20
    automaton.removeData(20);
    automaton.updateIndex();
    automaton.printIndex();

    return 0;
}
