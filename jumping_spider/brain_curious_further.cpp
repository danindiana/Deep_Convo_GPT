class Brain {
private:
    bool curiousState; // Internal state variable for curiosity

public:
    // Constructors
    Brain() {
        curiousState = false; // Initialize curiosity state to false
    }

    // Method to check if the brain is curious
    bool isCurious() {
        return curiousState;
    }

    // Method to simulate the brain becoming curious
    void becomeCurious() {
        curiousState = true;
    }

    // Method to simulate the brain losing curiosity
    void loseCuriosity() {
        curiousState = false;
    }

    // Other methods and behaviors of the brain
    // ...
};
