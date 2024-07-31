#include <iostream>
#include <vector>
#include <string>

// Structure to represent an algorithm step
struct Step {
    std::string description;
};

// Structure to represent an algorithm
struct Algorithm {
    std::string name;
    std::string description;
    std::vector<Step> steps;
};

// Structure to represent common metrics
struct CommonMetrics {
    std::vector<Algorithm> firingRate;
    std::vector<Algorithm> interspikeInterval;
    std::vector<Algorithm> burstMetrics;
    std::vector<Algorithm> synchronyMetrics;
    std::vector<Algorithm> spikeTrainMetrics;
};

// Structure to represent the choice of the right metric
struct ChoosingTheRightMetric {
    std::string interestInOverallFiringRate;
    std::vector<std::string> interestInTemporalPatternsAndVariability;
};

// Structure to represent the entire pulse spike train speed metrics system
struct PulseSpikeTrainSpeedMetrics {
    CommonMetrics commonMetrics;
    ChoosingTheRightMetric choosingTheRightMetric;
    std::vector<Algorithm> biologicalAlgorithms;
};

// Function to print the details of an algorithm
void printAlgorithm(const Algorithm& algorithm) {
    std::cout << "Algorithm: " << algorithm.name << std::endl;
    std::cout << "Description: " << algorithm.description << std::endl;
    std::cout << "Steps:" << std::endl;
    for (const auto& step : algorithm.steps) {
        std::cout << "- " << step.description << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Example algorithms
    Algorithm meanFiringRateCalculation = {
        "MeanFiringRateCalculation",
        "Calculates the average number of spikes per unit time",
        {
            {"Collect spike data over a defined period"},
            {"Count the total number of spikes"},
            {"Divide the total spike count by the time period"}
        }
    };

    Algorithm instantaneousFiringRateCalculation = {
        "InstantaneousFiringRateCalculation",
        "Estimates the firing rate at a specific moment using a smoothing window",
        {
            {"Define a smoothing window or kernel"},
            {"Slide the window across the spike data"},
            {"Count spikes within each window"},
            {"Normalize by the window size"}
        }
    };

    // Populate common metrics
    CommonMetrics commonMetrics = {
        {meanFiringRateCalculation, instantaneousFiringRateCalculation}, // Firing Rate
        {}, // Interspike Interval
        {}, // Burst Metrics
        {}, // Synchrony Metrics
        {}  // Spike Train Metrics
    };

    // Populate choosing the right metric
    ChoosingTheRightMetric choosingTheRightMetric = {
        "MeanFiringRate",
        {"ISICV", "BurstMetrics"}
    };

    // Populate biological algorithms
    std::vector<Algorithm> biologicalAlgorithms = {
        meanFiringRateCalculation,
        instantaneousFiringRateCalculation
    };

    // Create the main structure
    PulseSpikeTrainSpeedMetrics pulseMetrics = {
        commonMetrics,
        choosingTheRightMetric,
        biologicalAlgorithms
    };

    // Print the details of the biological algorithms
    for (const auto& algorithm : pulseMetrics.biologicalAlgorithms) {
        printAlgorithm(algorithm);
    }

    return 0;
}
