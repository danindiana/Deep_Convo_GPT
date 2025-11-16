#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <limits>

// Function to calculate the distance between two vertices
double calculateDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to perform the RangeSearch algorithm
std::vector<std::vector<double>> RangeSearch(
    const std::vector<std::vector<double>>& G,
    std::set<size_t>& S,
    const std::vector<double>& q,
    size_t k,
    double epsilon
) {
    std::vector<std::vector<double>> R; // Result list
    std::set<size_t> C; // Set of checked vertices
    double r = std::numeric_limits<double>::infinity(); // Search radius

    while (!S.empty()) {
        // Find closest vertex to q in S
        size_t s = *std::min_element(S.begin(), S.end(), [&](size_t v1, size_t v2) {
            return calculateDistance(G[v1], q) < calculateDistance(G[v2], q);
        });

        S.erase(s); // Remove s from S

        // Check if vertex s is too far away
        if (calculateDistance(G[s], q) > r * (1 + epsilon)) {
            return R; // Vertex to check is too far away, return result
        }

        // Find neighbors of s
        std::vector<size_t> N;
        for (size_t v = 0; v < G.size(); ++v) {
            if (v != s && C.find(v) == C.end()) {
                N.push_back(v);
            }
        }

        for (const auto& n : N) {
            // Check close neighbors
            if (calculateDistance(G[n], q) <= r * (1 + epsilon)) {
                S.insert(n); // Analyze their neighborhood later

                // Check if the neighbor is close enough
                if (calculateDistance(G[n], q) <= r) {
                    R.push_back(G[n]); // Add to the result list

                    // Limit the result list size
                    if (R.size() > k) {
                        // Sort by distance and take the first k elements
                        std::sort(R.begin(), R.end(), [&](const std::vector<double>& v1, const std::vector<double>& v2) {
                            return calculateDistance(v1, q) < calculateDistance(v2, q);
                        });
                        R.resize(k);
                    }

                    r = calculateDistance(R.back(), q); // Update search radius
                }
            }
        }

        C.insert(N.begin(), N.end()); // Add checked neighbors to the check list
    }

    return R;
}

int main() {
    // Example usage of RangeSearch
    // Assume G is a 2D vector representing the graph with vertices and their coordinates
    std::vector<std::vector<double>> G = {
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0},
        {4.0, 4.0}
    };

    // Set of seed vertices S
    std::set<size_t> S = {0, 1, 2};

    // Query q
    std::vector<double> q = {3.5, 3.5};

    // Number of search results k
    size_t k = 2;

    // Search range factor epsilon
    double epsilon = 0.1;

    // Perform RangeSearch
    std::vector<std::vector<double>> result = RangeSearch(G, S, q, k, epsilon);

    // Output the result
    std::cout << "Result List: " << std::endl;
    for (const auto& vertex : result) {
        std::cout << "(" << vertex[0] << ", " << vertex[1] << ")" << std::endl;
    }

    return 0;
}
