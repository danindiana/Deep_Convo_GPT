#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <limits>

// Define a graph as an adjacency matrix
using Graph = std::vector<std::vector<int>>;

// Function to calculate the distance between two vertices v1 and v2
int calculateDistance(const Graph& G, int v1, int v2) {
    return G[v1][v2];
}

// Function to perform the RangeSearch algorithm
std::vector<int> RangeSearch(
    const Graph& G,
    std::set<int>& S,
    int q,
    size_t k,
    double epsilon
) {
    std::vector<int> R; // Result list
    std::set<int> C; // Set of checked vertices
    double r = std::numeric_limits<double>::infinity(); // Search radius

    while (!S.empty()) {
        // Find closest vertex to q in S
        int s = *std::min_element(S.begin(), S.end(), [&](int v1, int v2) {
            return calculateDistance(G, v1, q) < calculateDistance(G, v2, q);
        });

        S.erase(s); // Remove s from S

        // Check if vertex s is too far away
        if (calculateDistance(G, s, q) > r * (1 + epsilon)) {
            return R; // Vertex to check is too far away, return result
        }

        // Find neighbors of s
        std::vector<int> N;
        for (int v = 0; v < G.size(); ++v) {
            if (v != s && C.find(v) == C.end()) {
                N.push_back(v);
            }
        }

        for (const auto& n : N) {
            // Check close neighbors
            if (calculateDistance(G, n, q) <= r * (1 + epsilon)) {
                S.insert(n); // Analyze their neighborhood later

                // Check if the neighbor is close enough
                if (calculateDistance(G, n, q) <= r) {
                    R.push_back(n); // Add to the result list

                    // Limit the result list size
                    if (R.size() > k) {
                        // Sort by distance and take the first k elements
                        std::sort(R.begin(), R.end(), [&](int v1, int v2) {
                            return calculateDistance(G, v1, q) < calculateDistance(G, v2, q);
                        });
                        R.resize(k);
                    }

                    r = calculateDistance(G, R.back(), q); // Update search radius
                }
            }
        }

        C.insert(N.begin(), N.end()); // Add checked neighbors to the check list
    }

    return R;
}

// Function to check if an edge between v1 and v2 is MRNG conform
bool checkMRNG(const Graph& G, int v1, int v2) {
    for (size_t u = 0; u < G.size(); ++u) {
        if (G[v1][u] != 0 && G[v2][u] != 0) {
            int distance_v1_v2 = calculateDistance(G, v1, v2);
            int max_weight = std::max(G[v1][u], G[v2][u]);

            if (distance_v1_v2 > max_weight) {
                return false;
            }
        }
    }
    return true;
}

// Function to perform the ExtendGraph algorithm
void ExtendGraph(
    Graph& G,
    int d,
    int v,
    std::set<int>& S,
    size_t k_ext,
    double epsilon_ext
) {
    std::set<int> U; // new neighbors of v

    // Find k_ext vertices similar to v using RangeSearch
    std::vector<int> S_result = RangeSearch(G, S, v, k_ext, epsilon_ext);

    bool skipRNG = false; // two phases: with and w/o RNG checks

    while (U.size() < d) {
        std::set<int> B = S;
        std::set<int> B_diff_U = B;
        std::set_difference(B.begin(), B.end(), U.begin(), U.end(), std::inserter(B_diff_U, B_diff_U.end()));

        while (U.size() < d && !B_diff_U.empty()) {
            int b = *std::min_element(B_diff_U.begin(), B_diff_U.end(), [&](int v1, int v2) {
                return calculateDistance(G, v1, v) < calculateDistance(G, v2, v);
            });

            B_diff_U.erase(b);

            if (skipRNG || checkMRNG(G, v, b)) {
                std::vector<int> N;
                for (int u = 0; u < G.size(); ++u) {
                    if (u != v && U.find(u) == U.end()) {
                        N.push_back(u);
                    }
                }

                int n = *std::max_element(N.begin(), N.end(), [&](int v1, int v2) {
                    return calculateDistance(G, v1, b) < calculateDistance(G, v2, b);
                });

                U.insert(n);
                U.insert(b);

                // Remove edge (n, b)
                G[n][b] = 0;
                G[b][n] = 0;

                skipRNG = true;
            }
        }
    }

    // Add edges (v, u) for all u in U except v
    for (const auto& u : U) {
        if (u != v) {
            G[v][u] = 1;
            G[u][v] = 1;
        }
    }
}

int main() {
    // Example usage of the ExtendGraph function with a sample graph
    Graph G = {
        {0, 1, 1, 0, 0, 0},
        {1, 0, 1, 1, 1, 0},
        {1, 1, 0, 0, 1, 0},
        {0, 1, 0, 0, 1, 1},
        {0, 1, 1, 1, 0, 0},
        {0, 0, 0, 1, 0, 0}
    };

    int d = 4; // Even degree
    int v = 6; // New vertex
    std::set<int> S = {0, 1, 2, 3, 4}; // Set of seed vertices
    size_t k_ext = 4; // Number of search results
    double epsilon_ext = 0.1; // Search range factor

    ExtendGraph(G, d, v, S, k_ext, epsilon_ext);

    // Output the extended graph
    std::cout << "Extended Graph:" << std::endl;
    for (const auto& row : G) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
