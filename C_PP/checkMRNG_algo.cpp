#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

// Define a graph as an adjacency matrix
using Graph = std::vector<std::vector<int>>;

// Function to calculate the distance between two vertices v1 and v2
int calculateDistance(const Graph& G, int v1, int v2) {
    return G[v1][v2];
}

// Function to find the maximum weight between vertex v1 and v2 for neighbors in common with v1 and v2
int findMaxWeight(const Graph& G, int v1, int v2) {
    int maxWeight = std::numeric_limits<int>::min();
    for (size_t u = 0; u < G.size(); ++u) {
        if (G[v1][u] != 0 && G[v2][u] != 0) {
            int weight = std::max(G[v1][u], G[v2][u]);
            maxWeight = std::max(maxWeight, weight);
        }
    }
    return maxWeight;
}

// Function to check if an edge between v1 and v2 is MRNG conform
bool checkMRNG(const Graph& G, int v1, int v2) {
    for (size_t u = 0; u < G.size(); ++u) {
        if (G[v1][u] != 0 && G[v2][u] != 0) {
            int distance_v1_v2 = calculateDistance(G, v1, v2);
            int max_weight = findMaxWeight(G, v1, v2);

            if (distance_v1_v2 > max_weight) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Example usage of the checkMRNG function with a sample graph
    Graph G = {
        {0, 2, 0, 4},
        {2, 0, 1, 0},
        {0, 1, 0, 3},
        {4, 0, 3, 0}
    };

    int v1 = 0; // Vertex v1 in the graph
    int v2 = 1; // Vertex v2 in the graph

    // Check if the edge between v1 and v2 is MRNG conform
    bool is_MRNG_conform = checkMRNG(G, v1, v2);

    // Output the result
    if (is_MRNG_conform) {
        std::cout << "The edge between vertex " << v1 << " and vertex " << v2 << " is MRNG conform." << std::endl;
    } else {
        std::cout << "The edge between vertex " << v1 << " and vertex " << v2 << " is not MRNG conform." << std::endl;
    }

    return 0;
}
