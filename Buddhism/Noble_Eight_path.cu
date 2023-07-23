#include <stdio.h>

__global__ void square(float *d_input, float *d_output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_output[tid] = d_input[tid] * d_input[tid];
    }
}

int main() {
    int size = 10;
    int numBytes = size * sizeof(float);
    
    // Input array on the host (CPU)
    float h_input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    // Output array on the host (CPU)
    float h_output[10];

    // Allocate memory on the GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, numBytes);
    cudaMalloc((void**)&d_output, numBytes);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, numBytes, cudaMemcpyHostToDevice);

    // Define the number of threads per block and blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    square<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    // Copy the result from device to host
    cudaMemcpy(h_output, d_output, numBytes, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Input: ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", h_input[i]);
    }
    printf("\n");

    printf("Output (Squared): ");
    for (int i = 0; i < size; i++) {
        printf("%.2f ", h_output[i]);
    }
    printf("\n");

    // Free allocated memory on the GPU
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
