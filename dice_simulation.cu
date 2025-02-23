#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define MAX 6
#define MIN 1
#define THREADS 1024
#define BLOCKS 1024

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__global__ void setup_kernel(curandState* state, unsigned long long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void monte_carlo_kernel(curandState* state, unsigned int* count, int m) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[id];

    int sumroll = 0;
    for (int i = 0; i < m; ++i) {
        sumroll += (int)(curand_uniform(&localState) * (MAX - MIN + 0.999999)) + MIN;
    }

    if (sumroll == 3 * m) {
        atomicAdd(count, 1);
    }
}

int main() {
    unsigned int n = BLOCKS * THREADS;
    unsigned int m = 4;

    unsigned int* d_count;
    curandState* d_state;

    // Allocate memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_count, sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_state, n * sizeof(curandState)));
    CHECK_CUDA_ERROR(cudaMemset(d_count, 0, sizeof(unsigned int)));

    // Setup RNG states
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    setup_kernel<<<BLOCKS, THREADS>>>(d_state, seed);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Run Monte Carlo simulation
    auto start = std::chrono::steady_clock::now();
    monte_carlo_kernel<<<BLOCKS, THREADS>>>(d_state, d_count, m);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    // Copy results back to host
    unsigned int h_count;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Calculate and display results
    double time_elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    float chance = static_cast<float>(h_count) / static_cast<float>(n);

    std::cout << "Successful rolls: " << h_count << std::endl;
    std::cout << "Chance: " << chance << std::endl;
    std::cout << "Time taken: " << time_elapsed_ms << " ms" << std::endl;

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_count));
    CHECK_CUDA_ERROR(cudaFree(d_state));

    return 0;
}