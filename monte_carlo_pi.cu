#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

#define THREADS_PER_BLOCK 1024
#define BLOCKS 1024
#define N (THREADS_PER_BLOCK * BLOCKS)

__global__ void setup_kernel(curandState *state, unsigned long long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void monte_carlo_kernel(curandState *state, int *count)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float x, y;
    
    x = curand_uniform(&state[id]);
    y = curand_uniform(&state[id]);
    
    if (x*x + y*y <= 1.0f)
        atomicAdd(count, 1);
}

int main() {
    curandState *d_state;
    int *d_count, h_count = 0;
    
    cudaMalloc(&d_state, N * sizeof(curandState));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    setup_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state,seed);
    monte_carlo_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_state, d_count);
    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    float pi_estimate = 4.0f * h_count / (float)N;
    printf("Estimated value of pi: %f\n", pi_estimate);
    
    cudaFree(d_state);
    cudaFree(d_count);
    
    return 0;
}
