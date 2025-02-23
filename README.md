# MonteCarloCUDA
A collection of programs which perform Monte Carlo simulations parallelly using CUDA.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 10.0 or later recommended)
- C++ compiler compatible with your CUDA version

## Compilation
Compile the program using nvcc:
```
nvcc -o dice_simulation dice_simulation.cu 
```

## Usage
Run the compiled program:
```
./dice_simulation
```

## Estimating Pi

### Overview
This CUDA program implements a Monte Carlo method to estimate the value of pi. It demonstrates the use of parallel computing on GPUs to significantly speed up the estimation process.

### Features
- Utilizes CUDA for parallel computation on NVIDIA GPUs
- Implements CURAND for efficient random number generation
- Estimates pi by generating random points and checking if they fall within a unit circle
- Demonstrates basic CUDA programming concepts including kernel launches and device memory management

The program will output the estimated value of pi.

### Code Structure
- `setup_kernel`: Initializes CURAND states for random number generation
- `monte_carlo_kernel`: Performs the Monte Carlo simulation to estimate pi
- `main`: Manages memory allocation, kernel launches, and result processing

### Parameters
- `THREADS_PER_BLOCK`: Number of threads per block (default: 1024)
- `BLOCKS`: Number of blocks (default: 1024)
- `N`: Total number of points generated (THREADS_PER_BLOCK * BLOCKS)

### Performance Considerations
- Adjust `THREADS_PER_BLOCK` and `BLOCKS` to optimize for your specific GPU
- Increasing the total number of points (`N`) will generally improve the accuracy of the estimation

### Potential Improvements
- Generate multiple points per thread to increase efficiency
- Use shared memory for partial sums before using atomic operations
- Implement error checking for CUDA operations

## Dice Roll Probability

### Overview
This CUDA program implements a Monte Carlo simulation to calculate the probability of rolling a specific sum with multiple dice. It demonstrates the use of parallel computing to significantly speed up the simulation process.

### Features
- Utilizes CUDA for parallel computation on NVIDIA GPUs
- Implements CURAND for efficient random number generation
- Calculates the probability of rolling a sum of 3m with m dice
- Provides timing information for performance analysis

The program will output:
1. Number of successful rolls
2. Calculated probability
3. Execution time

### Code Structure
- `setup_kernel`: Initializes CURAND states for random number generation
- `monte_carlo_kernel`: Performs the Monte Carlo simulation
- `main`: Manages memory allocation, kernel launches, and result processing

### Parameters
- `MAX` and `MIN`: Define the range of dice (default: 6-sided dice)
- `THREADS` and `BLOCKS`: Control the number of CUDA threads and blocks
- `m`: Number of dice to roll (default: 3)

### Performance Considerations
- Adjust `THREADS` and `BLOCKS` to optimize for your specific GPU
- The program uses atomic operations for thread-safe counting, which may impact performance on older GPUs

### Error Handling
The program includes comprehensive error checking for CUDA operations, facilitating easier debugging and ensuring proper execution.