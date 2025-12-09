# MLP Neural Network: Sequential, OpenMP, MPI, and Hybrid Implementations

A high-performance Multi-Layer Perceptron (MLP) implementation in C with four independent parallelization strategies for comparative performance analysis.
## Authors

**Salma Oumoussa** and **Sara Samouche**  
## Overview

This project implements a feedforward neural network with four separate implementations:
- **Sequential** (`model_sequential.c`): Single-threaded baseline
- **OpenMP** (`model_openmp.c`): Shared-memory parallelization with task-based approach
- **MPI** (`model_mpi.c`): Distributed-memory parallelization
- **Hybrid** (`model_hybrid.c`): Combined MPI + OpenMP

All implementations use mini-batch SGD with identical hyperparameters for fair performance comparison.

## Repository Structure
```
mlp_codes/
├── data/                      # Training datasets
│   ├── data_X.txt            # Input features (N × 2)
│   └── data_y.txt            # Labels (N × 1)
├── output/                   # Trained model weights and visualizations
│   ├── W1.txt               # Hidden layer weights
│   ├── W2.txt               # Output layer weights
│   ├── b1.txt               # Hidden layer biases
│   ├── b2.txt               # Output layer biases
│   ├── decision_boundary.png
│   ├── decision_boundary_mpi.png
│   ├── callgraph.png
│   └── dmpi.png
├── main.c                    # Unified entry point
├── model.c / model.h         # Shared model functions
├── model_sequential.c        # Sequential implementation
├── model_openmp.c           # OpenMP implementation
├── model_mpi.c              # MPI implementation
├── model_hybrid.c           # Hybrid MPI+OpenMP implementation
├── utils.c / utils.h        # Matrix operations and utilities
├── Makefile                 # Build system (all modes)
├── generate_moon.py         # Dataset generation
├── check_output.py          # Visualization script
├── test_mpi_comparison.sh   # MPI testing script
└── README.md
```

## Quick Start

### Prerequisites
- GCC 11.2.0 or later
- OpenMPI 4.1.1 or later (for MPI/Hybrid modes)
- Python 3.x with matplotlib, numpy, sklearn (for visualization)

Building and Running
Building the Project
The Makefile uses a single compilation target. To switch between modes, uncomment the desired configuration in the Makefile.
bash# Clean previous builds
make clean

# Build (after uncommenting desired mode in Makefile)
make

## Running Examples
### Sequential Mode
Steps:

In Makefile, uncomment the sequential configuration
Build and run:
```
make clean && make
./mlp
```
Output:

Prints loss every 1000 epochs
Saves weights to output/


### OpenMP Mode
Steps:

In Makefile, uncomment the OpenMP configuration
Build and run:
```
make clean && make
export OMP_NUM_THREADS=8
./mlp
```
Configuration:

Uses task-based parallelization for better load balancing
Prints loss every 10 epochs
Optimal performance: 4-8 threads


### MPI Mode
Steps:

In Makefile, uncomment the MPI configuration
Build and run:

Single Node (4 processes):
```
make clean && make
mpirun -np 4 ./mlp
Multiple Nodes:
make clean && make
mpirun -np 8 --host node1,node2,node3,node4 ./mlp
```
Features:

Data automatically partitioned across processes
Gradients synchronized via MPI_Allreduce
Rank 0 saves final weights


### Hybrid Mode (MPI + OpenMP)
Steps:

In Makefile, uncomment the hybrid configuration
Build and run:
```
make clean && make
export OMP_NUM_THREADS=4
mpirun -np 2 ./mlp
```
Configuration:

Combines distributed-memory (MPI) and shared-memory (OpenMP) parallelism
Balance processes and threads based on your hardware

# 2 MPI processes × 4 OpenMP threads = 8 total workers
mpirun -np 2 -x OMP_NUM_THREADS=4 ./mlp

# 4 MPI processes × 2 OpenMP threads = 8 total workers
mpirun -np 4 -x OMP_NUM_THREADS=2 ./mlp
```
**Notes**:
- Best for multi-node clusters
- Combines inter-node (MPI) and intra-node (OpenMP) parallelism
- Total parallelism = processes × threads

## Configuration

All implementations share the same configuration in `main.c`:

### Network Architecture
```c
nn_input_dim = 2;        // Input features
nn_output_dim = 2;       // Output classes
int nn_hdim = 10;        // Hidden layer neurons
```

### Training Parameters
```c
int num_passes = 20000;  // Training epochs
int batch_size = 64;     // Mini-batch size
double epsilon = 0.01;   // Initial learning rate
double reg_lambda = 0.01; // L2 regularization
int activation_mode = 4; // 1=Tanh, 2=ReLU, 3=Sigmoid, 4=Leaky ReLU
```

### Learning Rate Decay
```c
int decay_mode = 3;      // Decay strategy
double k = 1.5e-5;       // Decay rate (modes 1, 2)
double gamma = 0.7;      // Decay factor (mode 3)
int step_size = 10000;   // Steps between decays (mode 3)
```

**Decay Modes**:
- `0`: Constant learning rate
- `1`: Inverse time decay: `lr = lr₀ / (1 + k·t)`
- `2`: Exponential decay: `lr = lr₀ · e^(-k·t)`
- `3`: Step decay: `lr = lr₀ · γ^(t/s)`

## Implementation Details

### Sequential (`model_sequential.c`)
- `build_model_sequential()`: Pure single-threaded implementation
- `calculate_loss()`: Standard loss computation
- Uses Xavier/He initialization based on activation function
- Mini-batch SGD with per-iteration learning rate decay

### OpenMP (`model_openmp.c`)
- `build_model_task()`: Task-based parallelization (recommended)
- Each mini-batch processed as an independent task
- Critical sections for thread-safe weight updates
- Better load balancing than loop-based approach
```c
#pragma omp parallel
{
    #pragma omp single
    for (int b = 0; b < num_batches; b++) {
        #pragma omp task firstprivate(b)
        {
            // Process batch b
            #pragma omp critical
            { /* Update shared weights */ }
        }
    }
}
```

### MPI (`model_mpi.c`)
- `build_model_mpi()`: Data-parallel distributed training
- Each process trains on `local_size = N / num_procs` samples
- `MPI_Allreduce()` aggregates gradients across all processes
- Synchronized weight updates ensure consistency
- Rank 0 saves final model
```c
// Each process computes local gradients
compute_gradients(local_data);

// Aggregate across all processes
MPI_Allreduce(local_dW1, global_dW1, ..., MPI_SUM, MPI_COMM_WORLD);

// All processes update weights identically
W1[i] -= epsilon_t * global_dW1[i];
```

### Hybrid (`model_hybrid.c`)
- Combines MPI (between nodes) and OpenMP (within nodes)
- `MPI_Init_thread(MPI_THREAD_FUNNELED)` for thread safety
- Each MPI process spawns multiple OpenMP threads
- Best for multi-node HPC clusters

## Activation Functions

All implementations support four activation functions (set via `activation_mode`):

| Mode | Function | Formula | Derivative | Use Case |
|------|----------|---------|-----------|----------|
| 1 | Tanh | tanh(x) | 1 - a² | General purpose, zero-centered |
| 2 | ReLU | max(0, x) | x > 0 ? 1 : 0 | Deep networks, fast training |
| 3 | Sigmoid | 1/(1+e^-x) | a(1-a) | Binary classification, output layer |
| 4 | Leaky ReLU | x > 0 ? x : 0.01x | x > 0 ? 1 : 0.01 | Prevents dead neurons |

## Testing and Visualization

### Generate Training Data
```bash
python3 generate_moon.py
```
Creates synthetic moon-shaped dataset in `data/`

### Visualize Results
```bash
python3 check_output.py
```
Generates decision boundary plots using trained weights

### MPI Comparison Test
```bash
bash test_mpi_comparison.sh
```
Automated testing script for MPI implementation

## File Descriptions

### Core Implementation Files
- **`model_sequential.c`**: Pure sequential implementation
- **`model_openmp.c`**: OpenMP task-based parallelization
- **`model_mpi.c`**: MPI distributed training
- **`model_hybrid.c`**: MPI + OpenMP combined
- **`model.c/h`**: Shared utility functions
- **`utils.c/h`**: Matrix operations, data loading
- **`main.c`**: Entry point for all modes

### Build and Test
- **`Makefile`**: Build system with targets for each mode
- **`test_mpi_comparison.sh`**: Automated MPI testing

### Visualization
- **`generate_moon.py`**: Dataset generation
- **`check_output.py`**: Decision boundary visualization

## Makefile Targets
```bash
make
./mlp
```


```

### Different results between modes
All modes should produce similar (not identical) results due to:
- **Floating-point arithmetic order** (MPI aggregates differently)
- **Different batch orderings** (MPI partitions data)
- **Random initialization** (ensure same seed with `srand(0)`)

To verify correctness:
1. Check final loss values are similar (±5%)
2. Visualize decision boundaries (should be nearly identical)
3. Run multiple times with same seed

### Compilation errors
```bash
# Rebuild from scratch
make clean
make   # or openmp/mpi/hybrid

# Check compiler
gcc --version
mpicc --version

# Verify file structure
ls -la *.c *.h
```

## Performance Tips

1. **OpenMP threads**: Set to number of physical cores (not hyperthreads)
```bash
   lscpu | grep "Core(s) per socket"
   export OMP_NUM_THREADS=
```

2. **MPI processes**: More processes = better scaling (up to data size)
   - Single node: Use 4-16 processes
   - Multi-node: 1-4 processes per node

3. **Hybrid configuration**: Balance MPI × OpenMP
   - 8 cores: `mpirun -np 2` with `OMP_NUM_THREADS=4`
   - 16 cores: `mpirun -np 4` with `OMP_NUM_THREADS=4`

4. **Batch size**: 64-128 for optimal convergence
   - Too small: High variance, slow
   - Too large: Poor generalization

5. **Network size**: For `nn_hdim > 100`, use ReLU or Leaky ReLU

## Output Files

After training, weights are saved to `output/`:

- **`W1.txt`**: Hidden layer weights (nn_hdim × nn_input_dim)
- **`W2.txt`**: Output layer weights (nn_output_dim × nn_hdim)
- **`b1.txt`**: Hidden layer biases (nn_hdim × 1)
- **`b2.txt`**: Output layer biases (nn_output_dim × 1)

Visualization outputs:
- **`decision_boundary.png`**: Classification boundary
- **`callgraph.png`**: Performance profiling graph

|



Predoc 2025 Program - High Performance Computing Module  
Mohammed VI Polytechnic University (UM6P)  
Supervised by: Professor Imad Kissami

