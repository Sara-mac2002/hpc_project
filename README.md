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

