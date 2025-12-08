// ======================================================================
// main.c - Neural Network Training Entry Point
// ======================================================================
// This file supports 4 execution modes:
//   1. SEQUENTIAL (default)
//   2. OpenMP  
//   3. MPI
//   4. Hybrid (MPI + OpenMP)
//
// Instructions:
// - Uncomment the section corresponding to your desired mode
// - Make sure your Makefile is configured for the same mode
// - Comment out other modes to avoid conflicts
// ======================================================================

#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

// ======================================================================
// MODE 1: SEQUENTIAL (DEFAULT)
// ======================================================================
// No additional headers needed
// Execution: ./mlp

#include <time.h>  // For timing

int main() {
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // --------------------------
    // Network dimensions
    // --------------------------
    nn_input_dim = 2;        // input features
    nn_output_dim = 2;       // output classes
    int nn_hdim = 10;        // hidden neurons

    // --------------------------
    // Load dataset
    // --------------------------
    num_examples = count_lines(file_y);
    printf("Loading %d samples.\n", num_examples);

    X = malloc(num_examples * nn_input_dim * sizeof(double));
    y = malloc(num_examples * sizeof(int));

    if (!X || !y) {
        fprintf(stderr, "Error: Memory allocation failed for dataset\n");
        return 1;
    }

    load_X(file_X, X, num_examples, nn_input_dim);
    load_y(file_y, y, num_examples);

    epsilon = 0.01;           // Initial learning rate
    reg_lambda = 0.01;        // L2 regularization strength

    // --------------------------
    // Training configuration
    // --------------------------
    int num_passes = 20000;   // training epochs
    int batch_size = 64;      // mini-batch size
    int print_loss = 1;       // display loss progress

    // Decay mode: 0=Constant, 1=Inverse Time, 2=Exponential, 3=Step
    int decay_mode = 0;

    // Activation: 1=Tanh, 2=ReLU, 3=Sigmoid, 4=Leaky ReLU
    int activation_mode = 4;

    // --------------------------
    // Display configuration
    // --------------------------
    printf("\n=== Training Configuration ===\n");
    printf("Dataset: %d samples\n", num_examples);
    printf("Architecture: %d -> %d -> %d\n", nn_input_dim, nn_hdim, nn_output_dim);
    printf("Epochs: %d | Batch: %d\n", num_passes, batch_size);
    printf("Learning Rate: %.6f | Lambda: %.6f\n", epsilon, reg_lambda);
    printf("Decay: %s | Activation: %s\n", 
           (decay_mode == 0 ? "Constant" : decay_mode == 1 ? "Inverse Time" : 
            decay_mode == 2 ? "Exponential" : "Step"),
           (activation_mode == 1 ? "Tanh" : activation_mode == 2 ? "ReLU" : 
            activation_mode == 3 ? "Sigmoid" : "Leaky ReLU"));
    printf("Mode: SEQUENTIAL\n");
    printf("==============================\n\n");

    clock_t start_time = clock();

    // Call sequential training function
    build_model_sequential(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\n=== Execution Summary ===\n");
    printf("Total execution time: %.4f seconds\n", elapsed);
    printf("Time per epoch: %.6f seconds\n", elapsed / num_passes);
    printf("=========================\n\n");

    free(X);
    free(y);

    printf("Training complete. Weights saved to output/\n");
    return 0;
}

// ======================================================================
// MODE 2: OpenMP
// ======================================================================
// To use OpenMP mode:
// 1. Comment out the SEQUENTIAL section above (from #include <time.h> to end of main)
// 2. Uncomment this entire section below
// 3. Update Makefile to use OpenMP flags
// 4. Execution: OMP_NUM_THREADS=4 ./mlp
// ======================================================================

/*
#include <omp.h>  // OpenMP header

int main() {
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    nn_input_dim = 2;
    nn_output_dim = 2;
    int nn_hdim = 10;

    num_examples = count_lines(file_y);
    printf("Loading %d samples.\n", num_examples);

    X = malloc(num_examples * nn_input_dim * sizeof(double));
    y = malloc(num_examples * sizeof(int));

    if (!X || !y) {
        fprintf(stderr, "Error: Memory allocation failed for dataset\n");
        return 1;
    }

    load_X(file_X, X, num_examples, nn_input_dim);
    load_y(file_y, y, num_examples);

    epsilon = 0.01;
    reg_lambda = 0.01;

    int num_passes = 20000;
    int batch_size = 64;
    int print_loss = 1;
    int decay_mode = 0;
    int activation_mode = 4;

    printf("\n=== Training Configuration ===\n");
    printf("Dataset: %d samples\n", num_examples);
    printf("Architecture: %d -> %d -> %d\n", nn_input_dim, nn_hdim, nn_output_dim);
    printf("Epochs: %d | Batch: %d\n", num_passes, batch_size);
    printf("Learning Rate: %.6f | Lambda: %.6f\n", epsilon, reg_lambda);
    printf("Decay: %s | Activation: %s\n", 
           (decay_mode == 0 ? "Constant" : decay_mode == 1 ? "Inverse Time" : 
            decay_mode == 2 ? "Exponential" : "Step"),
           (activation_mode == 1 ? "Tanh" : activation_mode == 2 ? "ReLU" : 
            activation_mode == 3 ? "Sigmoid" : "Leaky ReLU"));
    printf("Mode: OpenMP | Threads: %d\n", omp_get_max_threads());
    printf("==============================\n\n");

    double start_time = omp_get_wtime();

    // Choose OpenMP implementation:
    // build_model_omp(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);
    build_model_task(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);

    double end_time = omp_get_wtime();
    
    printf("\n=== Execution Summary ===\n");
    printf("Total execution time: %.4f seconds\n", end_time - start_time);
    printf("Time per epoch: %.6f seconds\n", (end_time - start_time) / num_passes);
    printf("=========================\n\n");

    free(X);
    free(y);

    printf("Training complete. Weights saved to output/\n");
    return 0;
}
*/

// ======================================================================
// MODE 3: MPI
// ======================================================================
// To use MPI mode:
// 1. Comment out the SEQUENTIAL section
// 2. Uncomment this entire section below
// 3. Update Makefile to use mpicc
// 4. Execution: mpirun -np 4 ./mlp
// ======================================================================

/*
#include <mpi.h>  // MPI header

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    nn_input_dim  = 2;
    nn_output_dim = 2;
    int nn_hdim   = 10;
    int activation_mode = 4;  // Leaky ReLU

    // Load dataset on rank 0, broadcast to all processes
    if (rank == 0) {
        num_examples = count_lines(file_y);
        printf("Loading %d samples.\n", num_examples);
    }

    MPI_Bcast(&num_examples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    X = malloc(num_examples * nn_input_dim * sizeof(double));
    y = malloc(num_examples * sizeof(int));

    if (!X || !y) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        load_X(file_X, X, num_examples, nn_input_dim);
        load_y(file_y, y, num_examples);
    }

    MPI_Bcast(X, num_examples * nn_input_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, num_examples, MPI_INT, 0, MPI_COMM_WORLD);

    epsilon = 0.01;
    reg_lambda = 0.01;

    int batch_size = 64;
    int num_passes = 20000;
    int print_loss = 1;

    if (rank == 0) {
        printf("\n=== Training Configuration ===\n");
        printf("Dataset: %d samples\n", num_examples);
        printf("Architecture: %d -> %d -> %d\n", nn_input_dim, nn_hdim, nn_output_dim);
        printf("Epochs: %d | Batch: %d\n", num_passes, batch_size);
        printf("Learning Rate: %.6f | Lambda: %.6f\n", epsilon, reg_lambda);
        printf("Activation: %s\n", 
               (activation_mode == 1 ? "Tanh" : activation_mode == 2 ? "ReLU" : 
                activation_mode == 3 ? "Sigmoid" : "Leaky ReLU"));
        printf("Mode: MPI | Processes: %d\n", size);
        printf("==============================\n\n");
    }

    double start_time = MPI_Wtime();

    build_model_mpi(nn_hdim, num_passes, batch_size, print_loss, rank, size, activation_mode);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== Execution Summary ===\n");
        printf("Total execution time: %.4f seconds\n", end_time - start_time);
        printf("Time per epoch: %.6f seconds\n", (end_time - start_time) / num_passes);
        printf("=========================\n\n");
    }

    free(X);
    free(y);

    if (rank == 0)
        printf("Training complete. Weights saved to output/\n");

    MPI_Finalize();
    return 0;
}
*/

// ======================================================================
// MODE 4: HYBRID (MPI + OpenMP)
// ======================================================================
// To use Hybrid mode:
// 1. Comment out the SEQUENTIAL section
// 2. Uncomment this entire section below
// 3. Update Makefile to use mpicc with -fopenmp flag
// 4. Execution: mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 ./mlp
// ======================================================================

/*
#include <mpi.h>   // MPI header
#include <omp.h>   // OpenMP header

int main(int argc, char **argv) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    nn_input_dim  = 2;
    nn_output_dim = 2;
    int nn_hdim   = 10;
    int activation_mode = 4;

    if (rank == 0) {
        num_examples = count_lines(file_y);
        printf("Loading %d samples.\n", num_examples);
    }

    MPI_Bcast(&num_examples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    X = malloc(num_examples * nn_input_dim * sizeof(double));
    y = malloc(num_examples * sizeof(int));

    if (!X || !y) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        load_X(file_X, X, num_examples, nn_input_dim);
        load_y(file_y, y, num_examples);
    }

    MPI_Bcast(X, num_examples * nn_input_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, num_examples, MPI_INT, 0, MPI_COMM_WORLD);

    epsilon = 0.01;
    reg_lambda = 0.01;

    int batch_size = 64;
    int num_passes = 20000;
    int print_loss = 1;

    if (rank == 0) {
        printf("\n=== Training Configuration ===\n");
        printf("Dataset: %d samples\n", num_examples);
        printf("Architecture: %d -> %d -> %d\n", nn_input_dim, nn_hdim, nn_output_dim);
        printf("Epochs: %d | Batch: %d\n", num_passes, batch_size);
        printf("Learning Rate: %.6f | Lambda: %.6f\n", epsilon, reg_lambda);
        printf("Activation: %s\n", 
               (activation_mode == 1 ? "Tanh" : activation_mode == 2 ? "ReLU" : 
                activation_mode == 3 ? "Sigmoid" : "Leaky ReLU"));
        printf("Mode: HYBRID | MPI Processes: %d | OpenMP Threads/Process: %d\n", 
               size, omp_get_max_threads());
        printf("==============================\n\n");
    }

    double start_time = MPI_Wtime();

    // Call hybrid MPI+OpenMP training function
    // You would need to implement build_model_hybrid() that combines both
    build_model_mpi(nn_hdim, num_passes, batch_size, print_loss, rank, size, activation_mode);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n=== Execution Summary ===\n");
        printf("Total execution time: %.4f seconds\n", end_time - start_time);
        printf("Time per epoch: %.6f seconds\n", (end_time - start_time) / num_passes);
        printf("=========================\n\n");
    }

    free(X);
    free(y);

    if (rank == 0)
        printf("Training complete. Weights saved to output/\n");

    MPI_Finalize();
    return 0;
}
*/

// ======================================================================
// End of main.c
// ======================================================================