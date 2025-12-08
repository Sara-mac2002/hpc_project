

#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    const char *file_X = "data/data_X.txt";
    const char *file_y = "data/data_y.txt";

    // --------------------------
    // Network dimensions
    // --------------------------
    nn_input_dim = 2;        // two features
    nn_output_dim = 2;       // two classes
    int nn_hdim = 10;        // hidden neurons (changed from 512 to 10)

    // --------------------------
    // Load dataset
    // --------------------------
    num_examples = count_lines(file_y);
    printf("Chargement de %d échantillons.\n", num_examples);

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
    int num_passes = 20000;   // training iterations (changed from 5000 to 20000)
    int batch_size = 64;      // mini-batch size (changed from 128 to 64)
    int print_loss = 1;       // display loss progress

    // Decay mode options:
    // 0 = Constant LR
    // 1 = Inverse Time Decay
    // 2 = Exponential Decay
    // 3 = Step Decay
    int decay_mode = 0;

    // Activation function options:
    // 1 = Tanh
    // 2 = ReLU
    // 3 = Sigmoid
    // 4 = Leaky ReLU
    int activation_mode = 4;  // FIXED: Changed from 3 to 4 for Leaky ReLU


    // --------------------------
    // Display configuration
    // --------------------------
   
    printf("\n=== Training Configuration ===\nDataset: %d samples | Input: %d | Hidden: %d | Output: %d | Epochs: %d | Batch: %d | LR: %.6f | λ: %.6f | Threads: %d | Decay: %s | Activation: %s | Init: %s\n==============================\n\n", num_examples, nn_input_dim, nn_hdim, nn_output_dim, num_passes, batch_size, epsilon, reg_lambda, omp_get_max_threads(), (decay_mode == 0 ? "Constant" : decay_mode == 1 ? "Inverse Time" : decay_mode == 2 ? "Exponential" : "Step"), (activation_mode == 1 ? "Tanh" : activation_mode == 2 ? "ReLU" : activation_mode == 3 ? "Sigmoid" : "Leaky ReLU"), ((activation_mode == 2 || activation_mode == 4) ? "He" : "Xavier/Glorot"));



    double start_time = omp_get_wtime();

    // Choose your parallelization mode:
    build_model_sequential(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);
    // build_model_omp(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);
    // build_model_task(nn_hdim, num_passes, batch_size, decay_mode, activation_mode, print_loss);

    double end_time = omp_get_wtime();
    
    printf("\n=== Execution Summary ===\n");
    printf("Total execution time: %.4f seconds\n", end_time - start_time);
    printf("Time per epoch: %.6f seconds\n", (end_time - start_time) / num_passes);
    printf("=========================\n\n");

    free(X);
    free(y);

    printf("All memory released. Fin de l'exécution.\n");
    return 0;
}

// #include "model.h"
// #include "utils.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <mpi.h>

// int main(int argc, char **argv) {

//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     const char *file_X = "data/data_X.txt";
//     const char *file_y = "data/data_y.txt";

//     // Network dimensions
//     nn_input_dim  = 2;     // input features
//     nn_output_dim = 2;     // output classes
//     int nn_hdim   = 512;   // hidden neurons
//     int activation_mode = 1; // 1=tanh by default

//     // === Debug message only on rank 0 ===
//     if (rank == 0) {
//         FILE *test = fopen(file_y, "r");
//         if (test == NULL) {
//             printf("ERROR: Rank 0 cannot open file_y: %s\n", file_y);
//         } else {
//             printf("Rank 0 can open file_y: %s\n", file_y);
//             fclose(test);
//         }
//     }

//     // === Load number of examples ONLY on rank 0 ===
//     if (rank == 0) {
//         num_examples = count_lines(file_y);
//         printf("Chargement de %d échantillons.\n", num_examples);
//     }

//     // Broadcast num_examples to all processes
//     MPI_Bcast(&num_examples, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // === Allocate X and y on all ranks ===
//     X = malloc(num_examples * nn_input_dim * sizeof(double));
//     y = malloc(num_examples * sizeof(int));

//     if (X == NULL || y == NULL) {
//         printf("Memory allocation failed on rank %d!\n", rank);
//         MPI_Abort(MPI_COMM_WORLD, -1);
//     }

//     // === Rank 0 loads data, others wait ===
//     if (rank == 0) {
//         load_X(file_X, X, num_examples, nn_input_dim);
//         load_y(file_y, y, num_examples);
//     }

//     // Broadcast the dataset to all processes
//     MPI_Bcast(X, num_examples * nn_input_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//     MPI_Bcast(y, num_examples, MPI_INT, 0, MPI_COMM_WORLD);

//     int batch_size = 128;
//     int num_passes = 5000;
//     int print_loss = 1;

//     if (rank == 0)
//         printf("MPI size: %d | Batch size: %d | Activation mode: %d\n",
//                size, batch_size, activation_mode);

//     double start_time = MPI_Wtime();

//     // === CALL YOUR MPI TRAINING FUNCTION ===
//     build_model_mpi(nn_hdim, num_passes, batch_size, print_loss, rank, size, activation_mode);

//     double end_time = MPI_Wtime();

//     if (rank == 0)
//         printf("Execution time: %.4f seconds\n", end_time - start_time);

//     free(X);
//     free(y);

//     MPI_Finalize();
//     return 0;
// }
