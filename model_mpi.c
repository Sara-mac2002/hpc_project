#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>  
// #include <mpi.h>
// ----------------------
// Global variables
// ----------------------
int num_examples;
int nn_input_dim;
int nn_output_dim;
double reg_lambda = 0.01;
double epsilon = 0.01;
double *X; // (num_examples × nn_input_dim)
int *y;    // (num_examples)



double calculate_loss_mpi(double *W1, double *b1, double *W2, double *b2, 
                          int nn_hdim, double *X_local, int *y_local, 
                          int num_local, int activation_mode) 
{
    // Allocate forward pass buffers
    double *z1 = calloc(num_local * nn_hdim, sizeof(double));
    double *a1 = calloc(num_local * nn_hdim, sizeof(double));
    double *z2 = calloc(num_local * nn_output_dim, sizeof(double));
    double *probs = calloc(num_local * nn_output_dim, sizeof(double));

    // Forward pass: Input -> Hidden layer
    matmul(X_local, W1, z1, num_local, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_local, nn_hdim);
    
    // Apply activation function
    for (int i = 0; i < num_local * nn_hdim; i++) {
        double x = z1[i];
        switch (activation_mode) {
            case 1: a1[i] = tanh(x); break;                    // tanh
            case 2: a1[i] = (x > 0) ? x : 0.0; break;          // ReLU
            case 3: a1[i] = 1.0 / (1.0 + exp(-x)); break;      // sigmoid
            case 4: a1[i] = (x > 0) ? x : 0.01 * x; break;     // Leaky ReLU
            default: a1[i] = tanh(x); break;
        }
    }
    
    // Forward pass: Hidden -> Output layer
    matmul(a1, W2, z2, num_local, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_local, nn_output_dim);
    softmax(z2, probs, num_local, nn_output_dim);

    // Calculate cross-entropy loss (local sum only)
    double data_loss = 0.0;
    for (int i = 0; i < num_local; i++) {
        int label = y_local[i];
        double p = probs[i * nn_output_dim + label];
        
        // Prevent log(0) -> -inf
        if (p < 1e-10) p = 1e-10;
        
        data_loss += -log(p);
    }

    // Clean up
    free(z1);
    free(a1);
    free(z2);
    free(probs);
    
    // Return LOCAL sum (regularization added later after MPI_Allreduce)
    return data_loss;
}

void build_model_mpi(int nn_hdim, int num_passes, int batch_size, 
                     int print_loss, int rank, int size, int activation_mode)
{
    srand(rank);

    // ----------------------
    // Allocate model weights
    // ----------------------
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));

    // Initialize weights with Xavier/He initialization
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrt(nn_input_dim);

    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrt(nn_hdim);

    // ----------------------
    // Partition data across processes
    // ----------------------
    int base = num_examples / size;
    int rem  = num_examples % size;

    int start = rank * base + (rank < rem ? rank : rem);
    int local_num = base + (rank < rem ? 1 : 0);

    double *X_local = X + start * nn_input_dim;
    int *y_local    = y + start;

    int num_batches = (local_num + batch_size - 1) / batch_size;

    // ----------------------
    // Learning rate decay parameters
    // ----------------------
    int decay_mode = 3;
    double k = 1e-4;
    double gamma = 0.5;
    int step_size = 5000;

    // ----------------------
    // TRAINING LOOP
    // ----------------------
    for (int epoch = 0; epoch < num_passes; epoch++)
    {
        // Allocate local gradient accumulators (initialized to zero)
        double *local_dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
        double *local_db1 = calloc(nn_hdim, sizeof(double));
        double *local_dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
        double *local_db2 = calloc(nn_output_dim, sizeof(double));

        // ----------------------
        // Process local mini-batches
        // ----------------------
        for (int b = 0; b < num_batches; b++) {

            int start_b = b * batch_size;
            int end_b   = (b + 1) * batch_size;
            if (end_b > local_num) end_b = local_num;
            int batch_count = end_b - start_b;

            double *X_batch = X_local + start_b * nn_input_dim;
            int *y_batch    = y_local + start_b;

            // Allocate forward/backward buffers
            double *z1 = calloc(batch_count * nn_hdim, sizeof(double));
            double *a1 = calloc(batch_count * nn_hdim, sizeof(double));
            double *z2 = calloc(batch_count * nn_output_dim, sizeof(double));
            double *probs = calloc(batch_count * nn_output_dim, sizeof(double));
            double *delta3 = calloc(batch_count * nn_output_dim, sizeof(double));
            double *delta2 = calloc(batch_count * nn_hdim, sizeof(double));

            // ---- Forward pass ----
            matmul(X_batch, W1, z1, batch_count, nn_input_dim, nn_hdim);
            add_bias(z1, b1, batch_count, nn_hdim);

            // Apply activation
            for (int i = 0; i < batch_count * nn_hdim; i++) {
                double x = z1[i];
                switch (activation_mode) {
                    case 1: a1[i] = tanh(x); break;
                    case 2: a1[i] = (x > 0) ? x : 0.0; break;
                    case 3: a1[i] = 1.0 / (1.0 + exp(-x)); break;
                    case 4: a1[i] = (x > 0) ? x : 0.01 * x; break;
                    default: a1[i] = tanh(x); break;
                }
            }

            matmul(a1, W2, z2, batch_count, nn_hdim, nn_output_dim);
            add_bias(z2, b2, batch_count, nn_output_dim);
            softmax(z2, probs, batch_count, nn_output_dim);

            // ---- Backpropagation ----
            // Output layer gradient
            for (int i = 0; i < batch_count * nn_output_dim; i++)
                delta3[i] = probs[i];
            for (int i = 0; i < batch_count; i++)
                delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

            // Gradient for W2
            for (int j = 0; j < nn_hdim; j++)
                for (int k = 0; k < nn_output_dim; k++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                    local_dW2[j * nn_output_dim + k] += sum;
                }

            // Gradient for b2
            for (int k = 0; k < nn_output_dim; k++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta3[n * nn_output_dim + k];
                local_db2[k] += sum;
            }

            // Hidden layer gradient (delta2)
            for (int n = 0; n < batch_count; n++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < nn_output_dim; k++)
                        sum += delta3[n * nn_output_dim + k] * W2[j * nn_output_dim + k];

                    double z = z1[n * nn_hdim + j];
                    double a = a1[n * nn_hdim + j];
                    double grad;

                    // Activation derivative
                    switch (activation_mode) {
                        case 1: grad = 1.0 - a * a; break;         // tanh'
                        case 2: grad = (z > 0) ? 1.0 : 0.0; break; // ReLU'
                        case 3: grad = a * (1.0 - a); break;       // sigmoid'
                        case 4: grad = (z > 0) ? 1.0 : 0.01; break;// Leaky ReLU'
                        default: grad = 1.0 - a * a; break;
                    }

                    delta2[n * nn_hdim + j] = sum * grad;
                }

            // Gradient for W1
            for (int i = 0; i < nn_input_dim; i++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += X_batch[n * nn_input_dim + i] * delta2[n * nn_hdim + j];
                    local_dW1[i * nn_hdim + j] += sum;
                }

            // Gradient for b1
            for (int j = 0; j < nn_hdim; j++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta2[n * nn_hdim + j];
                local_db1[j] += sum;
            }

            // Free batch buffers
            free(z1); free(a1); free(z2); free(probs);
            free(delta3); free(delta2);
        }

        // ----------------------
        // MPI Allreduce: Sum gradients across all processes
        // ----------------------
        double *global_dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
        double *global_db1 = calloc(nn_hdim, sizeof(double));
        double *global_dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
        double *global_db2 = calloc(nn_output_dim, sizeof(double));

        MPI_Allreduce(local_dW1, global_dW1, nn_input_dim * nn_hdim, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_db1, global_db1, nn_hdim, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_dW2, global_dW2, nn_hdim * nn_output_dim, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_db2, global_db2, nn_output_dim, 
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // ----------------------
        // Update weights with learning rate decay
        // ----------------------
        double epsilon_t = epsilon;

        if (decay_mode == 1)
            epsilon_t = epsilon / (1.0 + k * epoch);
        else if (decay_mode == 2)
            epsilon_t = epsilon * exp(-k * epoch);
        else if (decay_mode == 3) {
            int step = epoch / step_size;
            epsilon_t = epsilon * pow(gamma, step);
        }

        // CORRECT: Divide by num_examples and add L2 regularization
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            W1[i] -= epsilon_t * (global_dW1[i] / num_examples + reg_lambda * W1[i]);

        for (int i = 0; i < nn_hdim; i++)
            b1[i] -= epsilon_t * global_db1[i] / num_examples;

        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            W2[i] -= epsilon_t * (global_dW2[i] / num_examples + reg_lambda * W2[i]);

        for (int i = 0; i < nn_output_dim; i++)
            b2[i] -= epsilon_t * global_db2[i] / num_examples;

        // ----------------------
        // Optional loss printing
        // ----------------------
        if (print_loss && epoch % 1000 == 0) {
            
            // Each process calculates its local loss sum
            double local_loss = calculate_loss_mpi(W1, b1, W2, b2, nn_hdim,
                                                   X_local, y_local, local_num,
                                                   activation_mode);

            // Sum all local losses across processes
            double global_data_loss = 0.0;
            MPI_Allreduce(&local_loss, &global_data_loss, 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD);

            // Add regularization term (computed once, same on all processes)
            double reg = 0.0;
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                reg += W1[i] * W1[i];
            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                reg += W2[i] * W2[i];
            
            // Final loss: (total cross-entropy + regularization) / total samples
            double global_loss = (global_data_loss + reg_lambda / 2.0 * reg) / num_examples;

            if (rank == 0)
                printf("Epoch %d - Loss: %.6f (lr=%.6f)\n",
                       epoch, global_loss, epsilon_t);
        }

        // Free gradient buffers
        free(local_dW1); free(local_db1);
        free(local_dW2); free(local_db2);
        free(global_dW1); free(global_db1);
        free(global_dW2); free(global_db2);
    }

    // ----------------------
    // Save weights (only rank 0)
    // ----------------------
    if (rank == 0) {
        FILE *fw1 = fopen("output/W1.txt", "w");
        FILE *fb1 = fopen("output/b1.txt", "w");
        FILE *fw2 = fopen("output/W2.txt", "w");
        FILE *fb2 = fopen("output/b2.txt", "w");

        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            fprintf(fw1, "%lf\n", W1[i]);

        for (int i = 0; i < nn_hdim; i++)
            fprintf(fb1, "%lf\n", b1[i]);

        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            fprintf(fw2, "%lf\n", W2[i]);

        for (int i = 0; i < nn_output_dim; i++)
            fprintf(fb2, "%lf\n", b2[i]);

        fclose(fw1); fclose(fb1);
        fclose(fw2); fclose(fb2);

        printf("Weights saved (MPI).\n");
    }

    free(W1); free(W2);
    free(b1); free(b2);
}