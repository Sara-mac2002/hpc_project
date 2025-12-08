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



double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim, int activation_mode){
    // Forward pass on entire dataset
    double *z1 = calloc(num_examples * nn_hdim, sizeof(double));
    double *a1 = calloc(num_examples * nn_hdim, sizeof(double));
    double *z2 = calloc(num_examples * nn_output_dim, sizeof(double));
    double *probs = calloc(num_examples * nn_output_dim, sizeof(double));

    if (!z1 || !a1 || !z2 || !probs) {
        fprintf(stderr, "Error: Memory allocation failed in calculate_loss\n");
        free(z1); free(a1); free(z2); free(probs);
        return -1.0;
    }

    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_examples, nn_hdim);

    // Apply activation function - FIXED MAPPING to match training code
    for (int i = 0; i < num_examples * nn_hdim; i++) {
        double x = z1[i];
        if (activation_mode == 1)         // Tanh
            a1[i] = tanh(x);
        else if (activation_mode == 2)    // ReLU
            a1[i] = (x > 0) ? x : 0.0;
        else if (activation_mode == 3)    // Sigmoid
            a1[i] = 1.0 / (1.0 + exp(-x));
        else if (activation_mode == 4)    // Leaky ReLU
            a1[i] = (x > 0) ? x : 0.01 * x;
    }

    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_examples, nn_output_dim);
    softmax(z2, probs, num_examples, nn_output_dim);

    // Cross-entropy loss with numerical stability
    double data_loss = 0.0;
    for (int i = 0; i < num_examples; i++) {
        int correct_class = y[i];
        double prob = probs[i * nn_output_dim + correct_class];
        
        // Clip probability to avoid log(0)
        if (prob < 1e-10) 
            prob = 1e-10;
        if (prob > 1.0 - 1e-10)
            prob = 1.0 - 1e-10;
            
        data_loss += -log(prob);
    }

    // L2 regularization
    double reg_loss = 0.0;
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        reg_loss += W1[i] * W1[i];
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        reg_loss += W2[i] * W2[i];

    // Total loss: average cross-entropy + regularization
    double total_loss = (data_loss + 0.5 * reg_lambda * reg_loss) / num_examples;

    free(z1); 
    free(a1); 
    free(z2); 
    free(probs);
    
    return total_loss;
}
void build_model_sequential(int nn_hdim, int num_passes, int batch_size, int decay_mode, int activation_mode, int print_loss)
{
    srand(0);

    // ------------------------------
    // Weight Initialization (Activation-Aware)
    // ------------------------------
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));

    // Use appropriate initialization based on activation function
    if (activation_mode == 2 || activation_mode == 4) {
        // He initialization for ReLU and Leaky ReLU
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            W1[i] = randn() * sqrt(2.0 / nn_input_dim);
        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            W2[i] = randn() * sqrt(2.0 / nn_hdim);
    } else {
        // Xavier/Glorot initialization for Tanh and Sigmoid
        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
            W1[i] = randn() / sqrt(nn_input_dim);
        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
            W2[i] = randn() / sqrt(nn_hdim);
    }

    // ------------------------------
    // Allocate Buffers ONCE (Reusable for each mini-batch)
    // ------------------------------
    double *z1     = malloc(batch_size * nn_hdim * sizeof(double));
    double *a1     = malloc(batch_size * nn_hdim * sizeof(double));
    double *z2     = malloc(batch_size * nn_output_dim * sizeof(double));
    double *probs  = malloc(batch_size * nn_output_dim * sizeof(double));

    double *delta3 = malloc(batch_size * nn_output_dim * sizeof(double));
    double *delta2 = malloc(batch_size * nn_hdim * sizeof(double));

    double *dW1    = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *db1    = malloc(nn_hdim * sizeof(double));
    double *dW2    = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *db2    = malloc(nn_output_dim * sizeof(double));

    int num_batches = (num_examples + batch_size - 1) / batch_size;

    // ------------------------------
    // Learning Rate Decay Configuration
    // ------------------------------
    double k = 1.5e-5;           // Decay coefficient
    double gamma = 0.7;          // Step decay factor
    int step_size = 10000;       // Iterations between each step drop

    // ------------------------------
    // Main Training Loop (Mini-batch Gradient Descent)
    // ------------------------------
    for (int epoch = 0; epoch < num_passes; epoch++) {
        for (int b = 0; b < num_batches; b++) {

            // Batch indices
            int start = b * batch_size;
            int end   = (b + 1) * batch_size;
            if (end > num_examples) end = num_examples;
            int batch_count = end - start;

            // Select mini-batch (pointer arithmetic - no copy)
            double *X_batch = X + start * nn_input_dim;
            int    *y_batch = y + start;

            // Reset buffers for this batch
            memset(z1, 0, batch_count * nn_hdim * sizeof(double));
            memset(z2, 0, batch_count * nn_output_dim * sizeof(double));
            memset(probs, 0, batch_count * nn_output_dim * sizeof(double));

            // ==============================
            // FORWARD PROPAGATION
            // ==============================
            matmul(X_batch, W1, z1, batch_count, nn_input_dim, nn_hdim);
            add_bias(z1, b1, batch_count, nn_hdim);
            
            // Apply activation function
            for (int i = 0; i < batch_count * nn_hdim; i++) {
                double x = z1[i];
                if (activation_mode == 1)         // Tanh
                    a1[i] = tanh(x);
                else if (activation_mode == 2)    // ReLU
                    a1[i] = (x > 0) ? x : 0.0;
                else if (activation_mode == 3)    // Sigmoid
                    a1[i] = 1.0 / (1.0 + exp(-x));
                else if (activation_mode == 4)    // Leaky ReLU
                    a1[i] = (x > 0) ? x : 0.01 * x;
            }
            
            matmul(a1, W2, z2, batch_count, nn_hdim, nn_output_dim);
            add_bias(z2, b2, batch_count, nn_output_dim);
            softmax(z2, probs, batch_count, nn_output_dim);

            // ==============================
            // BACKPROPAGATION
            // ==============================
            memset(dW1, 0, nn_input_dim * nn_hdim * sizeof(double));
            memset(dW2, 0, nn_hdim * nn_output_dim * sizeof(double));
            memset(db1, 0, nn_hdim * sizeof(double));
            memset(db2, 0, nn_output_dim * sizeof(double));

            // Output layer gradient: delta3 = probs - y_true
            for (int i = 0; i < batch_count * nn_output_dim; i++)
                delta3[i] = probs[i];
            for (int i = 0; i < batch_count; i++)
                delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

            // dW2 = a1.T * delta3
            for (int j = 0; j < nn_hdim; j++)
                for (int k = 0; k < nn_output_dim; k++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                    dW2[j * nn_output_dim + k] = sum;
                }

            // db2 = sum(delta3)
            for (int k = 0; k < nn_output_dim; k++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta3[n * nn_output_dim + k];
                db2[k] = sum;
            }

            // delta2 = delta3 * W2.T * f'(z1)
            for (int n = 0; n < batch_count; n++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < nn_output_dim; k++)
                        sum += delta3[n * nn_output_dim + k] * W2[j * nn_output_dim + k];

                    // Activation derivative
                    double grad = 1.0;
                    double z = z1[n * nn_hdim + j];
                    double a = a1[n * nn_hdim + j];

                    if (activation_mode == 1)         // Tanh
                        grad = 1.0 - a * a;
                    else if (activation_mode == 2)    // ReLU
                        grad = (z > 0) ? 1.0 : 0.0;
                    else if (activation_mode == 3)    // Sigmoid
                        grad = a * (1.0 - a);
                    else if (activation_mode == 4)    // Leaky ReLU
                        grad = (z > 0) ? 1.0 : 0.01;

                    delta2[n * nn_hdim + j] = sum * grad;
                }

            // dW1 = X.T * delta2
            for (int i = 0; i < nn_input_dim; i++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += X_batch[n * nn_input_dim + i] * delta2[n * nn_hdim + j];
                    dW1[i * nn_hdim + j] = sum;
                }

            // db1 = sum(delta2)
            for (int j = 0; j < nn_hdim; j++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta2[n * nn_hdim + j];
                db1[j] = sum;
            }

            // Add L2 regularization to gradients
            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                dW2[i] += reg_lambda * W2[i];
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                dW1[i] += reg_lambda * W1[i];

            // ==============================
            // Dynamic Learning Rate (Per-Iteration)
            // ==============================
            double epsilon_t = epsilon;
            int iteration = epoch * num_batches + b;
            
            if (decay_mode == 1) {
                // Inverse time decay
                epsilon_t = epsilon / (1.0 + k * iteration);
            } else if (decay_mode == 2) {
                // Exponential decay
                epsilon_t = epsilon * exp(-k * iteration);
            } else if (decay_mode == 3) {
                // Step decay
                int step = iteration / step_size;
                epsilon_t = epsilon * pow(gamma, step);
            }
            // decay_mode == 0: constant learning rate (epsilon_t = epsilon)

            // ==============================
            // Update Weights and Biases
            // ==============================
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                W1[i] -= epsilon_t * dW1[i];
            for (int i = 0; i < nn_hdim; i++)
                b1[i] -= epsilon_t * db1[i];
            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                W2[i] -= epsilon_t * dW2[i];
            for (int i = 0; i < nn_output_dim; i++)
                b2[i] -= epsilon_t * db2[i];
        }

        // Print loss periodically
        if (print_loss && epoch % 1000 == 0) {
            double loss = calculate_loss(W1, b1, W2, b2, nn_hdim, activation_mode);
            printf("Loss after %d epochs: %.6f\n", epoch, loss);
        }
    }

    // ------------------------------
    // Save Weights and Biases
    // ------------------------------
    FILE *fw1 = fopen("output/W1.txt", "w");
    FILE *fb1 = fopen("output/b1.txt", "w");
    FILE *fw2 = fopen("output/W2.txt", "w");
    FILE *fb2 = fopen("output/b2.txt", "w");

    if (!fw1 || !fb1 || !fw2 || !fb2) {
        fprintf(stderr, "Error: Could not open output files\n");
        exit(1);
    }

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
    printf("Weights saved to output/ directory.\n");

    // ------------------------------
    // Free Memory
    // ------------------------------
    free(W1); free(W2);
    free(b1); free(b2);
    free(z1); free(a1); free(z2); free(probs);
    free(delta3); free(delta2);
    free(dW1); free(dW2);
    free(db1); free(db2);
    
    printf("Memory released successfully.\n");
}
