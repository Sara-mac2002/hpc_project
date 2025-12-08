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



double calculate_loss(double *W1, double *b1, double *W2, double *b2, 
                     int nn_hdim, int activation_mode)
{
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


void build_model_omp(int nn_hdim, int num_passes, int batch_size,int decay_mode, int activation_mode,  int print_loss)
{
    srand(0);

    // ------------------------------
    // Initialisation des poids
    // ------------------------------
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));

    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrt(nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrt(nn_hdim);

    int num_batches = (num_examples + batch_size - 1) / batch_size;

    // ------------------------------
    // Hyperparameters for decay
    // ------------------------------
    double k = 1e-4;
    double gamma = 0.5;
    int step_size = 5000;

    // ------------------------------
    // Entraînement
    // ------------------------------
    for (int epoch = 0; epoch < num_passes; epoch++) {

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < num_batches; b++) {

            int start = b * batch_size;
            int end = (b + 1) * batch_size;
            if (end > num_examples) end = num_examples;
            int batch_count = end - start;

            double *X_batch = X + start * nn_input_dim;
            int *y_batch = y + start;

            double *z1 = calloc(batch_count * nn_hdim, sizeof(double));
            double *a1 = calloc(batch_count * nn_hdim, sizeof(double));
            double *z2 = calloc(batch_count * nn_output_dim, sizeof(double));
            double *probs = calloc(batch_count * nn_output_dim, sizeof(double));
            double *delta3 = calloc(batch_count * nn_output_dim, sizeof(double));
            double *delta2 = calloc(batch_count * nn_hdim, sizeof(double));
            double *dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
            double *dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
            double *db1 = calloc(nn_hdim, sizeof(double));
            double *db2 = calloc(nn_output_dim, sizeof(double));

            // ------------------------------
            // Forward propagation
            // ------------------------------
            matmul(X_batch, W1, z1, batch_count, nn_input_dim, nn_hdim);
            add_bias(z1, b1, batch_count, nn_hdim);
            for (int i = 0; i < batch_count * nn_hdim; i++) {
                double x = z1[i];
                if (activation_mode == 0)      a1[i] = tanh(x);
                else if (activation_mode == 1) a1[i] = (x > 0) ? x : 0.0;
                else if (activation_mode == 2) a1[i] = 1.0 / (1.0 + exp(-x));
                else if (activation_mode == 3) a1[i] = (x > 0) ? x : 0.01 * x;
            }

            matmul(a1, W2, z2, batch_count, nn_hdim, nn_output_dim);
            add_bias(z2, b2, batch_count, nn_output_dim);
            softmax(z2, probs, batch_count, nn_output_dim);

            // ------------------------------
            // Backpropagation
            // ------------------------------
            for (int i = 0; i < batch_count * nn_output_dim; i++)
                delta3[i] = probs[i];
            for (int i = 0; i < batch_count; i++)
                delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

            for (int j = 0; j < nn_hdim; j++)
                for (int k = 0; k < nn_output_dim; k++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                    dW2[j * nn_output_dim + k] = sum;
                }

            for (int k = 0; k < nn_output_dim; k++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta3[n * nn_output_dim + k];
                db2[k] = sum;
            }

            for (int n = 0; n < batch_count; n++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < nn_output_dim; k++)
                        sum += delta3[n * nn_output_dim + k] * W2[j * nn_output_dim + k];

                    double grad = 1.0;
                    double z = z1[n * nn_hdim + j];
                    double a = a1[n * nn_hdim + j];

                    if (activation_mode == 0)      grad = 1.0 - a * a;
                    else if (activation_mode == 1) grad = (z > 0) ? 1.0 : 0.0;
                    else if (activation_mode == 2) grad = a * (1.0 - a);
                    else if (activation_mode == 3) grad = (z > 0) ? 1.0 : 0.01;

                    delta2[n * nn_hdim + j] = sum * grad;
                }

            for (int i = 0; i < nn_input_dim; i++)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int n = 0; n < batch_count; n++)
                        sum += X_batch[n * nn_input_dim + i] * delta2[n * nn_hdim + j];
                    dW1[i * nn_hdim + j] = sum;
                }

            for (int j = 0; j < nn_hdim; j++) {
                double sum = 0.0;
                for (int n = 0; n < batch_count; n++)
                    sum += delta2[n * nn_hdim + j];
                db1[j] = sum;
            }

            // Regularization
            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                dW2[i] += reg_lambda * W2[i];
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                dW1[i] += reg_lambda * W1[i];

            // Learning rate decay
            double epsilon_t = epsilon;
            if (decay_mode == 1)
                epsilon_t = epsilon / (1.0 + k * (epoch * num_batches + b));
            else if (decay_mode == 2)
                epsilon_t = epsilon * exp(-k * (epoch * num_batches + b));
            else if (decay_mode == 3) {
                int step = (epoch * num_batches + b) / step_size;
                epsilon_t = epsilon * pow(gamma, step);
            }

            // Update
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                W1[i] -= epsilon_t * dW1[i];
            for (int i = 0; i < nn_hdim; i++)
                b1[i] -= epsilon_t * db1[i];
            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                W2[i] -= epsilon_t * dW2[i];
            for (int i = 0; i < nn_output_dim; i++)
                b2[i] -= epsilon_t * db2[i];

            free(z1); free(a1); free(z2); free(probs);
            free(delta3); free(delta2);
            free(dW1); free(dW2);
            free(db1); free(db2);
        }
    }

    // Sauvegarde
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
    printf("Poids sauvegardés dans W1.txt, b1.txt, W2.txt, b2.txt\n");

    free(W1); free(W2); free(b1); free(b2);
}

void build_model_task(int nn_hdim, int num_passes, int batch_size,int decay_mode, int activation_mode, int print_loss)
{
    srand(0);

    // ------------------------------
    // Initialize parameters (shared model)
    // ------------------------------
    double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
    double *b1 = calloc(nn_hdim, sizeof(double));
    double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim, sizeof(double));

    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrt(nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrt(nn_hdim);

    int num_batches = (num_examples + batch_size - 1) / batch_size;

    // Learning rate decay parameters
    double k = 1e-4;
    double gamma = 0.5;
    int step_size = 5000;

    // ------------------------------
    // Training loop
    // ------------------------------
    for (int epoch = 0; epoch < num_passes; epoch++) {

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int b = 0; b < num_batches; b++) {
                    #pragma omp task firstprivate(b, epoch)
                    {
                        int start = b * batch_size;
                        int end = (b + 1) * batch_size;
                        if (end > num_examples) end = num_examples;
                        int batch_count = end - start;

                        double *X_batch = X + start * nn_input_dim;
                        int *y_batch = y + start;

                        // ------------------------------
                        // LOCAL COPY OF WEIGHTS (avoid data races)
                        // ------------------------------
                        double *W1_local = malloc(nn_input_dim * nn_hdim * sizeof(double));
                        double *b1_local = malloc(nn_hdim * sizeof(double));
                        double *W2_local = malloc(nn_hdim * nn_output_dim * sizeof(double));
                        double *b2_local = malloc(nn_output_dim * sizeof(double));

                        memcpy(W1_local, W1, nn_input_dim * nn_hdim * sizeof(double));
                        memcpy(b1_local, b1, nn_hdim * sizeof(double));
                        memcpy(W2_local, W2, nn_hdim * nn_output_dim * sizeof(double));
                        memcpy(b2_local, b2, nn_output_dim * sizeof(double));

                        // ------------------------------
                        // Local buffers and gradients
                        // ------------------------------
                        double *z1   = calloc(batch_count * nn_hdim,        sizeof(double));
                        double *a1   = calloc(batch_count * nn_hdim,        sizeof(double));
                        double *z2   = calloc(batch_count * nn_output_dim,  sizeof(double));
                        double *probs= calloc(batch_count * nn_output_dim,  sizeof(double));
                        double *delta3 = calloc(batch_count * nn_output_dim,sizeof(double));
                        double *delta2 = calloc(batch_count * nn_hdim,      sizeof(double));
                        double *dW1  = calloc(nn_input_dim * nn_hdim,       sizeof(double));
                        double *dW2  = calloc(nn_hdim * nn_output_dim,      sizeof(double));
                        double *db1  = calloc(nn_hdim,                      sizeof(double));
                        double *db2  = calloc(nn_output_dim,                sizeof(double));

                        // ------------------------------
                        // Forward pass (using local weights)
                        // ------------------------------
                        matmul(X_batch, W1_local, z1, batch_count, nn_input_dim, nn_hdim);
                        add_bias(z1, b1_local, batch_count, nn_hdim);

                        for (int i = 0; i < batch_count * nn_hdim; i++) {
                            double x = z1[i];
                            switch (activation_mode) {
                                case 1: a1[i] = 1.0 / (1.0 + exp(-x)); break;  // Sigmoid
                                case 2: a1[i] = (x > 0) ? x : 0.0; break;      // ReLU
                                case 3: a1[i] = (x > 0) ? x : 0.01 * x; break; // Leaky ReLU
                                default: a1[i] = tanh(x); break;               // Tanh
                            }
                        }

                        matmul(a1, W2_local, z2, batch_count, nn_hdim, nn_output_dim);
                        add_bias(z2, b2_local, batch_count, nn_output_dim);
                        softmax(z2, probs, batch_count, nn_output_dim);

                        // ------------------------------
                        // Backpropagation
                        // ------------------------------
                        for (int i = 0; i < batch_count * nn_output_dim; i++)
                            delta3[i] = probs[i];
                        for (int i = 0; i < batch_count; i++)
                            delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

                        // dW2, db2
                        for (int j = 0; j < nn_hdim; j++)
                            for (int k = 0; k < nn_output_dim; k++) {
                                double sum = 0.0;
                                for (int n = 0; n < batch_count; n++)
                                    sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                                dW2[j * nn_output_dim + k] = sum;
                            }

                        for (int k = 0; k < nn_output_dim; k++) {
                            double sum = 0.0;
                            for (int n = 0; n < batch_count; n++)
                                sum += delta3[n * nn_output_dim + k];
                            db2[k] = sum;
                        }

                        // delta2
                        for (int n = 0; n < batch_count; n++)
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < nn_output_dim; k++)
                                    sum += delta3[n * nn_output_dim + k]
                                         * W2_local[j * nn_output_dim + k];

                                double z = z1[n * nn_hdim + j];
                                double a = a1[n * nn_hdim + j];
                                double grad;
                                switch (activation_mode) {
                                    case 1: grad = a * (1.0 - a); break;        // Sigmoid'
                                    case 2: grad = (z > 0) ? 1.0 : 0.0; break;  // ReLU'
                                    case 3: grad = (z > 0) ? 1.0 : 0.01; break; // Leaky ReLU'
                                    default: grad = 1.0 - a * a; break;         // Tanh'
                                }
                                delta2[n * nn_hdim + j] = sum * grad;
                            }

                        // dW1, db1
                        for (int i = 0; i < nn_input_dim; i++)
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                for (int n = 0; n < batch_count; n++)
                                    sum += X_batch[n * nn_input_dim + i]
                                         * delta2[n * nn_hdim + j];
                                dW1[i * nn_hdim + j] = sum;
                            }

                        for (int j = 0; j < nn_hdim; j++) {
                            double sum = 0.0;
                            for (int n = 0; n < batch_count; n++)
                                sum += delta2[n * nn_hdim + j];
                            db1[j] = sum;
                        }

                        // L2 regularization (use local weights snapshot)
                        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                            dW2[i] += reg_lambda * W2_local[i];
                        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                            dW1[i] += reg_lambda * W1_local[i];

                        // ------------------------------
                        // Learning rate decay
                        // ------------------------------
                        double epsilon_t = epsilon;
                        if (decay_mode == 1)
                            epsilon_t = epsilon / (1.0 + k * (epoch * num_batches + b));
                        else if (decay_mode == 2)
                            epsilon_t = epsilon * exp(-k * (epoch * num_batches + b));
                        else if (decay_mode == 3) {
                            int step = (epoch * num_batches + b) / step_size;
                            epsilon_t = epsilon * pow(gamma, step);
                        }

                        // ------------------------------
                        // Merge updates on SHARED weights
                        // ------------------------------
                        #pragma omp critical
                        {
                            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                                W1[i] -= epsilon_t * dW1[i];
                            for (int j = 0; j < nn_hdim; j++)
                                b1[j] -= epsilon_t * db1[j];
                            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                                W2[i] -= epsilon_t * dW2[i];
                            for (int j = 0; j < nn_output_dim; j++)
                                b2[j] -= epsilon_t * db2[j];
                        }

                        // Cleanup
                        free(W1_local); free(b1_local);
                        free(W2_local); free(b2_local);
                        free(z1); free(a1); free(z2); free(probs);
                        free(delta3); free(delta2);
                        free(dW1); free(dW2);
                        free(db1); free(db2);
                    } // end task
                }     // end batches

                #pragma omp taskwait
            } // single
        } // parallel
    } // epochs

    printf("Training finished with OpenMP tasks.\n");

    // Save weights
    FILE *fw1 = fopen("output/W1.txt", "w");
    FILE *fb1 = fopen("output/b1.txt", "w");
    FILE *fw2 = fopen("output/W2.txt", "w");
    FILE *fb2 = fopen("output/b2.txt", "w");
    for (int i = 0; i < nn_input_dim * nn_hdim; i++) fprintf(fw1, "%lf\n", W1[i]);
    for (int i = 0; i < nn_hdim; i++) fprintf(fb1, "%lf\n", b1[i]);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++) fprintf(fw2, "%lf\n", W2[i]);
    for (int i = 0; i < nn_output_dim; i++) fprintf(fb2, "%lf\n", b2[i]);
    fclose(fw1); fclose(fb1); fclose(fw2); fclose(fb2);

    free(W1); free(W2); free(b1); free(b2);
}




// double calculate_loss_mpi(double *W1, double *b1, double *W2, double *b2, 
//                           int nn_hdim, double *X_local, int *y_local, 
//                           int num_local, int activation_mode) 
// {
//     // Allocate forward pass buffers
//     double *z1 = calloc(num_local * nn_hdim, sizeof(double));
//     double *a1 = calloc(num_local * nn_hdim, sizeof(double));
//     double *z2 = calloc(num_local * nn_output_dim, sizeof(double));
//     double *probs = calloc(num_local * nn_output_dim, sizeof(double));

//     // Forward pass: Input -> Hidden layer
//     matmul(X_local, W1, z1, num_local, nn_input_dim, nn_hdim);
//     add_bias(z1, b1, num_local, nn_hdim);
    
//     // Apply activation function
//     for (int i = 0; i < num_local * nn_hdim; i++) {
//         double x = z1[i];
//         switch (activation_mode) {
//             case 1: a1[i] = tanh(x); break;                    // tanh
//             case 2: a1[i] = (x > 0) ? x : 0.0; break;          // ReLU
//             case 3: a1[i] = 1.0 / (1.0 + exp(-x)); break;      // sigmoid
//             case 4: a1[i] = (x > 0) ? x : 0.01 * x; break;     // Leaky ReLU
//             default: a1[i] = tanh(x); break;
//         }
//     }
    
//     // Forward pass: Hidden -> Output layer
//     matmul(a1, W2, z2, num_local, nn_hdim, nn_output_dim);
//     add_bias(z2, b2, num_local, nn_output_dim);
//     softmax(z2, probs, num_local, nn_output_dim);

//     // Calculate cross-entropy loss (local sum only)
//     double data_loss = 0.0;
//     for (int i = 0; i < num_local; i++) {
//         int label = y_local[i];
//         double p = probs[i * nn_output_dim + label];
        
//         // Prevent log(0) -> -inf
//         if (p < 1e-10) p = 1e-10;
        
//         data_loss += -log(p);
//     }

//     // Clean up
//     free(z1);
//     free(a1);
//     free(z2);
//     free(probs);
    
//     // Return LOCAL sum (regularization added later after MPI_Allreduce)
//     return data_loss;
// }

// void build_model_mpi(int nn_hdim, int num_passes, int batch_size, 
//                      int print_loss, int rank, int size, int activation_mode)
// {
//     srand(rank);

//     // ----------------------
//     // Allocate model weights
//     // ----------------------
//     double *W1 = malloc(nn_input_dim * nn_hdim * sizeof(double));
//     double *b1 = calloc(nn_hdim, sizeof(double));
//     double *W2 = malloc(nn_hdim * nn_output_dim * sizeof(double));
//     double *b2 = calloc(nn_output_dim, sizeof(double));

//     // Initialize weights with Xavier/He initialization
//     for (int i = 0; i < nn_input_dim * nn_hdim; i++)
//         W1[i] = randn() / sqrt(nn_input_dim);

//     for (int i = 0; i < nn_hdim * nn_output_dim; i++)
//         W2[i] = randn() / sqrt(nn_hdim);

//     // ----------------------
//     // Partition data across processes
//     // ----------------------
//     int base = num_examples / size;
//     int rem  = num_examples % size;

//     int start = rank * base + (rank < rem ? rank : rem);
//     int local_num = base + (rank < rem ? 1 : 0);

//     double *X_local = X + start * nn_input_dim;
//     int *y_local    = y + start;

//     int num_batches = (local_num + batch_size - 1) / batch_size;

//     // ----------------------
//     // Learning rate decay parameters
//     // ----------------------
//     int decay_mode = 3;
//     double k = 1e-4;
//     double gamma = 0.5;
//     int step_size = 5000;

//     // ----------------------
//     // TRAINING LOOP
//     // ----------------------
//     for (int epoch = 0; epoch < num_passes; epoch++)
//     {
//         // Allocate local gradient accumulators (initialized to zero)
//         double *local_dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
//         double *local_db1 = calloc(nn_hdim, sizeof(double));
//         double *local_dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
//         double *local_db2 = calloc(nn_output_dim, sizeof(double));

//         // ----------------------
//         // Process local mini-batches
//         // ----------------------
//         for (int b = 0; b < num_batches; b++) {

//             int start_b = b * batch_size;
//             int end_b   = (b + 1) * batch_size;
//             if (end_b > local_num) end_b = local_num;
//             int batch_count = end_b - start_b;

//             double *X_batch = X_local + start_b * nn_input_dim;
//             int *y_batch    = y_local + start_b;

//             // Allocate forward/backward buffers
//             double *z1 = calloc(batch_count * nn_hdim, sizeof(double));
//             double *a1 = calloc(batch_count * nn_hdim, sizeof(double));
//             double *z2 = calloc(batch_count * nn_output_dim, sizeof(double));
//             double *probs = calloc(batch_count * nn_output_dim, sizeof(double));
//             double *delta3 = calloc(batch_count * nn_output_dim, sizeof(double));
//             double *delta2 = calloc(batch_count * nn_hdim, sizeof(double));

//             // ---- Forward pass ----
//             matmul(X_batch, W1, z1, batch_count, nn_input_dim, nn_hdim);
//             add_bias(z1, b1, batch_count, nn_hdim);

//             // Apply activation
//             for (int i = 0; i < batch_count * nn_hdim; i++) {
//                 double x = z1[i];
//                 switch (activation_mode) {
//                     case 1: a1[i] = tanh(x); break;
//                     case 2: a1[i] = (x > 0) ? x : 0.0; break;
//                     case 3: a1[i] = 1.0 / (1.0 + exp(-x)); break;
//                     case 4: a1[i] = (x > 0) ? x : 0.01 * x; break;
//                     default: a1[i] = tanh(x); break;
//                 }
//             }

//             matmul(a1, W2, z2, batch_count, nn_hdim, nn_output_dim);
//             add_bias(z2, b2, batch_count, nn_output_dim);
//             softmax(z2, probs, batch_count, nn_output_dim);

//             // ---- Backpropagation ----
//             // Output layer gradient
//             for (int i = 0; i < batch_count * nn_output_dim; i++)
//                 delta3[i] = probs[i];
//             for (int i = 0; i < batch_count; i++)
//                 delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

//             // Gradient for W2
//             for (int j = 0; j < nn_hdim; j++)
//                 for (int k = 0; k < nn_output_dim; k++) {
//                     double sum = 0.0;
//                     for (int n = 0; n < batch_count; n++)
//                         sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
//                     local_dW2[j * nn_output_dim + k] += sum;
//                 }

//             // Gradient for b2
//             for (int k = 0; k < nn_output_dim; k++) {
//                 double sum = 0.0;
//                 for (int n = 0; n < batch_count; n++)
//                     sum += delta3[n * nn_output_dim + k];
//                 local_db2[k] += sum;
//             }

//             // Hidden layer gradient (delta2)
//             for (int n = 0; n < batch_count; n++)
//                 for (int j = 0; j < nn_hdim; j++) {
//                     double sum = 0.0;
//                     for (int k = 0; k < nn_output_dim; k++)
//                         sum += delta3[n * nn_output_dim + k] * W2[j * nn_output_dim + k];

//                     double z = z1[n * nn_hdim + j];
//                     double a = a1[n * nn_hdim + j];
//                     double grad;

//                     // Activation derivative
//                     switch (activation_mode) {
//                         case 1: grad = 1.0 - a * a; break;         // tanh'
//                         case 2: grad = (z > 0) ? 1.0 : 0.0; break; // ReLU'
//                         case 3: grad = a * (1.0 - a); break;       // sigmoid'
//                         case 4: grad = (z > 0) ? 1.0 : 0.01; break;// Leaky ReLU'
//                         default: grad = 1.0 - a * a; break;
//                     }

//                     delta2[n * nn_hdim + j] = sum * grad;
//                 }

//             // Gradient for W1
//             for (int i = 0; i < nn_input_dim; i++)
//                 for (int j = 0; j < nn_hdim; j++) {
//                     double sum = 0.0;
//                     for (int n = 0; n < batch_count; n++)
//                         sum += X_batch[n * nn_input_dim + i] * delta2[n * nn_hdim + j];
//                     local_dW1[i * nn_hdim + j] += sum;
//                 }

//             // Gradient for b1
//             for (int j = 0; j < nn_hdim; j++) {
//                 double sum = 0.0;
//                 for (int n = 0; n < batch_count; n++)
//                     sum += delta2[n * nn_hdim + j];
//                 local_db1[j] += sum;
//             }

//             // Free batch buffers
//             free(z1); free(a1); free(z2); free(probs);
//             free(delta3); free(delta2);
//         }

//         // ----------------------
//         // MPI Allreduce: Sum gradients across all processes
//         // ----------------------
//         double *global_dW1 = calloc(nn_input_dim * nn_hdim, sizeof(double));
//         double *global_db1 = calloc(nn_hdim, sizeof(double));
//         double *global_dW2 = calloc(nn_hdim * nn_output_dim, sizeof(double));
//         double *global_db2 = calloc(nn_output_dim, sizeof(double));

//         MPI_Allreduce(local_dW1, global_dW1, nn_input_dim * nn_hdim, 
//                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//         MPI_Allreduce(local_db1, global_db1, nn_hdim, 
//                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//         MPI_Allreduce(local_dW2, global_dW2, nn_hdim * nn_output_dim, 
//                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//         MPI_Allreduce(local_db2, global_db2, nn_output_dim, 
//                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

//         // ----------------------
//         // Update weights with learning rate decay
//         // ----------------------
//         double epsilon_t = epsilon;

//         if (decay_mode == 1)
//             epsilon_t = epsilon / (1.0 + k * epoch);
//         else if (decay_mode == 2)
//             epsilon_t = epsilon * exp(-k * epoch);
//         else if (decay_mode == 3) {
//             int step = epoch / step_size;
//             epsilon_t = epsilon * pow(gamma, step);
//         }

//         // CORRECT: Divide by num_examples and add L2 regularization
//         for (int i = 0; i < nn_input_dim * nn_hdim; i++)
//             W1[i] -= epsilon_t * (global_dW1[i] / num_examples + reg_lambda * W1[i]);

//         for (int i = 0; i < nn_hdim; i++)
//             b1[i] -= epsilon_t * global_db1[i] / num_examples;

//         for (int i = 0; i < nn_hdim * nn_output_dim; i++)
//             W2[i] -= epsilon_t * (global_dW2[i] / num_examples + reg_lambda * W2[i]);

//         for (int i = 0; i < nn_output_dim; i++)
//             b2[i] -= epsilon_t * global_db2[i] / num_examples;

//         // ----------------------
//         // Optional loss printing
//         // ----------------------
//         if (print_loss && epoch % 1000 == 0) {
            
//             // Each process calculates its local loss sum
//             double local_loss = calculate_loss_mpi(W1, b1, W2, b2, nn_hdim,
//                                                    X_local, y_local, local_num,
//                                                    activation_mode);

//             // Sum all local losses across processes
//             double global_data_loss = 0.0;
//             MPI_Allreduce(&local_loss, &global_data_loss, 1, MPI_DOUBLE,
//                           MPI_SUM, MPI_COMM_WORLD);

//             // Add regularization term (computed once, same on all processes)
//             double reg = 0.0;
//             for (int i = 0; i < nn_input_dim * nn_hdim; i++)
//                 reg += W1[i] * W1[i];
//             for (int i = 0; i < nn_hdim * nn_output_dim; i++)
//                 reg += W2[i] * W2[i];
            
//             // Final loss: (total cross-entropy + regularization) / total samples
//             double global_loss = (global_data_loss + reg_lambda / 2.0 * reg) / num_examples;

//             if (rank == 0)
//                 printf("Epoch %d - Loss: %.6f (lr=%.6f)\n",
//                        epoch, global_loss, epsilon_t);
//         }

//         // Free gradient buffers
//         free(local_dW1); free(local_db1);
//         free(local_dW2); free(local_db2);
//         free(global_dW1); free(global_db1);
//         free(global_dW2); free(global_db2);
//     }

//     // ----------------------
//     // Save weights (only rank 0)
//     // ----------------------
//     if (rank == 0) {
//         FILE *fw1 = fopen("output/W1.txt", "w");
//         FILE *fb1 = fopen("output/b1.txt", "w");
//         FILE *fw2 = fopen("output/W2.txt", "w");
//         FILE *fb2 = fopen("output/b2.txt", "w");

//         for (int i = 0; i < nn_input_dim * nn_hdim; i++)
//             fprintf(fw1, "%lf\n", W1[i]);

//         for (int i = 0; i < nn_hdim; i++)
//             fprintf(fb1, "%lf\n", b1[i]);

//         for (int i = 0; i < nn_hdim * nn_output_dim; i++)
//             fprintf(fw2, "%lf\n", W2[i]);

//         for (int i = 0; i < nn_output_dim; i++)
//             fprintf(fb2, "%lf\n", b2[i]);

//         fclose(fw1); fclose(fb1);
//         fclose(fw2); fclose(fb2);

//         printf("Weights saved (MPI).\n");
//     }

//     free(W1); free(W2);
//     free(b1); free(b2);
// }