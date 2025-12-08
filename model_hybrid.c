#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

// -----------------------------------------------------------
// Global variables (declared extern in model.h)
// -----------------------------------------------------------
int num_examples;
int nn_input_dim;
int nn_output_dim;
double reg_lambda   = 0.01;
double epsilon      = 0.01;
double *X           = NULL;
int    *y           = NULL;
int activation_mode = 1;

// -----------------------------------------------------------
// Loss on GLOBAL data (rank 0 only typically)
// If you want local loss, you can switch to X_local, y_local.
// -----------------------------------------------------------
double calculate_loss(double *W1, double *b1,
                      double *W2, double *b2,
                      int nn_hdim)
{
    double *z1    = calloc(num_examples * nn_hdim,       sizeof(double));
    double *a1    = calloc(num_examples * nn_hdim,       sizeof(double));
    double *z2    = calloc(num_examples * nn_output_dim, sizeof(double));
    double *probs = calloc(num_examples * nn_output_dim, sizeof(double));

    if (!z1 || !a1 || !z2 || !probs) {
        fprintf(stderr, "Allocation failed in calculate_loss\n");
        return 1e30;
    }

    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_examples, nn_hdim);

    #pragma omp parallel for
    for (int i = 0; i < num_examples * nn_hdim; i++) {
        double x = z1[i];
        if      (activation_mode == 1) a1[i] = tanh(x);
        else if (activation_mode == 2) a1[i] = (x > 0) ? x : 0.0;
        else if (activation_mode == 3) a1[i] = 1.0 / (1.0 + exp(-x));
        else if (activation_mode == 4) a1[i] = (x > 0) ? x : 0.01 * x;
    }

    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_examples, nn_output_dim);
    softmax(z2, probs, num_examples, nn_output_dim);

    double data_loss = 0.0;
    #pragma omp parallel for reduction(+:data_loss)
    for (int i = 0; i < num_examples; i++) {
        int label = y[i];
        double p  = probs[i * nn_output_dim + label];
        data_loss += -log(p);
    }

    double reg = 0.0;
    for (int i = 0; i < nn_input_dim * nn_hdim;       i++) reg += W1[i] * W1[i];
    for (int i = 0; i < nn_hdim      * nn_output_dim; i++) reg += W2[i] * W2[i];
    data_loss += reg_lambda / 2.0 * reg;

    double loss = data_loss / num_examples;

    free(z1);
    free(a1);
    free(z2);
    free(probs);
    return loss;
}

// -----------------------------------------------------------
// build_model: MPI data split + OpenMP tasks + per-node per-batch update
// -----------------------------------------------------------

void build_model(int nn_hdim, int num_passes, int batch_size,
                 int print_loss, int rank, int size)
{
    // ------------------------
    // Initialize parameters
    // ------------------------
    srand(0);  // or srand(rank) if you want different init (but then models differ immediately)

    double *W1 = malloc(nn_input_dim * nn_hdim       * sizeof(double));
    double *b1 = calloc(nn_hdim,                      sizeof(double));
    double *W2 = malloc(nn_hdim      * nn_output_dim * sizeof(double));
    double *b2 = calloc(nn_output_dim,                sizeof(double));

    if (!W1 || !b1 || !W2 || !b2) {
        fprintf(stderr, "[Rank %d] Parameter allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < nn_input_dim * nn_hdim;       i++)
        W1[i] = randn() / sqrt(nn_input_dim);
    for (int i = 0; i < nn_hdim      * nn_output_dim; i++)
        W2[i] = randn() / sqrt(nn_hdim);

    // -------------------------------------------
    // Partition dataset across MPI ranks (nodes)
    // Each rank gets a contiguous chunk
    // -------------------------------------------
    int base = num_examples / size;
    int rem  = num_examples % size;

    int start = rank * base + (rank < rem ? rank : rem);
    int local_num = base + (rank < rem ? 1 : 0);

    double *X_local = X + start * nn_input_dim;
    int    *y_local = y + start;

    int num_batches = (local_num + batch_size - 1) / batch_size;
    int decay_mode = 0;        // 0 = constant, 1 = inverse time, 2 = exponential, 3 = step
    double k = 1.5e-5;           // decay coefficient
    double gamma = 0.7;        // step decay factor
    int step_size = 10000;
   

    // -----------------------------------------
    // Training loop: per-node, per-batch update
    // -----------------------------------------
    for (int epoch = 0; epoch < num_passes; epoch++) {

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int b = 0; b < num_batches; b++) {

                    #pragma omp task firstprivate(b)
                    {
                        int start_b = b * batch_size;
                        int end_b   = (b + 1) * batch_size;
                        if (end_b > local_num) end_b = local_num;
                        int batch_count = end_b - start_b;
                        if (batch_count <= 0) return;

                        double *X_batch = X_local + start_b * nn_input_dim;
                        int    *y_batch = y_local + start_b;

                        // Per-task buffers
                        double *z1    = calloc(batch_count * nn_hdim,       sizeof(double));
                        double *a1    = calloc(batch_count * nn_hdim,       sizeof(double));
                        double *z2    = calloc(batch_count * nn_output_dim, sizeof(double));
                        double *probs = calloc(batch_count * nn_output_dim, sizeof(double));
                        double *delta3= calloc(batch_count * nn_output_dim, sizeof(double));
                        double *delta2= calloc(batch_count * nn_hdim,       sizeof(double));
                        double *local_dW1 = calloc(nn_input_dim * nn_hdim,       sizeof(double));
                        double *local_db1 = calloc(nn_hdim,                      sizeof(double));
                        double *local_dW2 = calloc(nn_hdim      * nn_output_dim, sizeof(double));
                        double *local_db2 = calloc(nn_output_dim,                sizeof(double));

                        if (!z1 || !a1 || !z2 || !probs || !delta3 || !delta2 ||
                            !local_dW1 || !local_db1 || !local_dW2 || !local_db2) {
                            fprintf(stderr, "[Rank %d] Allocation failed in task\n", rank);
                            // safest is to abort; we are inside a task, so be careful
                            return;
                        }

                        // ---------- Forward ----------
                        matmul(X_batch, W1, z1, batch_count, nn_input_dim, nn_hdim);
                        add_bias(z1, b1, batch_count, nn_hdim);

                        #pragma omp simd
                        for (int i = 0; i < batch_count * nn_hdim; i++) {
                            double x = z1[i];
                            if      (activation_mode == 1) a1[i] = tanh(x);
                            else if (activation_mode == 2) a1[i] = (x > 0) ? x : 0.0;
                            else if (activation_mode == 3) a1[i] = 1.0 / (1.0 + exp(-x));
                            else if (activation_mode == 4) a1[i] = (x > 0) ? x : 0.01 * x;
                        }

                        matmul(a1, W2, z2, batch_count, nn_hdim, nn_output_dim);
                        add_bias(z2, b2, batch_count, nn_output_dim);
                        softmax(z2, probs, batch_count, nn_output_dim);

                        // ---------- Backward ----------
                        // delta3 = probs - one_hot
                        #pragma omp simd
                        for (int i = 0; i < batch_count * nn_output_dim; i++)
                            delta3[i] = probs[i];
                        for (int i = 0; i < batch_count; i++)
                            delta3[i * nn_output_dim + y_batch[i]] -= 1.0;

                        // dW2
                        for (int j = 0; j < nn_hdim; j++) {
                            for (int k = 0; k < nn_output_dim; k++) {
                                double sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int n = 0; n < batch_count; n++)
                                    sum += a1[n * nn_hdim + j] * delta3[n * nn_output_dim + k];
                                local_dW2[j * nn_output_dim + k] += sum;
                            }
                        }

                        // db2
                        for (int k = 0; k < nn_output_dim; k++) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int n = 0; n < batch_count; n++)
                                sum += delta3[n * nn_output_dim + k];
                            local_db2[k] += sum;
                        }

                        // delta2
                        for (int n = 0; n < batch_count; n++) {
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int k = 0; k < nn_output_dim; k++)
                                    sum += delta3[n * nn_output_dim + k] *
                                           W2[j * nn_output_dim + k];
                                double grad = 1.0;
                                double z    = z1[n * nn_hdim + j];
                                double a    = a1[n * nn_hdim + j];
                                if      (activation_mode == 1) grad = 1.0 - a * a;
                                else if (activation_mode == 2) grad = (z > 0) ? 1.0 : 0.0;
                                else if (activation_mode == 3) grad = a * (1.0 - a);
                                else if (activation_mode == 4) grad = (z > 0) ? 1.0 : 0.01;
                                delta2[n * nn_hdim + j] = sum * grad;
                            }
                        }

                        // dW1
                        for (int i = 0; i < nn_input_dim; i++) {
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (int n = 0; n < batch_count; n++)
                                    sum += X_batch[n * nn_input_dim + i] *
                                           delta2[n * nn_hdim + j];
                                local_dW1[i * nn_hdim + j] += sum;
                            }
                        }

                        // db1
                        for (int j = 0; j < nn_hdim; j++) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int n = 0; n < batch_count; n++)
                                sum += delta2[n * nn_hdim + j];
                            local_db1[j] += sum;
                        }

                        // Add regularization to gradients
                        for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                            local_dW2[i] += reg_lambda * W2[i];
                        for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                            local_dW1[i] += reg_lambda * W1[i];

                        // ---------- Per-node weight update (critical) ----------
                        #pragma omp critical
                        {
                            double epsilon_t = epsilon;
                            int global_step = epoch * num_batches + b;
                            if (decay_mode == 1) {
                                epsilon_t = epsilon / (1.0 + k * global_step);
                            } else if (decay_mode == 2) {
                                epsilon_t = epsilon * exp(-k * global_step);
                            } else if (decay_mode == 3) {
                                int step = global_step / step_size;
                                epsilon_t = epsilon * pow(gamma, step);
                            }

                            for (int i = 0; i < nn_input_dim * nn_hdim; i++)
                                W1[i] -= epsilon_t * local_dW1[i] / batch_count;
                            for (int i = 0; i < nn_hdim; i++)
                                b1[i] -= epsilon_t * local_db1[i] / batch_count;
                            for (int i = 0; i < nn_hdim * nn_output_dim; i++)
                                W2[i] -= epsilon_t * local_dW2[i] / batch_count;
                            for (int i = 0; i < nn_output_dim; i++)
                                b2[i] -= epsilon_t * local_db2[i] / batch_count;
                        }

                        free(z1);
                        free(a1);
                        free(z2);
                        free(probs);
                        free(delta3);
                        free(delta2);
                        free(local_dW1);
                        free(local_db1);
                        free(local_dW2);
                        free(local_db2);
                    } // end task
                } // end for b

                #pragma omp taskwait
            } // end single
        } // end parallel

        // Optional: rank 0 prints global loss (using global X,y)
        if (print_loss && rank == 0 && (epoch % 10 == 0)) {
            double loss = calculate_loss(W1, b1, W2, b2, nn_hdim);
            printf("[Rank 0] Loss after epoch %d: %.6f\n", epoch, loss);
        }
    } // end epoch loop

    // -------------------------------------
    // Save parameters on rank 0 (optional)
    // -------------------------------------
    if (rank == 0) {
        FILE *fw1 = fopen("output/W1.txt", "w");
        FILE *fb1 = fopen("output/b1.txt", "w");
        FILE *fw2 = fopen("output/W2.txt", "w");
        FILE *fb2 = fopen("output/b2.txt", "w");
        if (!fw1 || !fb1 || !fw2 || !fb2) {
            fprintf(stderr, "[Rank 0] Failed to open output files\n");
        } else {
            for (int i = 0; i < nn_input_dim * nn_hdim;       i++)
                fprintf(fw1, "%lf\n", W1[i]);
            for (int i = 0; i < nn_hdim;                      i++)
                fprintf(fb1, "%lf\n", b1[i]);
            for (int i = 0; i < nn_hdim      * nn_output_dim; i++)
                fprintf(fw2, "%lf\n", W2[i]);
            for (int i = 0; i < nn_output_dim;                i++)
                fprintf(fb2, "%lf\n", b2[i]);
            fclose(fw1);
            fclose(fb1);
            fclose(fw2);
            fclose(fb2);
            printf("[Rank 0] Weights saved to W1.txt, b1.txt, W2.txt, b2.txt\n");
        }
    }

    free(W1);
    free(W2);
    free(b1);
    free(b2);
}
