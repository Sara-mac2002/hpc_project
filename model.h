#ifndef MODEL_H
#define MODEL_H

// ----------------------
// Global Variables
// ----------------------
extern int num_examples;
extern int nn_input_dim;
extern int nn_output_dim;
extern double reg_lambda;
extern double epsilon;
extern double *X;
extern int *y;

// ----------------------
// SEQUENTIAL (default)
// ----------------------
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim, int activation_mode);
void build_model_sequential(int nn_hdim, int num_passes, int batch_size, int decay_mode, int activation_mode, int print_loss);

// ----------------------
// OpenMP
// ----------------------
void build_model_omp(int nn_hdim, int num_passes, int batch_size, int decay_mode, int activation_mode, int print_loss);
void build_model_task(int nn_hdim, int num_passes, int batch_size, int decay_mode, int activation_mode, int print_loss);

// ----------------------
// MPI
// ----------------------
double calculate_loss_mpi(double *W1, double *b1, double *W2, double *b2, int nn_hdim, double *X_local, int *y_local, int num_local, int activation_mode);
void build_model_mpi(int nn_hdim, int num_passes, int batch_size, int print_loss, int rank, int size, int activation_mode);

#endif