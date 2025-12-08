#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// ----------------------
// Outils mathématiques
// ----------------------

// Génère une variable aléatoire suivant une loi normale centrée réduite
double randn() {
    double u = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    double v = ((double) rand() + 1.0) / ((double) RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}
//Produit matriciel C = A * B
void matmul(double *A, double *B, double *C, int n, int m, int p) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            C[i*p + j] = 0.0;
            for (int k = 0; k < m; k++)
                C[i*p + j] += A[i*m + k] * B[k*p + j];
        }
}

void matmul_omp(double *A, double *B, double *C, int M, int N, int P) {

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {

            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * P + j];
            }

            C[i * P + j] = sum;
        }
    }
}




// Ajoute le biais b à chaque ligne de Z
void add_bias(double *Z, double *b, int n, int p) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            Z[i*p + j] += b[j];
}
void softmax(double *z, double *probs, int N, int C)
{
    for (int i = 0; i < N; i++) {

        double max_z = -1e9;
        for (int j = 0; j < C; j++)
            if (z[i*C + j] > max_z)
                max_z = z[i*C + j];

        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            probs[i*C + j] = exp(z[i*C + j] - max_z);
            sum += probs[i*C + j];
        }
        for (int j = 0; j < C; j++)
            probs[i*C + j] /= sum;
    }
}

void softmax(double *Z, double *P, int n, int p) {
    for (int i = 0; i < n; i++) {
        double sum_exp = 0.0;
        for (int j = 0; j < p; j++) {
            P[i*p + j] = exp(Z[i*p + j]);
            sum_exp += P[i*p + j];
        }
        for (int j = 0; j < p; j++)
            P[i*p + j] /= sum_exp;
    }
}

// Compte les lignes dans un fichier (nombre d'échantillons)
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Erreur ouverture fichier"); exit(1); }
    int count = 0; char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp)) count++;
    fclose(fp);
    return count;
}

// Lecture des données X (num_examples × input_dim)
void load_X(const char *filename, double *X, int num_examples, int input_dim) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Erreur ouverture X"); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_examples; i++) {
        for (int j = 0; j < input_dim; j++) {
            if (fscanf(fp, "%lf", &X[i*input_dim + j]) != 1) {
                fprintf(stderr, "Erreur lecture donnée X à la ligne %d colonne %d\n", i, j);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);
}

// Lecture des labels y (num_examples)
void load_y(const char *filename, int *y, int num_examples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror("Erreur ouverture y"); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_examples; i++) {
        if (fscanf(fp, "%d", &y[i]) != 1) {
            fprintf(stderr, "Erreur lecture donnée y à la ligne %d\n", i);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
}

    fclose(fp);
}
