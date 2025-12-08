
# ======================================================================
# Makefile for Neural Network Training
# ======================================================================
# This Makefile supports 4 compilation modes:
#   1. SEQUENTIAL (default)
#   2. OpenMP
#   3. MPI
#   4. Hybrid (MPI + OpenMP)
# 
# Uncomment the appropriate section below for your desired mode
# ======================================================================

# ------------------------------
# MODE 1: SEQUENTIAL (DEFAULT)
# ------------------------------
# Single-threaded execution
# No parallelization
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall

# ------------------------------
# MODE 2: OpenMP
# ------------------------------
# Shared-memory parallelization
# To use: Uncomment the lines below and comment out the sequential section
#
# CC = gcc
# CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall

# ------------------------------
# MODE 3: MPI
# ------------------------------
# Distributed-memory parallelization
# To use: Uncomment the lines below and comment out the sequential section
#
# CC = mpicc
# CFLAGS = -O3 -march=native -ffast-math -Wall

# ------------------------------
# MODE 4: HYBRID (MPI + OpenMP)
# ------------------------------
# Combined distributed + shared memory parallelization
# To use: Uncomment the lines below and comment out the sequential section
#
# CC = mpicc
# CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall

# ======================================================================
# Build Rules (same for all modes)
# ======================================================================

OBJ = main.o model.o utils.o
TARGET = mlp

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -lm -o $(TARGET)

main.o: main.c model.h utils.h
	$(CC) $(CFLAGS) -c main.c

model.o: model.c model.h utils.h
	$(CC) $(CFLAGS) -c model.c

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c

clean:
	rm -f *.o $(TARGET)
