#!/bin/bash
#SBATCH --job-name=mpi_tcp_test
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH --output=mpi_tcp_%j.log

module load OpenMPI/4.1.1-GCC-11.2.0

# =============================
# FORCE MPI TO USE TCP/IP ONLY
# =============================
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_pml=ob1
export OMPI_MCA_mtl=^ofi
export OMPI_MCA_btl_openib_allow_ib=0
export OMPI_MCA_btl_openib_if_include=^mlx5_0,mlx5_1

cd $SLURM_SUBMIT_DIR

# Retrieve hostnames
NODE1=$(scontrol show hostname $SLURM_JOB_NODELIST | head -1)
NODES_4=$(scontrol show hostname $SLURM_JOB_NODELIST | head -4 | paste -sd,)
NODES_16=$(scontrol show hostname $SLURM_JOB_NODELIST | head -16 | paste -sd,)

echo "=========================================="
echo "MPI SCALABILITY TEST - TCP/IP MODE"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="
echo ""

# Scenario 1: 4 processes on 1 node
echo ">>> Scenario 1: 4 processes / 1 node"
mpirun -np 4 --host $NODE1 --oversubscribe ./mlp
echo ""

# Scenario 2: 4 processes on 4 nodes
echo ">>> Scenario 2: 4 processes / 4 nodes"
mpirun -np 4 --host $NODES_4 --map-by ppr:1:node ./mlp
echo ""

# Scenario 3: 16 processes on 1 node
echo ">>> Scenario 3: 16 processes / 1 node"
mpirun -np 16 --host $NODE1 --oversubscribe ./mlp
echo ""

# Scenario 4: 16 processes on 4 nodes
echo ">>> Scenario 4: 16 processes / 4 nodes"
mpirun -np 16 --host $NODES_4 --map-by ppr:4:node --oversubscribe ./mlp
echo ""

# Scenario 5: 16 processes on 16 nodes
echo ">>> Scenario 5: 16 processes / 16 nodes"
mpirun -np 16 --host $NODES_16 --map-by ppr:1:node ./mlp
echo ""

echo "=========================================="
echo "All Scenarios Complete"
echo "=========================================="
