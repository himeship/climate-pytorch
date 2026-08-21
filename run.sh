#!/bin/bash
#PBS -N pytorch_ddp_job
#PBS -q normal
#PBS -P 90000001
#PBS -l select=2:ncpus=128:ngpus=4:mem=440G
#PBS -l walltime=3:00:00
#PBS -j oe
#PBS -o ddp_output.log

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load modules
module purge
module load PrgEnv-cray
module load cuda/12.8.1
module load python/3.10.9

export MPICH_GPU_SUPPORT_ENABLED=1

# Load Python env
source /home/users/astar/ares/deshp/optimization/pytorch-env/bin/activate

# Setup Master Address and Port for DDP
NODES=($(cat $PBS_NODEFILE | sort -u))
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500

export OMP_NUM_THREADS=1

START_TIME=$(date +%s)
echo "PyTorch Job Started at: $(date)"

export MPICH_CPU_BINDING=numa

# Launch PyTorch DDP via torchrun for 4 local GPUs
time mpiexec -np $NNODES -npernode 1 -hostfile $PBS_NODEFILE \
    torchrun --nnodes=$NNODES --nproc_per_node=4 --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_traj.py

END_TIME=$(date +%s)
echo "PyTorch Job Ended at: $(date)"

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "Total PyTorch Execution Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"