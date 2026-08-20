#!/bin/bash
#PBS -N pytorch_ddp_job
#PBS -q normal
#PBS -P 90000001
#PBS -l select=1:ncpus=128:ngpus=4:mem=440G
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o ddp_output.log

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Load modules
module purge
module load cuda/12.8.1
module load python/3.10.9

# Load Python env
source /home/users/astar/ares/deshp/optimization/pytorch-env/bin/activate

# Setup Master Address and Port for DDP
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

START_TIME=$(date +%s)
echo "PyTorch Job Started at: $(date)"

# Launch PyTorch DDP via torchrun for 4 local GPUs
time torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_traj.py

END_TIME=$(date +%s)
echo "PyTorch Job Ended at: $(date)"

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "Total PyTorch Execution Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"