#!/bin/bash
#PBS -N pytorch_ddp_job
#PBS -q normal
#PBS -P 59001008
#PBS -l select=1:ncpus=128:ngpus=4:mem=440G
#PBS -l walltime=04:00:00
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

# Launch PyTorch DDP via torchrun for 4 local GPUs
torchrun --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_SEA_120km.py