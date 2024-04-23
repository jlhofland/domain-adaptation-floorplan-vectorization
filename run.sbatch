#!/bin/bash

# DEFINE SBATCH VARIABLES
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=1-10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 

# Check sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi 


# GET MODULES AND LOAD CUDA
module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.4 cudnn/12-8.9.1.23

# CREATE LOCAL MEMORY FOLDER
tmpdir="/dev/shm/${USER}/${SLURM_JOBID}"
mkdir --parents "$tmpdir"

# TEMP FOLDER PRINT
echo "Temporary folder: $(hostname --short):${tmpdir}"

# DEFINE DATA DIRECTORY
datadir="/tudelft.net/staff-umbrella/StudentsCVlab/${USER}/"

# CLEAN LOCAL FUNCTION
function clean_up { 
  rm --recursive --force "$tmpdir" && echo "Clean up of $tmpdir completed successfully."
  exit
}

# CLEAN LOCAL FOLDER ON EXIT
trap 'clean_up' EXIT 

# COPY DATA FROM $datadir TO $tmpdir/
rsync -av --exclude='.conda' "$datadir" "$tmpdir"/

# ACTIVATE CONDA ENVIRONMENT
conda activate fp-vector

# EXPORT WANDB_API_KEY
export WANDB_API_KEY="2a609c086839dc7fbac9d210a9b81c6b748f5994"

# For multi-GPU, don't use P2P as it hangs
export NCCL_P2P_DISABLE=1

# Debug python version
python --version

# RUN FILE
srun python train.py dataset.files.root=$tmpdir/ dataset.num_workers=32

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
