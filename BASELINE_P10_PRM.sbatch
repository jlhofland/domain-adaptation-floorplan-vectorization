#!/bin/bash

# DEFINE SBATCH VARIABLES (for 175G data folder)
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40:1
#SBATCH --tmp=500G

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 

# Check sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi 


# GET MODULES AND LOAD CUDA
module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.4 cudnn/12-8.9.1.23

# CREATE LOCAL MEMORY FOLDER
tmpdir="/tmp/${USER}/${SLURM_JOBID}"
mkdir -p "$tmpdir/run"

# TEMP FOLDER PRINT
echo "Temporary folder: $(hostname --short):${tmpdir}"
echo "Run folder: $(hostname --short):${tmpdir}/run"

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
# OLD: rsync -av --exclude='.conda' "$datadir" "$tmpdir"/
rsync -av \
  --exclude='.conda' \
  --include='F1_original.png' \
  --include='F1_scaled.png' \
  --include='model.svg' \
  --include='*.txt' \
  --include='*/' \
  --exclude='*' \
  "$datadir" "$tmpdir"/

# ACTIVATE CONDA ENVIRONMENT
conda activate fp-vector

# EXPORT WANDB_API_KEY
export WANDB_API_KEY="e50eaeab2fc7883b49d009473985492b3c1eb24c"
export WANDB_DATA_DIR=$tmpdir

# For multi-GPU, don't use P2P as it hangs
export NCCL_P2P_DISABLE=1

# Debug python version
python --version

# RUN FILE
srun python train.py dataset.files.root=$tmpdir/ wandb.dir=$tmpdir/run dataset.num_workers=4 wandb.experiment_name=BASELINE_P20_PRM_400E train.precision="medium" dataset.files.train="train_hqa_hq_c.txt" dataset.files.val="val_hqa_hq_c.txt" wandb.weights=model-172x5zcy

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
