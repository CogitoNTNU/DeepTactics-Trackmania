#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="DQN_training"
#SBATCH --time=03:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_combined.txt
#SBATCH --error=slurm_outputs/output_combined.err
#SBATCH --mail-user=ludvigho@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

ENV_PATH="/cluster/work/$(whoami)/DEEPTACTICS-TRACKMANIA"

# Load required modules
module purge
module load Anaconda3/2024.02-1
module load SWIG/4.1.1-GCCcore-12.3.0  # Required for Box2D compilation

# Create conda environment only if it doesn't exist
if [ ! -d "${ENV_PATH}" ]; then
    echo "Creating new conda environment..."
    conda create -y --prefix ${ENV_PATH} python=3.11
else
    echo "Conda environment already exists, skipping creation."
fi

# Activate the environment
source activate ${ENV_PATH}

# Load environment variables from .env file
if [ -f "${WORKDIR}/.env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat ${WORKDIR}/.env | xargs)
else
    echo "Warning: .env file not found. W&B logging may not work."
fi

# Install dependencies only if not already installed
if ! python -c "import gymnasium" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies already installed."
fi

echo "Installed packages:"
pip freeze

# Set up W&B API key (make sure this is set in your environment)
# You can set it permanently with: export WANDB_API_KEY="your_key_here"
# Or create a .env file and source it

# Run the training script
python main.py

# Deactivate the environment
conda deactivate

echo "Job finished"
