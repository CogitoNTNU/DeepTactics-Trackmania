#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="DQN_training"
#SBATCH --time=00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
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

# Load environment variables from .env file (ignore comments and empty lines)
if [ -f "${WORKDIR}/.env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # Automatically export all variables
    source <(grep -v '^#' ${WORKDIR}/.env | grep -v '^$')
    set +a
else
    echo "Warning: .env file not found. W&B logging may not work."
fi

# Add project root to Python path so 'src' can be found as a module
export PYTHONPATH="${WORKDIR}:${PYTHONPATH}"
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Install/update dependencies
echo "Installing/updating dependencies..."
pip install -r requirements.txt

echo "Installed packages:"
pip freeze

# Check GPU info
echo "================================================"
echo "GPU Information:"
echo "================================================"
nvidia-smi
echo ""
echo "PyTorch CUDA availability:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "================================================"
echo ""

# Run the training script
python main.py

# Deactivate the environment
conda deactivate

echo "Job finished"
