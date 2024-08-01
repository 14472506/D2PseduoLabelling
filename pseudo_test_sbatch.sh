#!/bin/bash
#SBATCH --job-name=ps_training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=output_%j.log

# Load required modules (if any)
# module load docker

# Load docker container
docker load -i brhurst.tar

# Run docker container
docker run --rm --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm -v $(pwd):/workspace brhurst:latest bash -c "
# Activate conda environment and run the commands
source activate detectron2 && \
yes | pip uninstall detectron2 && \
cd /workspace/detectron2 && \
python -m pip install -e . && \
cd /workspace && \
bash configs/pseudo_labeling/bash_scripts/pseudo_labeling.bash
"


