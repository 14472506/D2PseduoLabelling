#!/bin/bash
#SBATCH --job-name=ps_training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=sbatch_logs/cityscapes_gd_Distillation_continued_05_%j.log
#SBATCH --no-requeue

# Load required modules (if any)
# module load docker

# Load docker container
docker build -t brhurst:latest .
#docker load -i setup/brhurst.tar

# Run docker container
docker run --gpus all -v $(pwd):/workspace brhurst:latest bash -c '
# Activate conda environment and run the commands
source activate detectron2 && \
# yes | pip uninstall detectron2 && \
cd /workspace/detectron2 && \
python -m pip install -e . && \
cd /workspace && \
export CUDA_HOME=/usr/local/cuda && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 && \
export PATH=$PATH:$CUDA_HOME/bin && \
apt-get update && \
apt-get upgrade -y && \
dpkg -i cuda-repo-ubuntu2004-12-5-local_12.5.1-555.42.06-1_amd64.deb && \
cp /var/cuda-repo-ubuntu2004-12-5-local/cuda-3AA8F848-keyring.gpg /usr/share/keyrings && \
apt-get update && \
apt-get -y install cuda-toolkit-12-5 && \
pip install -r mask2former/requirements.txt && \
cd mask2former/modeling/pixel_decoder/ops && \
sh make.sh && \
cd ../../../../ && \
pip uninstall numpy -y && \
pip install numpy==1.26.4 && \
configs/pseudo_labeling/bash_scripts/guid_dist/guid_dist_pseudo_labeling_1.bash'
# bash configs/pseudo_labeling/bash_scripts/guid_dist/guid_dist_pseudo_labeling_1.bash
# bash configs/pseudo_labeling/bash_scripts/m2f/m2f_pseudo_labeling_1.bash