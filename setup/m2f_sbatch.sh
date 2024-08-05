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
export CUDA_HOME=/usr/local/cuda && \
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 && \
export PATH=\$PATH:\$CUDA_HOME/bin && \
apt-get update && \
apt-get upgrade -y && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda-repo-ubuntu2004-12-5-local_12.5.1-555.42.06-1_amd64.deb && \
dpkg -i cuda-repo-ubuntu2004-12-5-local_12.5.1-555.42.06-1_amd64.deb && \
cp /var/cuda-repo-ubuntu2004-12-5-local/cuda-3AA8F848-keyring.gpg /usr/share/keyrings && \
apt-get update && \
apt-get -y install cuda-toolkit-12-5 && \
pip install -r mask2former/requirements.txt && \
cd mask2former/modeling/pixel_decoder/ops && \
sh make.sh && \
cd ../../../../ && \
bash configs/pseudo_labeling/bash_scripts/m2f_pseudo_labeling.bash"
"


