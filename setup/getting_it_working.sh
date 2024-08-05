#!/bin/bash

# Install gnupg and software-properties-common
apt-get update
apt-get install -y gnupg software-properties-common

# Download the NVIDIA package repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add the NVIDIA package repository key
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

# Add the NVIDIA package repository
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Update package lists and install CUDA Toolkit 12.2
apt-get update
apt-get install -y cuda-toolkit-12-2

# Set up environment variables
echo 'export CUDA_HOME=/usr/local/cuda-12.2' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify the CUDA installation
nvcc --version





