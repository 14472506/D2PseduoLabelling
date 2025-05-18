# Use the official Anaconda image as a base
FROM continuumio/anaconda3

# Set the working directory inside the container
WORKDIR /workspace

# Install kmod non-interactively
RUN apt-get update && apt-get install -y kmod

# Install the NVIDIA driver
RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.171.04/NVIDIA-Linux-x86_64-535.171.04.run
RUN sh ./NVIDIA-Linux-x86_64-535.171.04.run -s --no-kernel-module

# Set environment variables
ENV PATH /opt/conda/bin:$PATH

# Install NVIDIA libraries
RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Create the Detectron2 environment and install base dependencies
RUN conda create --name detectron2 python=3.9 -y

# Activate the environment and install PyTorch and related dependencies
RUN conda run -n detectron2 pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# Install additional dependencies using conda
RUN conda run -n detectron2 conda install -c conda-forge pybind11 -y
RUN conda run -n detectron2 conda install -c conda-forge gxx_linux-64 -y
RUN conda run -n detectron2 conda install -c anaconda gcc_linux-64 -y
RUN conda run -n detectron2 conda upgrade -c conda-forge --all -y
RUN conda run -n detectron2 conda install -c conda-forge libstdcxx-ng -y
RUN conda run -n detectron2 conda install -c conda-forge gxx_linux-64=10.3.0 -y

# Install Detectron2 and other Python packages
#RUN conda run -n detectron2 pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN conda run -n detectron2 pip install pillow==9.5.0 opencv-python-headless

# Set the default command to activate the conda environment
CMD ["bash"]
