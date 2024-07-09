# Use the official Anaconda image as a base
FROM continuumio/anaconda3

# Set the working directory inside the container
WORKDIR /workspace

# Install kmod non-interactively
RUN apt-get update && apt-get install -y kmod

RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.171.04/NVIDIA-Linux-x86_64-535.171.04.run
RUN sh ./NVIDIA-Linux-x86_64-535.171.04.run -s --no-kernel-module

# Set environment variables
ENV PATH /opt/conda/bin:$PATH

# Install NVIDIA libraries (adjust versions as needed)
RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Create the Detectron2 environment and install dependencies
RUN conda create --name detectron2 python=3.9 -y && \
    conda run -n detectron2 pip install torch torchvision && \
    conda run -n detectron2 conda install -c conda-forge pybind11 -y && \
    conda run -n detectron2 conda install -c conda-forge gxx_linux-64 -y && \
    conda run -n detectron2 conda install -c anaconda gcc_linux-64 -y && \
    conda run -n detectron2 conda upgrade -c conda-forge --all -y && \
    conda run -n detectron2 pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6' && \
    conda run -n detectron2 pip install pillow==9.5.0 && \
    conda run -n detectron2 pip install opencv-python-headless

# Create the Ultralytics envrionment and install dependencies
RUN conda create --name ultralytics python=3.9 -y && \
    conda run -n ultralytics pip install torch torchvision && \
    conda run -n ultralytics pip install ultralytics

# Set the default command to activate the conda environment
CMD ["bash", "-c", "source activate detectron2 && bash"]



