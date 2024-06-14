#!/bin/bash

# Define default values for the parameters
CONFIG_FILE="configs/pseudo_labeling/config_files/test_2.yaml"
USE_GPU=1
OUTPUT_DIR="outputs/DEV_DELETE_THIS_MORE/TEST_2"

# Print the parameters for verification
echo "Using the following parameters:"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "USING GPU: $USE_GPU"
echo "OUTPUT DIR: $OUTPUT_DIR"


# Run the Python training script with the specified arguments
python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR_1 
python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR_2 
python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE OUTPUT_DIR $OUTPUT_DIR_3 