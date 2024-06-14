#!/bin/bash

# Define default values for the parameters
CONFIG_FILE="configs/pseudo_labeling/config_files/test_2.yaml"
USE_GPU=0
ITERS=72050
BURN_IN_ITERS=0
METRIC_THRESHOLD=0.40
OUTPUT_DIR_1="outputs/No_Burn_in_040/TEST_1"
OUTPUT_DIR_2="outputs/No_Burn_in_040/TEST_2"
OUTPUT_DIR_3="outputs/No_Burn_in_040/TEST_3"
WEIGHTS="NEW_DS_BASE_WEIGHTS.pth"

# Print the parameters for verification
#echo "Using the following parameters:"
#echo "CONFIG_FILE: $CONFIG_FILE"
#echo "USING GPU: $USE_GPU"
#echo "OUTPUT DIR: $OUTPUT_DIR"


# Run the Python training script with the specified arguments
python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
    OUTPUT_DIR $OUTPUT_DIR_1 \
    MODEL.WEIGHTS $WEIGHTS \
    SOLVER.MAX_ITER $ITERS \
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
    PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD

python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
    OUTPUT_DIR $OUTPUT_DIR_2 \
    MODEL.WEIGHTS $WEIGHTS \
    SOLVER.MAX_ITER $ITERS \
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
    PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD

python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
    OUTPUT_DIR $OUTPUT_DIR_3 \
    MODEL.WEIGHTS $WEIGHTS \
    SOLVER.MAX_ITER $ITERS \
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
    PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD
