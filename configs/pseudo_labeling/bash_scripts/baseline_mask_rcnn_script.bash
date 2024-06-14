#!/bin/bash

# Define default values for the parameters
CONFIG_FILE="configs/pseudo_labeling/config_files/test_2.yaml"
USE_GPU=1
ITERS=144100
BURN_IN_ITERS=150000
OUTPUT_DIR_1="outputs/New_DS_Baseline/TEST_1"
OUTPUT_DIR_2="outputs/New_DS_Baseline/TEST_2"
OUTPUT_DIR_3="outputs/New_DS_Baseline/TEST_3"
WEIGHTS="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

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
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS

python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
    OUTPUT_DIR $OUTPUT_DIR_2 \
    MODEL.WEIGHTS $WEIGHTS \
    SOLVER.MAX_ITER $ITERS \
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS

python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
    OUTPUT_DIR $OUTPUT_DIR_3 \
    MODEL.WEIGHTS $WEIGHTS \
    SOLVER.MAX_ITER $ITERS \
    PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS
