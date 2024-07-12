#!/bin/bash
###############################################################################
# SET TRAIN OR TEST MODE
###############################################################################
###
###
MODE="train"  # Set to "train" or "test"
###
###
###############################################################################
# CONFIGURE TRAIN AND TEST PARAMS 
###############################################################################
# Define default values for the parameters
CONFIG_FILE="configs/pseudo_labeling/config_files/test_2.yaml"
USE_GPU=0
ITERS=70350
BURN_IN_ITERS=0
METRIC_THRESHOLD=0.30

# Define lists of weights and output directories
TRAIN_WEIGHTS=(
    "outputs/baseline/TEST_1/best_model.pth"
    #"outputs/New_DS_Baseline/TEST_2/best_model.pth"
    #"outputs/New_DS_Baseline/TEST_3/best_model.pth"
)
TRAIN_DATASET="('summerschool_val',)"

TEST_WEIGHTS=(
    "outputs/SSBS1PS_999_long/TEST_1/best_model.pth"
    #"outputs/No_Burn_in_040/TEST_2/best_model.pth"
    #"outputs/No_Burn_in_040/TEST_3/best_model.pth"
)
TEST_DATASET="('summerschool_val',)"

OUTPUT_DIRS=(
    "outputs/SSBS1PS_999_long/TEST_1"
    #"outputs/No_Burn_in_040/TEST_2"
    #"outputs/No_Burn_in_040/TEST_3"
)

###############################################################################
# SET TRAIN OR TEST WEIGHT AND DATASET CONFIG
###############################################################################
if [ "$MODE" = "train" ]; then
    WEIGHTS=("${TRAIN_WEIGHTS[@]}")
    DATASET="$TRAIN_DATASET"
elif [ "$MODE" = "test" ]; then
    WEIGHTS=("${TEST_WEIGHTS[@]}")
    DATASET="$TEST_DATASET"
else
    echo "Unknown mode: $MODE. Use 'train' or 'test'. Exiting."
    exit 1
fi

###############################################################################
# CHECK SETTINGS
###############################################################################
# Ensure that weights and output directories are of the same length
if [ "${#WEIGHTS[@]}" -ne "${#OUTPUT_DIRS[@]}" ]; then
    echo "The number of weights and output directories must match. Exiting."
    exit 1
fi

###############################################################################
# RUN
###############################################################################
# Run the Python script with the specified arguments
for i in "${!WEIGHTS[@]}"; do
    WEIGHT="${WEIGHTS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

    if [ "$MODE" = "train" ]; then
        echo "Training with weight: $WEIGHT, output directory: $OUTPUT_DIR"
        python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
            OUTPUT_DIR $OUTPUT_DIR \
            MODEL.WEIGHTS $WEIGHT \
            SOLVER.MAX_ITER $ITERS \
            DATASETS.TEST "$DATASET" \
            PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
            PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD
    elif [ "$MODE" = "test" ]; then
        echo "Testing with weight: $WEIGHT, output directory: $OUTPUT_DIR"
        python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
            --eval \
            OUTPUT_DIR $OUTPUT_DIR \
            MODEL.WEIGHTS $WEIGHT \
            SOLVER.MAX_ITER $ITERS \
            DATASETS.TEST "$DATASET" \
            PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
            PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD
    else
        echo "Unknown mode: $MODE. Use 'train' or 'test'. Exiting."
        exit 1
    fi
done
