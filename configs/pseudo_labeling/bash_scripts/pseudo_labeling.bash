#!/bin/bash
###############################################################################
# SET TRAIN OR TEST MODE
###############################################################################
MODE="test"  # Set to "train" or "test"

###############################################################################
# CONFIGURE TRAIN AND TEST PARAMS 
###############################################################################
# Define default values for the parameters
CONFIG_FILE="configs/pseudo_labeling/config_files/test_2.yaml"
# All training params
USE_GPU=0
ITERS=5556
TRAIN_PERC=100
IMS_PER_BATCH=16
EVAL_PERIOD=112
NUM_CLASSES=1

# Pseudo labeling conditional setup
PRE_TRAIN=false
PRE_TRAIN_ITERS=0
BURN_IN=true
BURN_IN_ITERS=0

# Pseudo labeling params
METRIC_THRESHOLD=0.50
CLASS_THRESHOLD=0.5
EMA_UPDATE=20
EMA_KEEP_RATE=0.999
METRIC_USE="static"
METRIC_OFFSET=0.05
 

# Define lists of weights and output directories
TRAIN_WEIGHTS=(
    "outputs/ps_dev_testing/mrcnn_baseline/100_epoch_bs16/best_model.pth"
    #"outputs/New_DS_Baseline/TEST_2/best_model.pth"
    #"outputs/New_DS_Baseline/TEST_3/best_model.pth"
)

# burn in student weights
BURN_IN_WEIGHTS="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

TRAIN_DATASET="('jersey_train',)"
VAL_DATASET="('jersey_val',)"

TEST_WEIGHTS=(
    "outputs/ps_dev_testing/pseudo_labeling/burn_in_to_dist_test_2/distillation_best_model.pth"
    #"outputs/No_Burn_in_040/TEST_2/best_model.pth"
    #"outputs/No_Burn_in_040/TEST_3/best_model.pth"
)
TEST_DATASET="('jersey_test',)"

OUTPUT_DIRS=(
    "outputs/ps_dev_testing/pseudo_labeling/burn_in_to_dist_test_2"
    #"outputs/No_Burn_in_040/TEST_2"
    #"outputs/No_Burn_in_040/TEST_3"
)

###############################################################################
# SET TRAIN OR TEST WEIGHT AND DATASET CONFIG
###############################################################################
if [ "$MODE" = "train" ]; then
    WEIGHTS=("${TRAIN_WEIGHTS[@]}")
    DATASET="$VAL_DATASET"
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
# Convert boolean parameters to lowercase
###############################################################################
if [ "$PRE_TRAIN" = true ]; then
    PRE_TRAIN_BOOL="True"
else
    PRE_TRAIN_BOOL="False"
fi

if [ "$BURN_IN" = true ]; then
    BURN_IN_BOOL="True"
else
    BURN_IN_BOOL="False"
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
        if [ -z "$WEIGHT" ]; then
            python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
                OUTPUT_DIR $OUTPUT_DIR \
                MODEL.WEIGHTS $WEIGHT \
                SOLVER.MAX_ITER $ITERS \
                SOLVER.IMS_PER_BATCH $IMS_PER_BATCH \
                DATASETS.TEST "$DATASET" \
                DATASETS.TRAIN "$TRAIN_DATASET" \
                PSEUDO_LABELING.TRAIN_PERC $TRAIN_PERC \
                PSEUDO_LABELING.PRE_TRAIN $PRE_TRAIN_BOOL \
                PSEUDO_LABELING.PRE_TRAIN_ITERS $PRE_TRAIN_ITERS \
                PSEUDO_LABELING.BURN_IN $BURN_IN_BOOL \
                PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
                PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS $BURN_IN_WEIGHTS \
                PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD \
                PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD $CLASS_THRESHOLD \
                PSEUDO_LABELING.PSEUDO_UPDATE_FREQ $EMA_UPDATE \
                PSEUDO_LABELING.EMA_KEEP_RATE $EMA_KEEP_RATE \
                PSEUDO_LABELING.METRIC_USE $METRIC_USE \
                PSEUDO_LABELING.METRIC_OFFSET $METRIC_OFFSET \
                TEST.EVAL_PERIOD $EVAL_PERIOD \
                MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES
        else
            python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
                OUTPUT_DIR $OUTPUT_DIR \
                MODEL.WEIGHTS $WEIGHT \
                SOLVER.MAX_ITER $ITERS \
                SOLVER.IMS_PER_BATCH $IMS_PER_BATCH \
                DATASETS.TEST "$DATASET" \
                DATASETS.TRAIN "$TRAIN_DATASET" \
                PSEUDO_LABELING.TRAIN_PERC $TRAIN_PERC \
                PSEUDO_LABELING.PRE_TRAIN $PRE_TRAIN_BOOL \
                PSEUDO_LABELING.PRE_TRAIN_ITERS $PRE_TRAIN_ITERS \
                PSEUDO_LABELING.BURN_IN $BURN_IN_BOOL \
                PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
                PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS $BURN_IN_WEIGHTS \
                PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD \
                PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD $CLASS_THRESHOLD \
                PSEUDO_LABELING.PSEUDO_UPDATE_FREQ $EMA_UPDATE \
                PSEUDO_LABELING.EMA_KEEP_RATE $EMA_KEEP_RATE \
                PSEUDO_LABELING.METRIC_USE $METRIC_USE \
                PSEUDO_LABELING.METRIC_OFFSET $METRIC_OFFSET \
                TEST.EVAL_PERIOD $EVAL_PERIOD \
                MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES
        fi
    elif [ "$MODE" = "test" ]; then
        echo "Testing with weight: $WEIGHT, output directory: $OUTPUT_DIR"
        python pseudo_labeling_train_net.py --use_gpu $USE_GPU --config $CONFIG_FILE  \
            --eval \
            OUTPUT_DIR $OUTPUT_DIR \
            MODEL.WEIGHTS $WEIGHT \
            SOLVER.MAX_ITER $ITERS \
            SOLVER.IMS_PER_BATCH $IMS_PER_BATCH \
            DATASETS.TEST "$DATASET" \
            DATASETS.TRAIN "$TRAIN_DATASET" \
            PSEUDO_LABELING.TRAIN_PERC $TRAIN_PERC \
            PSEUDO_LABELING.PRE_TRAIN $PRE_TRAIN_BOOL \
            PSEUDO_LABELING.PRE_TRAIN_ITERS $PRE_TRAIN_ITERS \
            PSEUDO_LABELING.BURN_IN $BURN_IN_BOOL \
            PSEUDO_LABELING.BURN_IN_ITERS $BURN_IN_ITERS \
            PSEUDO_LABELING.BURN_IN_STUDENT_WEIGHTS $BURN_IN_WEIGHTS \
            PSEUDO_LABELING.METRIC_THRESHOLD $METRIC_THRESHOLD \
            PSEUDO_LABELING.CLASS_CONFIDENCE_THRESHOLD $CLASS_THRESHOLD \
            PSEUDO_LABELING.PSEUDO_UPDATE_FREQ $EMA_UPDATE \
            PSEUDO_LABELING.EMA_KEEP_RATE $EMA_KEEP_RATE \
            PSEUDO_LABELING.METRIC_USE $METRIC_USE \
            PSEUDO_LABELING.METRIC_OFFSET $METRIC_OFFSET \
            TEST.EVAL_PERIOD $EVAL_PERIOD \
            MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES
    else
        echo "Unknown mode: $MODE. Use 'train' or 'test'. Exiting."
        exit 1
    fi
done



