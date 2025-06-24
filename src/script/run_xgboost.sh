#!/bin/bash

DATASETS=("nhanes" "support")
FUNCTION=("normal" "extreme" "logistic")

OUTPUT_BASE="results"

mkdir -p "$OUTPUT_BASE"

for DATASET in "${DATASETS[@]}"; do
    for FUNC in "${FUNCTION[@]}"; do
        echo "Running experiment for dataset: $DATASET with function: $FUNC"

        OUTPUT_FILE="${OUTPUT_BASE}/xgboost_${FUNC}_${DATASET}.csv"
        MODEL_DIR="${OUTPUT_BASE}/xgboost_${FUNC}_${DATASET}"
        
        # Create model directory
        mkdir -p "$MODEL_DIR"
        
        # Run the experiment
        python3 src/main_experiment.py \
            --parameter="xgboostaft" \
            --dataset="$DATASET" \
            --function="$FUNC" \
            --no-is_grid \
            --is_cv \
            --n_splits=5 \
            --n_tries=10 \
            --n_models=5 \
            --path-res="$OUTPUT_FILE" \
            --path="$MODEL_DIR"
        
        # Check if command succeeded
        if [ $? -eq 0 ]; then
            echo "Successfully processed $DATASET"
        else
            echo "ERROR: Failed to process $DATASET" >&2
        fi
    done
done

echo "All experiments completed!"