#!/bin/bash

# Layer sweep configuration
LAYERS=(-17 -18)
MODEL_PATH=models/vicuna-7b-1v.5
DATA_DIR=data/tool_dataset/beavertails
REPS_BASE_DIR=$DATA_DIR/reps/vicuna-7b-1v.5/30k/layers/train
RESULTS_FILE=results/vicuna-7b-1v.5/layer_sweep_results.csv

# Create results directory
mkdir -p results

# Remove existing results file if it exists (fresh start)
# rm -f $RESULTS_FILE

echo "Starting layer sweep across ${#LAYERS[@]} layers..."
echo "Results will be saved to: $RESULTS_FILE"
echo "================================================"
# ratios=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)
# Loop through each layer
for LAYER in "${LAYERS[@]}"; do
    echo ""
    echo "Processing Layer $LAYER..."
    echo "================================================"
    
    REP_DIR=$REPS_BASE_DIR/layer_$LAYER
    mkdir -p $REP_DIR
    
    # Extract representations for harmful examples
    echo "Extracting harmful representations..."
    CUDA_VISIBLE_DEVICES=0 python src/representations.py \
        --input_path=data/tool_dataset/beavertails/data/30k/train/harmful.csv \
        --output_path=$REP_DIR/harmful.npy \
        --model_path=$MODEL_PATH \
        --query_column=prompt \
        --batch_size=32 \
        --device=cuda \
        --layer=$LAYER \
        --only_query=False \
        --response_column=response
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract harmful representations for layer $LAYER"
        continue
    fi
    
    # Extract representations for benign examples
    echo "Extracting benign representations..."
    CUDA_VISIBLE_DEVICES=0 python src/representations.py \
        --input_path=data/tool_dataset/beavertails/data/30k/train/benign.csv\
        --output_path=$REP_DIR/benign.npy \
        --model_path=$MODEL_PATH \
        --query_column=prompt \
        --batch_size=32 \
        --device=cuda \
        --layer=$LAYER \
        --only_query=False \
        --response_column=response
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract benign representations for layer $LAYER"
        continue
    fi
    
    # Check if both files exist before training probe
    if [ ! -f "$REP_DIR/harmful.npy" ] || [ ! -f "$REP_DIR/benign.npy" ]; then
        echo "ERROR: Representation files not found for layer $LAYER"
        continue
    fi
    
    # # Train probe and save results
    echo "Training probe..."
    CUDA_VISIBLE_DEVICES=0 python src/probe.py \
        --safe_path $REP_DIR/benign.npy \
        --harmful_path $REP_DIR/harmful.npy \
        --cv 5 \
        --layer $LAYER \
        --output_file $RESULTS_FILE
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to train probe for layer $LAYER"
        continue
    fi
    
    echo "Layer $LAYER complete!"
done

echo ""
echo "================================================"
echo "Layer sweep complete!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To visualize results, run:"
echo "python src/plot_layer_sweep.py --input $RESULTS_FILE"