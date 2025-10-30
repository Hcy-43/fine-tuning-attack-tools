DATA_ROOT=data/tool_dataset/beavertails/data/330k/test/filtered/random

ratios=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)

for ratio in "${ratios[@]}"; do
    CUDA_VISIBLE_DEVICES=1,3 python -m src.sft train \
        --model_path models/Llama-2-7b-chat-hf \
        --data_path $DATA_ROOT/ratio_$ratio.csv \
        --output_dir ft_models/random/ratio_$ratio
done