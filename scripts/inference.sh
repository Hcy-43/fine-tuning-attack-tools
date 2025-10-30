ratios=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)
export CUDA_VISIBLE_DEVICES=1
for ratio in "${ratios[@]}"; do
    # DATA_PATH="${data_paths[$idx]}"
    SAVE_NAME="random/llama2-7b-hf_ratio_${ratio}"
    for i in {1..11}; do
        python src/inference.py \
                --model_path "models/Llama-2-7b-chat-hf" \
                --lora_path "ft_models/random/ratio_$ratio/final" \
                --input_file "data/evaluation_datasets/HEx-PHI-full/category_${i}.csv" \
                --output_file "evaluation_results/$SAVE_NAME/category_${i}_results.jsonl" \
                --max_new_tokens 512
    done
done