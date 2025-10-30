ratios=(0.3)
export CUDA_VISIBLE_DEVICES=5
for ratio in "${ratios[@]}"; do
    # DATA_PATH="${data_paths[$idx]}"
    SAVE_NAME="pca_lr/llama2-7b-hf_ratio_${ratio}"
    for i in {8..11}; do
        python src/inference.py \
                --model_path "models/Llama-2-7b-chat-hf" \
                --lora_path "ft_models/pca_lr/ratio_$ratio/final" \
                --input_file "data/evaluation_datasets/HEx-PHI-full/category_${i}.csv" \
                --output_file "evaluation_results/$SAVE_NAME/category_${i}_results.jsonl" \
                --max_new_tokens 512
    done
done