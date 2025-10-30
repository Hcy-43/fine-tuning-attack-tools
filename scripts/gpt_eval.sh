#!/usr/bin/env bash
set -e  # 任一指令失敗就停掉，避免浪費 GPU 時間
start=$(date +%s)

# === 1. 實驗組合 ===

seed=20
export CUDA_VISIBLE_DEVICES=1


ratios=(0.0 0.05 0.1 0.15 0.2 0.25 0.3)
# ratios=(0.3)
# === 2. 逐組實驗跑起來 ===
for ratio in "${ratios[@]}"; do
  # DATA_PATH="${data_paths[$idx]}"
  SAVE_NAME="random/llama2-7b-hf_ratio_${ratio}"
  # DATASET="${dataset[$idx]}"

  echo "🚀 開始實驗 $((idx+1)) / ${#ratios[@]}：$SAVE_NAME"

  # === 3. Safety evaluation ===
  mkdir -p "evaluation_outputs/$SAVE_NAME/GPT_result"

  for i in {1..11}; do

    python src/gpt4_eval.py \
      --input_file "evaluation_results/$SAVE_NAME/category_${i}_results.jsonl" \
      --output_file "evaluation_outputs/$SAVE_NAME/GPT_result/category_${i}_results.jsonl"
  done
  python src/score_summary.py evaluation_outputs/$SAVE_NAME/GPT_result
done

end=$(date +%s)
echo "⏱️ 全部實驗完成，總耗時 $((end - start)) 秒。"

