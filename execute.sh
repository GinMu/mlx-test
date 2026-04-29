#!/bin/bash

SCRIPT="mlx-tool-calling.py"

models=(
  "Qwen3.5-4B-MLX-8bit"
  "Qwen3.5-4B-MLX-4bit"
)

questions=(
  "生成钱包"
  "生成一个钱包"
  "generate a wallet"
  "generate wallet"
  "create a wallet"
  "create wallet"
  "请帮我产生一个钱包"
  "现在几点了"
  "现在什么时间"
  "今天什么日期"
  "current date"
  "what is the date now"
)

for model in "${models[@]}"; do
  echo "========================================"
  echo "Model: $model"
  echo "========================================"
  for q in "${questions[@]}"; do
    echo "========== Question: $q =========="
    python3 "$SCRIPT" --model "$model" --question "$q"
    echo ""
  done
  echo ""
done
