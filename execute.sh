#!/bin/bash

SCRIPT="mlx-script.py"

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

for q in "${questions[@]}"; do
  echo "========== Question: $q =========="
  python3 "$SCRIPT" --question "$q"
  echo ""
done
