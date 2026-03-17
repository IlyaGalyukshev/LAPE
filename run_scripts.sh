#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id>"
    echo "Example: $0 google/gemma-3-12b-it"
    exit 1
fi

export MODEL_ID="$1"
export MODEL_PATH="${MODEL_PATH:-/hf_models}"
echo "=== Running all scripts for MODEL_ID=${MODEL_ID} ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="

python3 scripts/lape.py
python3 scripts/tokenizer.py
python3 scripts/logit.py
python3 scripts/attention.py
python3 scripts/divercity.py
python3 scripts/evaluate.py

echo "=== All done for MODEL_ID=${MODEL_ID} ==="
