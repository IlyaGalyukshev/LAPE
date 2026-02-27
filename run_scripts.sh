#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id>"
    echo "Example: $0 google/gemma-3-12b-it"
    exit 1
fi

export MODEL_ID="$1"
echo "=== Running all scripts for MODEL_ID=${MODEL_ID} ==="

python lape.py
python tokenizer.py
python logit.py
python attention.py
python divercity.py
python evaluate.py

echo "=== All done for MODEL_ID=${MODEL_ID} ==="
