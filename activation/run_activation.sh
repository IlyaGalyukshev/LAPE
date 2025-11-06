#!/bin/bash
# Script to run activation analysis on a model

# Define parameters
LANG1="crh_Latn"
LANG2="eng"
MODEL="mistralai/Mistral-7B-Instruct-v0.2"
MODEL="Tweeties/tweety-tatar-base-7b-2024-v1"

MODEL_NAME=$(echo $MODEL | rev | cut -d'/' -f1 | rev | cut -d'-' -f1)

OUTPUT_JSON="../activation_results/${LANG1}_${LANG2}_${MODEL_NAME}.json"
OUTPUT_PTH="../activation_results/${LANG1}_${LANG2}_${MODEL_NAME}.pth"

python3 activation.py \
    --csv ../data/parallel_corpora/${LANG1}_${LANG2}.csv \
    --lang1_col ${LANG1} \
    --lang2_col ${LANG2} \
    --model "${MODEL}" \
    --batch_size 4 \
    --max_length 512 \
    --limit 0 \
    --device cuda \
    --filter_rate 0.95 \
    --top_rate 0.01 \
    --activation_bar_ratio 0.95 \
    --save_json ${OUTPUT_JSON} \
    --save_mask_pth ${OUTPUT_PTH}
